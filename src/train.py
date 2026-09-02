"""Train and evaluate the compact DeepONet on the prepared OperatorDataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# The project keeps optional wheels in work/pydeps so the host Python install is untouched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for OPTIONAL_DEPS in (PROJECT_ROOT / "work" / "pydeps", PROJECT_ROOT.parent.parent / "work" / "pydeps"):
    if OPTIONAL_DEPS.exists():
        sys.path.append(str(OPTIONAL_DEPS))

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from cfd_pretrain.common import count_parameters, read_json_attr, resolve_path, select_device, set_seed, write_json
from cfd_pretrain.dataset import H5OperatorDataset
from cfd_pretrain.metrics import field_metrics, mean_field_baseline
from cfd_pretrain.model import DeepONet


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1).expand_as(prediction)
    squared = (prediction - target) ** 2
    return squared.masked_select(valid > 0.5).mean()


def read_normalization(data_path: Path) -> dict[str, Any]:
    with h5py.File(data_path, "r") as handle:
        return read_json_attr(handle.attrs["normalization"])


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    losses: list[float] = []
    for batch in loader:
        branch = batch["branch"].to(device)
        trunk = batch["trunk"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        with torch.set_grad_enabled(train):
            prediction = model(branch, trunk)
            loss = masked_mse(prediction, target, mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def collect_split(model, loader, device):
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    for batch in loader:
        predictions.append(model(batch["branch"].to(device), batch["trunk"].to(device)).cpu().numpy())
        targets.append(batch["target"].numpy())
        masks.append(batch["mask"].numpy())
        case_ids.append(np.asarray(batch["case_id"]))
        frame_indices.append(np.asarray(batch["frame_index"]))
    return {
        "prediction": np.concatenate(predictions, axis=0),
        "target": np.concatenate(targets, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "case_id": np.concatenate(case_ids, axis=0),
        "frame_index": np.concatenate(frame_indices, axis=0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_metrics(data_path: Path, model, device, split: str, batch_size: int, normalization: dict[str, Any]):
    dataset = H5OperatorDataset(data_path, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values = collect_split(model, loader, device)
    field_mean = np.asarray(normalization["field_mean"], dtype=np.float32)
    field_std = np.asarray(normalization["field_std"], dtype=np.float32)
    metrics = field_metrics(values["prediction"], values["target"], values["mask"], field_mean, field_std)
    return metrics, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--data", default="processed/operator_dataset.h5")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "npu"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    data_path = resolve_path(args.data)
    seed = int(config.get("seed", 20260827))
    set_seed(seed)
    torch.set_num_threads(int(config["training"].get("num_threads", 1)))
    requested_device = args.device or config["training"].get("device", "auto")
    selected = select_device(requested_device)
    if isinstance(selected, tuple):
        device, device_audit = selected
    else:
        device, device_audit = selected, {"requested": requested_device, "resolved": str(selected), "fallback_reason": None}
    run_name = args.run_name or config["training"].get("run_name", "public_pretrain")
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / run_name
    log_dir = PROJECT_ROOT / "logs" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = H5OperatorDataset(data_path, "train")
    val_dataset = H5OperatorDataset(data_path, "val")
    test_dataset = H5OperatorDataset(data_path, "test")
    batch_size = int(config["training"].get("batch_size", 8))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model_config = {key: value for key, value in config["model"].items() if key != "name"}
    model_config.update(
        {
            "branch_dim": train_dataset.branch_dim,
            "trunk_dim": train_dataset.trunk_dim,
            "output_channels": train_dataset.output_channels,
        }
    )
    model = DeepONet(**model_config).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-6)),
    )
    normalization = read_normalization(data_path)
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    epochs = int(args.epochs or config["training"].get("epochs", 180))
    patience = int(config["training"].get("patience", 35))

    print(f"device={device}; model_parameters={parameter_count}; train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config(),
                    "config": config,
                    "normalization": normalization,
                    "device_audit": device_audit,
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "parameter_count": parameter_count,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch={epoch:04d} train={train_loss:.6g} val={val_loss:.6g} best={best_val:.6g}")
        if stale >= patience:
            print(f"early_stop epoch={epoch}; best_epoch={best_epoch}")
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "config": config,
            "normalization": normalization,
            "device_audit": device_audit,
            "epoch": history[-1]["epoch"],
            "best_val_loss": best_val,
            "parameter_count": parameter_count,
        },
        checkpoint_dir / "last.pt",
    )
    write_csv(log_dir / "training_history.csv", history, ["epoch", "train_loss", "val_loss"])
    write_csv(PROJECT_ROOT / "training_history.csv", history, ["epoch", "train_loss", "val_loss"])

    best_checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    best_model = DeepONet(**best_checkpoint["model_config"]).to(device)
    best_model.load_state_dict(best_checkpoint["model_state"])
    validation_metrics, val_values = evaluate_metrics(data_path, best_model, device, "val", batch_size, normalization)
    test_metrics, test_values = evaluate_metrics(data_path, best_model, device, "test", batch_size, normalization)
    # The baseline is the mean nondimensionalized field at each query point from training cases.
    with h5py.File(data_path, "r") as handle:
        train_target = handle["train/target"][:]
        train_mask = handle["train/mask"][:]
    baseline_model = mean_field_baseline(train_target, train_mask)
    baseline_val = field_metrics(
        np.broadcast_to(baseline_model[None, :, :], val_values["target"].shape),
        val_values["target"],
        val_values["mask"],
        np.asarray(normalization["field_mean"], dtype=np.float32),
        np.asarray(normalization["field_std"], dtype=np.float32),
    )
    baseline_test = field_metrics(
        np.broadcast_to(baseline_model[None, :, :], test_values["target"].shape),
        test_values["target"],
        test_values["mask"],
        np.asarray(normalization["field_mean"], dtype=np.float32),
        np.asarray(normalization["field_std"], dtype=np.float32),
    )
    field_mean = np.asarray(normalization["field_mean"], dtype=np.float32)
    field_std = np.asarray(normalization["field_std"], dtype=np.float32)
    uniform_inlet_model = ((np.asarray([1.0, 0.0], dtype=np.float32) - field_mean) / field_std).reshape(1, 1, -1)
    zero_field_model = ((np.asarray([0.0, 0.0], dtype=np.float32) - field_mean) / field_std).reshape(1, 1, -1)
    uniform_val = field_metrics(np.broadcast_to(uniform_inlet_model, val_values["target"].shape), val_values["target"], val_values["mask"], field_mean, field_std)
    uniform_test = field_metrics(np.broadcast_to(uniform_inlet_model, test_values["target"].shape), test_values["target"], test_values["mask"], field_mean, field_std)
    zero_val = field_metrics(np.broadcast_to(zero_field_model, val_values["target"].shape), val_values["target"], val_values["mask"], field_mean, field_std)
    zero_test = field_metrics(np.broadcast_to(zero_field_model, test_values["target"].shape), test_values["target"], test_values["mask"], field_mean, field_std)
    metrics_rows = []
    for split, kind, values in (
        ("val", "model", validation_metrics),
        ("val", "mean_field_baseline", baseline_val),
        ("val", "uniform_inlet_baseline", uniform_val),
        ("val", "zero_field_baseline", zero_val),
        ("test", "model", test_metrics),
        ("test", "mean_field_baseline", baseline_test),
        ("test", "uniform_inlet_baseline", uniform_test),
        ("test", "zero_field_baseline", zero_test),
    ):
        row = {"split": split, "method": kind}
        row.update(values)
        metrics_rows.append(row)
    metric_fields = ["split", "method", "field_rmse", "field_mae", "u_rmse", "u_mae", "u_relative_l2", "v_rmse", "v_mae", "v_relative_l2", "velocity_relative_l2", "valid_fraction"]
    write_csv(log_dir / "validation_metrics.csv", metrics_rows, metric_fields)
    write_csv(PROJECT_ROOT / "validation_metrics.csv", metrics_rows, metric_fields)
    summary = {
        "run_name": run_name,
        "checkpoint_best": str(checkpoint_dir / "best.pt"),
        "checkpoint_last": str(checkpoint_dir / "last.pt"),
        "best_epoch": best_epoch,
        "best_val_loss_model_space": best_val,
        "parameter_count": parameter_count,
        "device": device_audit,
        "validation": validation_metrics,
        "validation_baseline": baseline_val,
        "validation_uniform_inlet_baseline": uniform_val,
        "validation_zero_field_baseline": zero_val,
        "test": test_metrics,
        "test_baseline": baseline_test,
        "test_uniform_inlet_baseline": uniform_test,
        "test_zero_field_baseline": zero_test,
        "validation_better_than_uniform_inlet": bool(validation_metrics["field_rmse"] < uniform_val["field_rmse"]),
        "test_better_than_uniform_inlet": bool(test_metrics["field_rmse"] < uniform_test["field_rmse"]),
        "pretraining_task": "physical-parameter/time -> 2D velocity field on a regular query grid",
        "pressure_available": False,
    }
    write_json(log_dir / "summary.json", summary)
    write_json(PROJECT_ROOT / "reports" / "pretraining_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
