"""Train one leakage-safe DeepONet-v2 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for OPTIONAL_DEPS in (PROJECT_ROOT / "work" / "pydeps", PROJECT_ROOT.parent.parent / "work" / "pydeps"):
    if OPTIONAL_DEPS.exists():
        sys.path.append(str(OPTIONAL_DEPS))

import h5py
import torch
from torch.utils.data import DataLoader

from cfd_pretrain.common import count_parameters, read_json_attr, resolve_path, select_device, set_seed, write_json
from cfd_pretrain.config import load_config
from cfd_pretrain.dataset import H5OperatorDataset
from cfd_pretrain.model_v2 import build_model


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1).expand_as(target)
    squared = (prediction - target) ** 2
    return (squared * valid).sum() / valid.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        branch = batch["branch"].to(device)
        trunk = batch["trunk"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        loss = masked_mse(model(branch, trunk), target, mask)
        valid_count = float(mask.sum().item() * target.shape[-1])
        total += float(loss.item()) * valid_count
        count += int(valid_count)
    return total / max(count, 1)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "val_loss", "learning_rate", "epoch_seconds"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v2_e1.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None, help="resume from a v2 checkpoint, including optimizer state when available")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    training_config = config["training"]
    data_path = resolve_path(args.data or data_config.get("output", "processed_v2/operator_dataset_v2.h5"))
    run_name = args.run_name or str(training_config.get("run_name", config.get("model", {}).get("name", "v2_run")))
    seed = int(config.get("seed", 20260827))
    set_seed(seed)
    torch.set_num_threads(int(training_config.get("num_threads", 1)))
    device, device_audit = select_device(str(training_config.get("device", "auto")))

    train_dataset = H5OperatorDataset(data_path, "train")
    val_dataset = H5OperatorDataset(data_path, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
    )

    model_config = dict(config["model"])
    model_config.update(
        {
            "branch_dim": train_dataset.branch_dim,
            "trunk_dim": train_dataset.trunk_dim,
            "output_channels": train_dataset.output_channels,
        }
    )
    model = build_model(model_config).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-6)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, int(training_config.get("lr_patience", 8))),
        min_lr=float(training_config.get("min_learning_rate", 1e-6)),
    )
    epochs = int(args.epochs if args.epochs is not None else training_config.get("epochs", 140))
    patience = int(training_config.get("patience", 25))
    grad_clip = float(training_config.get("grad_clip", 1.0))
    checkpoint_dir = PROJECT_ROOT / "checkpoints_v2" / run_name
    log_dir = PROJECT_ROOT / "logs_v2" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as handle:
        normalization = read_json_attr(handle.attrs["normalization"])
        split_manifest = read_json_attr(handle.attrs["split_manifest"])
        schema = str(handle.attrs["schema"])

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    resumed_from = None
    if args.resume:
        resume_path = resolve_path(args.resume)
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state"])
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_val = float(resume_checkpoint.get("best_val_loss", float("inf")))
        stale_epochs = int(resume_checkpoint.get("stale_epochs", 0))
        resumed_from = str(resume_path)
        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_epoch = int(best_checkpoint.get("best_epoch", best_checkpoint.get("epoch", max(start_epoch - 1, 0))))
        history_path = log_dir / "training_history.csv"
        if history_path.exists():
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                history = [{key: float(value) if key != "epoch" else int(value) for key, value in row.items()} for row in csv.DictReader(handle)]
        elif resume_checkpoint.get("history"):
            history = list(resume_checkpoint["history"])
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            branch = batch["branch"].to(device)
            trunk = batch["trunk"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            loss = masked_mse(model(branch, trunk), target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            valid_count = float(mask.sum().item() * target.shape[-1])
            train_total += float(loss.item()) * valid_count
            train_count += int(valid_count)
        train_loss = train_total / max(train_count, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        scheduler.step(val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        improved = val_loss < best_val - 1e-8
        if improved:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "config": config,
            "normalization": normalization,
            "split_manifest": split_manifest,
            "schema": schema,
            "device_audit": device_audit,
            "epoch": epoch,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "parameter_count": parameter_count,
            "seed": seed,
            "run_name": run_name,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "resumed_from": resumed_from,
            "history": history,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if improved:
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or stale_epochs == 0:
            print(json.dumps({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": learning_rate}, ensure_ascii=False))
        if stale_epochs >= patience:
            break

    runtime_seconds = time.perf_counter() - started
    write_history(log_dir / "training_history.csv", history)
    write_history(PROJECT_ROOT / "training_history_v2.csv", history)
    summary = {
        "experiment_id": run_name,
        "dataset": str(data_path),
        "schema": schema,
        "case_splits": split_manifest.get("case_splits"),
        "grid": split_manifest.get("grid"),
        "model": model.config(),
        "parameter_count": parameter_count,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": max((int(row["epoch"]) for row in history), default=0),
        "history_rows_written": len(history),
        "best_val_loss_model_space": best_val,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "runtime_seconds": runtime_seconds,
        "device": device_audit,
        "checkpoint_best": str(checkpoint_dir / "best.pt"),
        "checkpoint_last": str(checkpoint_dir / "last.pt"),
        "early_stopping_patience": patience,
        "resumed_from": resumed_from,
    }
    write_json(log_dir / "summary.json", summary)
    write_json(PROJECT_ROOT / "reports_v2" / f"{run_name}_training_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
