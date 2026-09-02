"""Generate ground-truth/prediction/error and learning-curve figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for OPTIONAL_DEPS in (PROJECT_ROOT / "work" / "pydeps", PROJECT_ROOT.parent.parent / "work" / "pydeps"):
    if OPTIONAL_DEPS.exists():
        sys.path.append(str(OPTIONAL_DEPS))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cfd_pretrain.common import read_json_attr, resolve_path
from cfd_pretrain.dataset import H5OperatorDataset
from cfd_pretrain.model import DeepONet


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="processed/operator_dataset.h5")
    parser.add_argument("--checkpoint", default="checkpoints/public_pretrain/best.pt")
    parser.add_argument("--history", default="training_history.csv")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    data_path = resolve_path(args.data)
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location="cpu", weights_only=False)
    model = DeepONet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = H5OperatorDataset(data_path, "test")
    with h5py.File(data_path, "r") as handle:
        normalization = read_json_attr(handle.attrs["normalization"])
        query_shape = tuple(read_json_attr(handle.attrs["query_shape"]))
        query_xy = handle["query_xy"][:]
    field_mean = np.asarray(normalization["field_mean"], dtype=np.float32)
    field_std = np.asarray(normalization["field_std"], dtype=np.float32)
    x = query_xy[:, 0].reshape(query_shape)
    y = query_xy[:, 1].reshape(query_shape)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def render(index: int, label: str) -> Path:
        sample = dataset[index]
        with torch.no_grad():
            prediction = model(sample["branch"].unsqueeze(0), sample["trunk"].unsqueeze(0))[0].numpy()
        target = sample["target"].numpy()
        mask = sample["mask"].numpy().astype(bool)
        prediction = prediction * field_std + field_mean
        target = target * field_std + field_mean
        error = np.abs(prediction - target)
        pred_mag = np.linalg.norm(prediction[:, :2], axis=-1)
        target_mag = np.linalg.norm(target[:, :2], axis=-1)
        mag_error = np.abs(pred_mag - target_mag)
        panels = [
            ("velocity magnitude — truth", target_mag, "viridis"),
            ("velocity magnitude — prediction", pred_mag, "viridis"),
            ("velocity magnitude — absolute error", mag_error, "magma"),
            ("u — truth", target[:, 0], "coolwarm"),
            ("u — prediction", prediction[:, 0], "coolwarm"),
            ("u — absolute error", error[:, 0], "magma"),
            ("v — truth", target[:, 1], "coolwarm"),
            ("v — prediction", prediction[:, 1], "coolwarm"),
            ("v — absolute error", error[:, 1], "magma"),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
        for axis, (title, values, cmap) in zip(axes.ravel(), panels):
            image = np.ma.array(values.reshape(query_shape), mask=~mask.reshape(query_shape))
            mesh = axis.pcolormesh(x, y, image, shading="auto", cmap=cmap)
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            fig.colorbar(mesh, ax=axis, shrink=0.8)
        fig.suptitle(f"CFDBench cylinder test case {sample['case_id']}, frame {sample['frame_index']} — pressure unavailable")
        target_path = out_dir / f"test_case_prediction_{label}.png"
        fig.savefig(target_path, dpi=180)
        plt.close(fig)
        return target_path

    early_path = render(0, "early")
    late_path = render(len(dataset) - 1, "late")
    # Keep the original stable filename as the late-time diagnostic.
    import shutil

    shutil.copyfile(late_path, out_dir / "test_case_prediction.png")

    history = read_csv(resolve_path(args.history))
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.semilogy([float(row["epoch"]) for row in history], [float(row["train_loss"]) for row in history], label="train")
    axis.semilogy([float(row["epoch"]) for row in history], [float(row["val_loss"]) for row in history], label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("masked model-space MSE")
    axis.set_title("DeepONet training history")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"prediction_figure": str(out_dir / "test_case_prediction.png"), "early_prediction_figure": str(early_path), "late_prediction_figure": str(late_path), "loss_figure": str(out_dir / "loss_curve.png")}, indent=2))


if __name__ == "__main__":
    main()
