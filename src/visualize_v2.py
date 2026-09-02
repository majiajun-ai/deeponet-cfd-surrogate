"""Create the required v2 field, wake and baseline figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for OPTIONAL_DEPS in (PROJECT_ROOT / "work" / "pydeps", PROJECT_ROOT.parent.parent / "work" / "pydeps"):
    if OPTIONAL_DEPS.exists():
        sys.path.append(str(OPTIONAL_DEPS))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cfd_pretrain.common import read_json, read_json_attr, resolve_path, write_json
from evaluate_v2 import _predict, _prepare_data_arrays


def _masked_image(values: np.ndarray, mask: np.ndarray, ny: int, nx: int) -> np.ndarray:
    return np.where(mask.reshape(ny, nx).astype(bool), values.reshape(ny, nx), np.nan)


def _save_field_figure(path: Path, case_id: int, time_value: float, truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray, trunk: np.ndarray, ny: int, nx: int) -> None:
    truth_u = _masked_image(truth[:, 0], mask, ny, nx)
    truth_v = _masked_image(truth[:, 1], mask, ny, nx)
    pred_u = _masked_image(prediction[:, 0], mask, ny, nx)
    pred_v = _masked_image(prediction[:, 1], mask, ny, nx)
    truth_speed = np.sqrt(truth_u**2 + truth_v**2)
    pred_speed = np.sqrt(pred_u**2 + pred_v**2)
    error_speed = np.abs(pred_speed - truth_speed)
    error_u = np.abs(pred_u - truth_u)
    error_v = np.abs(pred_v - truth_v)
    x = trunk[:, 0].reshape(ny, nx)
    y = trunk[:, 1].reshape(ny, nx)
    extent = [float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))]
    panels = [
        (truth_speed, "ground truth |V|", "viridis"),
        (pred_speed, "DeepONet |V|", "viridis"),
        (error_speed, "absolute |V| error", "magma"),
        (truth_u, "ground truth u", "coolwarm"),
        (pred_u, "prediction u", "coolwarm"),
        (error_u, "absolute u error", "magma"),
        (truth_v, "ground truth v", "coolwarm"),
        (pred_v, "prediction v", "coolwarm"),
        (error_v, "absolute v error", "magma"),
    ]
    figure, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for axis, (values, title, cmap) in zip(axes.ravel(), panels):
        finite = values[np.isfinite(values)]
        if "error" in title:
            vmin, vmax = 0.0, float(np.percentile(finite, 98)) if len(finite) else 1.0
        else:
            vmin, vmax = float(np.percentile(finite, 2)) if len(finite) else -1.0, float(np.percentile(finite, 98)) if len(finite) else 1.0
            if "u" in title or "v" in title:
                bound = max(abs(vmin), abs(vmax))
                vmin, vmax = -bound, bound
        image = axis.imshow(values, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-6), aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("x/D")
        axis.set_ylabel("y/D")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(f"CFDBench v2 case {case_id}, time_norm={time_value:.3f}")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _load_history(experiment_id: str) -> tuple[list[float], list[float], list[float]]:
    path = PROJECT_ROOT / "logs_v2" / experiment_id / "training_history.csv"
    if not path.exists():
        return [], [], []
    epochs: list[float] = []
    train: list[float] = []
    val: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epochs.append(float(row["epoch"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return epochs, train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--case-id", type=int, default=None)
    args = parser.parse_args()

    data_path = resolve_path(args.data)
    checkpoint_path = resolve_path(args.checkpoint)
    test, _, split_manifest, _ = _predict(checkpoint_path, data_path, "test")
    test = _prepare_data_arrays(test, split_manifest)
    ny, nx = int(split_manifest["grid"]["ny"]), int(split_manifest["grid"]["nx"])
    figure_dir = PROJECT_ROOT / "figures_v2"
    figure_dir.mkdir(parents=True, exist_ok=True)
    candidate_cases = sorted(set(int(value) for value in test["case_id"].tolist()))
    case_id = args.case_id if args.case_id is not None else int(split_manifest["interpolation_test_case_ids"][0])
    if case_id not in candidate_cases:
        raise ValueError(f"case {case_id} not in test cases {candidate_cases}")
    selected_case = test["case_id"] == case_id
    case_indices = np.where(selected_case)[0]
    figure_paths: dict[str, str] = {}
    for label, target_time in (("early", 0.0), ("mid", 0.5), ("late", 1.0)):
        index = int(case_indices[np.argmin(np.abs(test["time_norm"][case_indices] - target_time))])
        path = figure_dir / f"test_case_prediction_{label}.png"
        _save_field_figure(path, case_id, float(test["time_norm"][index]), test["target_physical"][index], test["prediction_physical"][index], test["mask"][index], test["trunk"][index], ny, nx)
        figure_paths[f"test_case_prediction_{label}"] = str(path)

    wake = test["mask"].astype(bool) & (test["trunk"][..., 0] > 0.0) & (test["trunk"][..., 0] <= 8.0) & (np.abs(test["trunk"][..., 1]) <= 3.0)
    absolute_error = np.linalg.norm(test["prediction_physical"] - test["target_physical"], axis=-1)
    wake_error = np.where(wake, absolute_error, np.nan).mean(axis=0)
    x = test["trunk"][0, :, 0].reshape(ny, nx)
    y = test["trunk"][0, :, 1].reshape(ny, nx)
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    image = axis.pcolormesh(x, y, wake_error.reshape(ny, nx), shading="auto", cmap="magma")
    axis.set_title("v2 mean wake velocity error over test cases/time")
    axis.set_xlabel("x/D")
    axis.set_ylabel("y/D")
    figure.colorbar(image, ax=axis, label="|error|")
    figure.savefig(figure_dir / "wake_error_map.png", dpi=160)
    plt.close(figure)
    figure_paths["wake_error_map"] = str(figure_dir / "wake_error_map.png")

    evaluation = read_json(PROJECT_ROOT / "reports_v2" / args.experiment_id / "evaluation.json")
    methods = ["deeponet", "uniform_inlet", "mean_field", "linear_re"]
    labels = {"deeponet": "DeepONet", "uniform_inlet": "uniform", "mean_field": "mean field", "linear_re": "linear Re"}
    colors = {"deeponet": "#1f77b4", "uniform_inlet": "#7f7f7f", "mean_field": "#ff7f0e", "linear_re": "#2ca02c"}
    by_case_rows = {method: evaluation["methods"][method]["by_case"] for method in methods}
    cases = [row["case_id"] for row in by_case_rows["deeponet"]]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x_locations = np.arange(len(cases))
    width = 0.2
    for offset, method in enumerate(methods):
        rows = {row["case_id"]: row for row in by_case_rows[method]}
        axes[0].bar(x_locations + (offset - 1.5) * width, [rows[case]["full"]["field_rmse"] for case in cases], width, label=labels[method], color=colors[method])
        axes[1].bar(x_locations + (offset - 1.5) * width, [rows[case]["wake"]["field_rmse"] for case in cases], width, label=labels[method], color=colors[method])
    for axis, title in zip(axes, ("full-field RMSE by test case", "wake RMSE by test case")):
        axis.set_title(title)
        axis.set_xticks(x_locations, [str(case) for case in cases])
        axis.set_xlabel("case id")
        axis.set_ylabel("RMSE")
    axes[0].legend(fontsize=8)
    figure.savefig(figure_dir / "metric_by_case.png", dpi=160)
    plt.close(figure)
    figure_paths["metric_by_case"] = str(figure_dir / "metric_by_case.png")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method in methods:
        rows = sorted(by_case_rows[method], key=lambda row: row["Re"])
        axes[0].plot([row["Re"] for row in rows], [row["full"]["field_rmse"] for row in rows], "o-", label=labels[method], color=colors[method])
        axes[1].plot([row["Re"] for row in rows], [row["wake"]["field_rmse"] for row in rows], "o-", label=labels[method], color=colors[method])
    for axis, title in zip(axes, ("full-field RMSE vs Re", "wake RMSE vs Re")):
        axis.set_title(title)
        axis.set_xlabel("Re")
        axis.set_ylabel("RMSE")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.savefig(figure_dir / "metric_vs_Re.png", dpi=160)
    plt.close(figure)
    figure_paths["metric_vs_Re"] = str(figure_dir / "metric_vs_Re.png")

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x_locations = np.arange(2)
    for method in methods:
        all_metrics = evaluation["methods"][method]["all"]["metrics"]
        axis.bar(x_locations + (methods.index(method) - 1.5) * width, [all_metrics["full"]["field_rmse"], all_metrics["wake"]["field_rmse"]], width, label=labels[method], color=colors[method])
    axis.set_xticks(x_locations, ["full field", "wake"])
    axis.set_ylabel("RMSE")
    axis.set_title("v2 baseline comparison on the same test cases")
    axis.legend()
    figure.savefig(figure_dir / "baseline_comparison.png", dpi=160)
    plt.close(figure)
    figure_paths["baseline_comparison"] = str(figure_dir / "baseline_comparison.png")

    epochs, train_loss, val_loss = _load_history(args.experiment_id)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if epochs:
        axis.plot(epochs, train_loss, label="train")
        axis.plot(epochs, val_loss, label="validation")
        axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("masked MSE (standardized space)")
    axis.set_title(f"{args.experiment_id} learning curve")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.savefig(figure_dir / "loss_curve.png", dpi=160)
    plt.close(figure)
    figure_paths["loss_curve"] = str(figure_dir / "loss_curve.png")

    write_json(figure_dir / "visualization_manifest.json", {"experiment_id": args.experiment_id, "case_id": case_id, "files": figure_paths})
    print(json.dumps({"experiment_id": args.experiment_id, "case_id": case_id, "files": figure_paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
