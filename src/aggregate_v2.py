"""Aggregate v2 training/evaluation artifacts into an ablation table."""

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

from cfd_pretrain.common import read_json, resolve_path, write_json


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default="reports_v2")
    parser.add_argument("--output", default="experiment_summary.csv")
    args = parser.parse_args()

    reports_root = resolve_path(args.reports_root)
    rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for evaluation_path in sorted(reports_root.glob("*/evaluation.json")):
        evaluation = _read(evaluation_path)
        experiment_id = str(evaluation["experiment_id"])
        if not experiment_id.startswith("v2_"):
            continue
        training_path = reports_root / f"{experiment_id}_training_summary.json"
        training = _read(training_path) if training_path.exists() else {}
        methods = evaluation["methods"]
        deeponet = methods["deeponet"]
        mean_field = methods["mean_field"]
        linear_re = methods["linear_re"]
        model_all = deeponet["all"]["metrics"]
        model_full = model_all["full"]
        row = {
            "experiment_id": experiment_id,
            "dataset_cases": evaluation["split_manifest"]["case_count"],
            "train_cases": len(evaluation["split_manifest"]["case_splits"]["train"]),
            "val_cases": len(evaluation["split_manifest"]["case_splits"]["val"]),
            "test_cases": len(evaluation["split_manifest"]["case_splits"]["test"]),
            "grid": f"{evaluation['split_manifest']['grid']['nx']}x{evaluation['split_manifest']['grid']['ny']}",
            "model_version": "enhanced_fourier" if evaluation["model_config"].get("coordinate_encoding") == "fourier" else ("larger_mlp" if evaluation["model_config"].get("width", 0) >= 160 else "baseline"),
            "parameter_count": evaluation.get("parameter_count"),
            "seed": training.get("seed", 20260827),
            "best_epoch": training.get("best_epoch"),
            "train_loss": training.get("final_train_loss"),
            "val_loss": training.get("best_val_loss_model_space"),
            "test_rmse": model_full["field_rmse"],
            "test_relative_l2": model_full["relative_l2"],
            "wake_rmse": deeponet["all"]["metrics"]["wake"]["field_rmse"],
            "near_cylinder_rmse": deeponet["all"]["metrics"]["near_cylinder"]["field_rmse"],
            "interpolation_test_rmse": deeponet["interpolation"]["metrics"]["full"]["field_rmse"],
            "extrapolation_test_rmse": deeponet["extrapolation"]["metrics"]["full"]["field_rmse"],
            "mean_field_test_rmse": mean_field["all"]["metrics"]["full"]["field_rmse"],
            "mean_field_wake_rmse": mean_field["all"]["metrics"]["wake"]["field_rmse"],
            "linear_re_test_rmse": linear_re["all"]["metrics"]["full"]["field_rmse"],
            "linear_re_interpolation_rmse": linear_re["interpolation"]["metrics"]["full"]["field_rmse"],
            "runtime_seconds": training.get("runtime_seconds"),
            "mean_field_beaten": bool(model_full["field_rmse"] < mean_field["all"]["metrics"]["full"]["field_rmse"]),
            "wake_mean_field_beaten": bool(deeponet["all"]["metrics"]["wake"]["field_rmse"] < mean_field["all"]["metrics"]["wake"]["field_rmse"]),
            "linear_re_interpolation_beaten": bool(deeponet["interpolation"]["metrics"]["full"]["field_rmse"] < linear_re["interpolation"]["metrics"]["full"]["field_rmse"]),
            "checkpoint_best": training.get("checkpoint_best", ""),
        }
        rows.append(row)
        evaluations.append(evaluation)

    rows.sort(key=lambda row: (float("inf") if row["test_rmse"] is None else float(row["test_rmse"])))
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["experiment_id"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if output_path != PROJECT_ROOT / "experiments" / "experiment_summary.csv":
        mirror = PROJECT_ROOT / "experiments" / "experiment_summary.csv"
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")

    best = rows[0] if rows else None
    summary = {
        "experiments_found": len(rows),
        "ranking_metric": "test full-field RMSE across the same case-level test set",
        "best_experiment": best,
        "all_experiments": rows,
        "negative_results_retained": True,
    }
    write_json(PROJECT_ROOT / "reports_v2" / "ablation_summary.json", summary)
    if best:
        write_json(PROJECT_ROOT / "reports_v2" / "best_model.json", best)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
