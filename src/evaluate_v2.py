"""Evaluate a v2 checkpoint against leakage-safe baselines and wake metrics."""

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
import numpy as np
import torch

from cfd_pretrain.common import read_json_attr, resolve_path, select_device, write_json
from cfd_pretrain.metrics_v2 import by_case, by_time, full_and_regions, probe_diagnostics
from cfd_pretrain.model_v2 import build_model


def _read_split(handle: h5py.File, split: str) -> dict[str, np.ndarray]:
    group = handle[split]
    return {
        "branch": group["branch"][:],
        "trunk": group["trunk"][:],
        "target": group["target"][:],
        "mask": group["mask"][:],
        "case_id": group["case_id"][:].astype(np.int64),
        "frame_index": group["frame_index"][:].astype(np.int64),
        "time_norm": group["time_norm"][:].astype(np.float32),
        "time_star": group["time_star"][:].astype(np.float32),
    }


def _predict(checkpoint: Path, data_path: Path, split: str, batch_size: int = 32) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    with h5py.File(data_path, "r") as handle:
        data = _read_split(handle, split)
        normalization = read_json_attr(handle.attrs["normalization"])
        split_manifest = read_json_attr(handle.attrs["split_manifest"])
    device, device_audit = select_device("cpu")
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint_data["model_config"]).to(device)
    model.load_state_dict(checkpoint_data["model_state"])
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data["branch"]), batch_size):
            stop = min(len(data["branch"]), start + batch_size)
            branch = torch.from_numpy(data["branch"][start:stop]).float().to(device)
            trunk = torch.from_numpy(data["trunk"][start:stop]).float().to(device)
            predictions.append(model(branch, trunk).cpu().numpy().astype(np.float32))
    prediction_norm = np.concatenate(predictions, axis=0)
    field_mean = np.asarray(normalization["field_mean"], dtype=np.float32)
    field_std = np.asarray(normalization["field_std"], dtype=np.float32)
    target = data["target"] * field_std + field_mean
    prediction = prediction_norm * field_std + field_mean
    data["target_physical"] = target
    data["prediction_physical"] = prediction
    data["device_audit"] = device_audit
    return data, normalization, split_manifest, checkpoint_data


def _mean_field(train: dict[str, np.ndarray], split_manifest: dict[str, Any]) -> np.ndarray:
    # Equal weight per training case; no validation/test values enter this statistic.
    case_means: list[np.ndarray] = []
    for case_id in split_manifest["case_splits"]["train"]:
        selected = train["case_id"] == int(case_id)
        target = train["target_physical"][selected]
        mask = train["mask"][selected]
        numerator = np.sum(target * mask[..., None], axis=0)
        denominator = np.sum(mask, axis=0)
        case_means.append(numerator / np.maximum(denominator[..., None], 1e-12))
    return np.mean(np.stack(case_means, axis=0), axis=0).astype(np.float32)


def _case_time_field(train: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for case_id in sorted(set(int(value) for value in train["case_id"].tolist())):
        selected = train["case_id"] == case_id
        order = np.argsort(train["time_norm"][selected])
        result[case_id] = {
            "time": train["time_norm"][selected][order],
            "field": train["target_physical"][selected][order],
            "Re": np.asarray([float(train["re_by_case"][case_id])], dtype=np.float32),
        }
    return result


def _sample_time(series: dict[str, np.ndarray], time_value: float) -> np.ndarray:
    times = series["time"]
    fields = series["field"]
    if time_value <= float(times[0]):
        return fields[0]
    if time_value >= float(times[-1]):
        return fields[-1]
    right = int(np.searchsorted(times, time_value, side="right"))
    left = right - 1
    weight = float((time_value - times[left]) / max(times[right] - times[left], 1e-12))
    return ((1.0 - weight) * fields[left] + weight * fields[right]).astype(np.float32)


def _linear_re_baseline(train: dict[str, np.ndarray], test: dict[str, np.ndarray], split_manifest: dict[str, Any]) -> np.ndarray:
    series = _case_time_field(train)
    train_cases = sorted(series, key=lambda case_id: float(train["re_by_case"][case_id]))
    train_re = np.asarray([float(train["re_by_case"][case_id]) for case_id in train_cases], dtype=np.float32)
    prediction = np.empty_like(test["target_physical"], dtype=np.float32)
    for row, case_id in enumerate(test["case_id"].tolist()):
        re_value = float(test["re_by_case"][int(case_id)])
        if re_value <= train_re[0]:
            lower_index, upper_index = 0, 1
        elif re_value >= train_re[-1]:
            lower_index, upper_index = len(train_re) - 2, len(train_re) - 1
        else:
            upper_index = int(np.searchsorted(train_re, re_value, side="right"))
            lower_index = upper_index - 1
        lower_case = train_cases[lower_index]
        upper_case = train_cases[upper_index]
        lower_field = _sample_time(series[lower_case], float(test["time_norm"][row]))
        upper_field = _sample_time(series[upper_case], float(test["time_norm"][row]))
        alpha = (re_value - train_re[lower_index]) / max(float(train_re[upper_index] - train_re[lower_index]), 1e-12)
        prediction[row] = ((1.0 - alpha) * lower_field + alpha * upper_field).astype(np.float32)
    return prediction


def _prepare_data_arrays(data: dict[str, np.ndarray], split_manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    data = dict(data)
    re_by_case = {int(case_id): float(values["Re"]) for case_id, values in split_manifest["case_parameters"].items()}
    data["re_by_case"] = re_by_case
    return data


def _flatten_rows(experiment_id: str, method: str, subset: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region in ("full", "wake", "near_cylinder"):
        for metric_name, value in result["metrics"][region].items():
            rows.append({"experiment_id": experiment_id, "method": method, "subset": subset, "scope": region, "metric": metric_name, "value": value})
    for time_name, metrics in result.get("by_time", {}).items():
        for metric_name, value in metrics.items():
            rows.append({"experiment_id": experiment_id, "method": method, "subset": subset, "scope": f"time_{time_name}", "metric": metric_name, "value": value})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()

    data_path = resolve_path(args.data)
    checkpoint_path = resolve_path(args.checkpoint)
    experiment_id = args.experiment_id or checkpoint_path.parent.name
    test, normalization, split_manifest, checkpoint_data = _predict(checkpoint_path, data_path, "test")
    with h5py.File(data_path, "r") as handle:
        train = _read_split(handle, "train")
    train["target_physical"] = train["target"] * np.asarray(normalization["field_std"], dtype=np.float32) + np.asarray(normalization["field_mean"], dtype=np.float32)
    train = _prepare_data_arrays(train, split_manifest)
    test = _prepare_data_arrays(test, split_manifest)

    mean_field = _mean_field(train, split_manifest)
    field_mean = np.asarray(normalization["field_mean"], dtype=np.float32)
    field_std = np.asarray(normalization["field_std"], dtype=np.float32)
    uniform = np.asarray([1.0, 0.0], dtype=np.float32)[None, None, :]
    linear_re = _linear_re_baseline(train, test, split_manifest)
    predictions = {
        "deeponet": test["prediction_physical"],
        "uniform_inlet": np.broadcast_to(uniform, test["target_physical"].shape).copy(),
        "mean_field": np.broadcast_to(mean_field[None, :, :], test["target_physical"].shape).copy(),
        "linear_re": linear_re,
    }
    test_types = {int(value): "interpolation" for value in split_manifest["interpolation_test_case_ids"]}
    test_types.update({int(value): "extrapolation" for value in split_manifest["extrapolation_test_case_ids"]})
    methods: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for method, prediction in predictions.items():
        method_result: dict[str, Any] = {}
        for subset_name, selected in (
            ("all", np.ones(len(test["case_id"]), dtype=bool)),
            ("interpolation", np.asarray([test_types[int(value)] == "interpolation" for value in test["case_id"]], dtype=bool)),
            ("extrapolation", np.asarray([test_types[int(value)] == "extrapolation" for value in test["case_id"]], dtype=bool)),
        ):
            if not selected.any():
                continue
            subset_result = {
                "metrics": full_and_regions(prediction[selected], test["target_physical"][selected], test["mask"][selected], test["trunk"][selected]),
                "by_time": by_time(prediction[selected], test["target_physical"][selected], test["mask"][selected], test["time_norm"][selected]),
            }
            method_result[subset_name] = subset_result
            csv_rows.extend(_flatten_rows(experiment_id, method, subset_name, subset_result))
        method_result["by_case"] = by_case(
            prediction,
            test["target_physical"],
            test["mask"],
            test["trunk"],
            test["time_norm"],
            test["case_id"],
            test["re_by_case"],
        )
        method_result["probe"] = probe_diagnostics(prediction, test["target_physical"], test["trunk"], test["time_norm"], test["case_id"])
        methods[method] = method_result

    evaluation = {
        "experiment_id": experiment_id,
        "data": str(data_path),
        "checkpoint": str(checkpoint_path),
        "model_config": checkpoint_data["model_config"],
        "parameter_count": checkpoint_data.get("parameter_count"),
        "split_manifest": split_manifest,
        "normalization_fit_scope": "training cases only",
        "definitions": {
            "wake": "0 < x_star <= 8 and |y_star| <= 3, obstacle excluded; x_star max is 8 for this release",
            "near_cylinder": "-1 <= x_star <= 3 and |y_star| <= 2, obstacle excluded",
            "linear_re": "two nearest training-Re case trajectories, linear interpolation in time_norm and Re; linear extrapolation only for end-point test cases",
            "mean_field": "equal-weight average of time-averaged training-case fields at each query point",
        },
        "test_case_ids": [int(value) for value in sorted(set(test["case_id"].tolist()))],
        "test_case_types": test_types,
        "methods": methods,
        "device": test["device_audit"],
    }
    output_dir = PROJECT_ROOT / "reports_v2" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation.json", evaluation)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "method", "subset", "scope", "metric", "value"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
