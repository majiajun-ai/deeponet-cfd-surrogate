"""Build the leakage-safe CFDBench cylinder DeepONet-v2 HDF5 dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for OPTIONAL_DEPS in (PROJECT_ROOT / "work" / "pydeps", PROJECT_ROOT.parent.parent / "work" / "pydeps"):
    if OPTIONAL_DEPS.exists():
        sys.path.append(str(OPTIONAL_DEPS))

import h5py
import numpy as np

from cfd_pretrain.common import resolve_path, write_json, write_json_attr
from cfd_pretrain.config import load_config
from prepare_dataset import bilinear_resample, load_cases, make_query_grid, robust_stats


BRANCH_FEATURES = ["Re", "U_inlet", "time_norm"]
TARGET_VARIABLES = ["u", "v"]


def _case_split(case_id: int, split_manifest: dict[str, Any]) -> str:
    for split in ("train", "val", "test"):
        if case_id in [int(value) for value in split_manifest["case_splits"][split]]:
            return split
    raise ValueError(f"Case {case_id} is not assigned to train/val/test")


def _check_splits(case_ids: list[int], split_manifest: dict[str, Any]) -> None:
    expected = set(case_ids)
    sets = {split: set(int(value) for value in split_manifest["case_splits"][split]) for split in ("train", "val", "test")}
    if any(not values for values in sets.values()):
        raise ValueError(f"Every split must contain at least one case: {sets}")
    if any(sets[left] & sets[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError(f"Case leakage in split manifest: {sets}")
    if set.union(*sets.values()) != expected:
        raise ValueError(f"Split union does not match selected cases: expected={expected}, actual={set.union(*sets.values())}")


def _sample_frame_indices(t_count: int, max_frames: int) -> np.ndarray:
    count = min(int(max_frames), int(t_count))
    if count < 1:
        raise ValueError("A case must contain at least one time frame")
    # Cover the complete available time interval instead of favoring only early frames.
    return np.unique(np.linspace(0, t_count - 1, count, dtype=np.int64))


def build_case_samples(case: dict[str, Any], query_xy: np.ndarray, dt: float, max_frames: int) -> dict[str, Any]:
    u = np.asarray(case["u"], dtype=np.float32)
    v = np.asarray(case["v"], dtype=np.float32)
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError(f"Expected equal [T,H,W] u/v arrays, got {u.shape} and {v.shape}")
    t_count = u.shape[0]
    params = case["params"]
    x_frac = (query_xy[:, 0] - float(params["x_min"])) / (float(params["x_max"]) - float(params["x_min"]))
    y_frac = (query_xy[:, 1] - float(params["y_min"])) / (float(params["y_max"]) - float(params["y_min"]))
    u_query = bilinear_resample(u, x_frac, y_frac)
    v_query = bilinear_resample(v, x_frac, y_frac)
    u_ref = max(float(case["vel_in"]), 1e-12)
    fields = np.stack([u_query / u_ref, v_query / u_ref], axis=-1).astype(np.float32)

    frame_indices = _sample_frame_indices(t_count, max_frames)
    time_norm = frame_indices.astype(np.float32) / max(float(t_count - 1), 1.0)
    time_star = frame_indices.astype(np.float32) * float(dt) * u_ref / float(case["D"])
    branch_raw = np.stack(
        [
            np.full(len(frame_indices), float(case["Re"]), dtype=np.float32),
            np.full(len(frame_indices), u_ref, dtype=np.float32),
            time_norm,
        ],
        axis=-1,
    )
    center_x = float(params.get("center_x", 0.0))
    center_y = float(params.get("center_y", 0.0))
    trunk = np.stack(
        [
            (query_xy[:, 0] - center_x) / float(case["D"]),
            (query_xy[:, 1] - center_y) / float(case["D"]),
        ],
        axis=-1,
    ).astype(np.float32)
    radius = float(params["radius"])
    mask = (((query_xy[:, 0] - center_x) ** 2 + (query_xy[:, 1] - center_y) ** 2) > radius**2).astype(np.float32)
    return {
        "branch_raw": branch_raw,
        "trunk": np.repeat(trunk[None, :, :], len(frame_indices), axis=0),
        "target_raw": fields[frame_indices],
        "mask": np.repeat(mask[None, :], len(frame_indices), axis=0),
        "case_id": np.full(len(frame_indices), int(case["case_id"]), dtype=np.int64),
        "frame_index": frame_indices.astype(np.int64),
        "time_norm": time_norm,
        "time_star": time_star,
        "query_xy": query_xy,
        "raw_time_count": t_count,
    }


def _concat(built: dict[int, dict[str, Any]], case_ids: list[int], name: str) -> np.ndarray:
    return np.concatenate([built[case_id][name] for case_id in case_ids], axis=0)


def _write_case_manifest(
    output_path: Path,
    cases: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    source_manifest_path: Path,
    frame_count: int,
    dt: float,
) -> None:
    source = json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.exists() else {}
    source_members: dict[tuple[int, str], dict[str, Any]] = {
        (int(member["case_id"]), str(member["name"])): member for member in source.get("members", [])
    }
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = int(case["case_id"])
        split = _case_split(case_id, split_manifest)
        if case_id in [int(value) for value in split_manifest.get("interpolation_test_case_ids", [])]:
            test_type = "interpolation"
        elif case_id in [int(value) for value in split_manifest.get("extrapolation_test_case_ids", [])]:
            test_type = "extrapolation"
        else:
            test_type = "not_test"
        params = case["params"]
        members = {name: source_members.get((case_id, name), {}) for name in ("case.json", "u.npy", "v.npy")}
        rows.append(
            {
                "case_id": case_id,
                "split": split,
                "test_type": test_type,
                "Re": float(case["Re"]),
                "U_inlet": float(case["vel_in"]),
                "density": float(case["density"]),
                "viscosity": float(case["viscosity"]),
                "D": float(case["D"]),
                "time_frames": int(case["u"].shape[0]),
                "selected_frames": int(min(frame_count, case["u"].shape[0])),
                "dt": float(dt),
                "time_end": float(max(case["u"].shape[0] - 1, 0) * dt),
                "time_star_end": float(max(case["u"].shape[0] - 1, 0) * dt * case["vel_in"] / case["D"]),
                "grid": f"{case['u'].shape[2]}x{case['u'].shape[1]}",
                "raw_case_dir": str(Path("raw/CFDBench/cylinder_bc_v2") / f"case{case_id:04d}"),
                "source_file_case_json": members["case.json"].get("archive_member", ""),
                "source_file_u": members["u.npy"].get("archive_member", ""),
                "source_file_v": members["v.npy"].get("archive_member", ""),
                "case_json_sha256": members["case.json"].get("sha256", ""),
                "u_sha256": members["u.npy"].get("sha256", ""),
                "v_sha256": members["v.npy"].get("sha256", ""),
                "x_min": float(params["x_min"]),
                "x_max": float(params["x_max"]),
                "y_min": float(params["y_min"]),
                "y_max": float(params["y_max"]),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["case_id"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v2_e1.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--split-output", default=None)
    parser.add_argument("--case-manifest-output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    raw_root = resolve_path(data_config.get("raw_root", "raw/CFDBench/cylinder_bc_v2"))
    output = resolve_path(args.output or data_config.get("output", "processed_v2/operator_dataset_v2.h5"))
    case_ids = [int(value) for value in data_config["case_ids"]]
    split_manifest = {
        "case_splits": {
            "train": [int(value) for value in data_config["train_case_ids"]],
            "val": [int(value) for value in data_config["val_case_ids"]],
            "test": [int(value) for value in data_config["test_case_ids"]],
        },
        "interpolation_test_case_ids": [int(value) for value in data_config.get("interpolation_test_case_ids", [])],
        "extrapolation_test_case_ids": [int(value) for value in data_config.get("extrapolation_test_case_ids", [])],
        "case_count": len(case_ids),
        "split_policy": "whole-case; approximately 70/15/15; test includes explicit interpolation and extrapolation groups",
    }
    _check_splits(case_ids, split_manifest)
    if set(split_manifest["interpolation_test_case_ids"]) | set(split_manifest["extrapolation_test_case_ids"]) != set(split_manifest["case_splits"]["test"]):
        raise ValueError("Interpolation/extrapolation test groups must partition the test cases")

    cases = load_cases(raw_root, case_ids, float(data_config.get("dt", 0.1)))
    case_by_id = {int(case["case_id"]): case for case in cases}
    nx = int(data_config["query_nx"])
    ny = int(data_config["query_ny"])
    query_xy, _, _ = make_query_grid(cases, nx, ny)
    built = {
        int(case["case_id"]): build_case_samples(
            case,
            query_xy,
            dt=float(data_config.get("dt", 0.1)),
            max_frames=int(data_config.get("max_frames", 40)),
        )
        for case in cases
    }

    train_ids = split_manifest["case_splits"]["train"]
    branch_train = _concat(built, train_ids, "branch_raw")
    target_train = _concat(built, train_ids, "target_raw")
    mask_train = _concat(built, train_ids, "mask")
    branch_mean, branch_std = robust_stats(branch_train, axis=0)
    field_values = target_train[mask_train.astype(bool)]
    field_mean, field_std = robust_stats(field_values, axis=0)
    normalization = {
        "coordinate": {
            "x_star": "(x - cylinder_center_x) / D",
            "y_star": "(y - cylinder_center_y) / D",
            "time_norm": "frame_index / (T_case - 1), in [0,1], passed through branch",
            "time_star": "frame_index * dt * Uref / D, retained as an audit field",
            "query_grid_source": "regular grid spanning the union of case.json domains; bilinear from 64x64 release",
        },
        "branch_features": BRANCH_FEATURES,
        "trunk_features": ["x_star", "y_star"],
        "target_variables": TARGET_VARIABLES,
        "field_physical": {"u_star": "u / Uref", "v_star": "v / Uref", "pressure": "not present"},
        "branch_mean": branch_mean.tolist(),
        "branch_std": branch_std.tolist(),
        "field_mean": field_mean.tolist(),
        "field_std": field_std.tolist(),
        "normalization_fit_scope": "training cases only",
    }
    split_manifest["case_parameters"] = {
        str(case_id): {"Re": float(case_by_id[case_id]["Re"]), "U_inlet": float(case_by_id[case_id]["vel_in"])} for case_id in case_ids
    }
    split_manifest["sample_count"] = {split: int(sum(len(built[case_id]["frame_index"]) for case_id in ids)) for split, ids in split_manifest["case_splits"].items()}
    split_manifest["grid"] = {"nx": nx, "ny": ny, "query_points": nx * ny}
    split_manifest["time_sampling"] = {"method": "linspace over full case interval", "max_frames": int(data_config.get("max_frames", 40))}

    split_output = resolve_path(args.split_output or data_config.get("split_output", "splits_v2.json"))
    write_json(split_output, split_manifest)
    normalization_output = PROJECT_ROOT / "metadata" / f"normalization_v2_{nx}x{ny}.json"
    write_json(normalization_output, normalization)
    source_manifest = resolve_path(data_config.get("source_manifest", "raw/CFDBench/source_manifest_v2.json"))
    case_manifest_output = resolve_path(args.case_manifest_output or data_config.get("case_manifest_output", "processed_v2/case_manifest.csv"))
    _write_case_manifest(case_manifest_output, cases, split_manifest, source_manifest, int(data_config.get("max_frames", 40)), float(data_config.get("dt", 0.1)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["schema"] = "OperatorDataset-v2"
        handle.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["source_dataset"] = "CFDBench interpolated cylinder/bc subset"
        handle.attrs["dimension"] = 2
        handle.attrs["target_variables"] = write_json_attr(TARGET_VARIABLES)
        handle.attrs["branch_features"] = write_json_attr(BRANCH_FEATURES)
        handle.attrs["trunk_features"] = write_json_attr(["x_star", "y_star"])
        handle.attrs["normalization"] = write_json_attr(normalization)
        handle.attrs["split_manifest"] = write_json_attr(split_manifest)
        handle.attrs["query_shape"] = write_json_attr([ny, nx])
        handle.attrs["time_condition"] = "time_norm is a branch condition; trunk contains x_star,y_star"
        handle.create_dataset("query_xy", data=query_xy.astype(np.float32), compression="gzip")
        for split in ("train", "val", "test"):
            ids = split_manifest["case_splits"][split]
            group = handle.create_group(split)
            branch_raw = _concat(built, ids, "branch_raw")
            target_raw = _concat(built, ids, "target_raw")
            group.create_dataset("branch", data=((branch_raw - branch_mean) / branch_std).astype(np.float32), compression="gzip")
            group.create_dataset("trunk", data=_concat(built, ids, "trunk").astype(np.float32), compression="gzip")
            group.create_dataset("target", data=((target_raw - field_mean) / field_std).astype(np.float32), compression="gzip")
            group.create_dataset("mask", data=_concat(built, ids, "mask").astype(np.float32), compression="gzip")
            group.create_dataset("case_id", data=_concat(built, ids, "case_id"), compression="gzip")
            group.create_dataset("frame_index", data=_concat(built, ids, "frame_index"), compression="gzip")
            group.create_dataset("time_norm", data=_concat(built, ids, "time_norm").astype(np.float32), compression="gzip")
            group.create_dataset("time_star", data=_concat(built, ids, "time_star").astype(np.float32), compression="gzip")

    case_audit = []
    for case in cases:
        case_audit.append(
            {
                "case_id": int(case["case_id"]),
                "split": _case_split(int(case["case_id"]), split_manifest),
                "parameters": {key: float(case[key]) for key in ("vel_in", "density", "viscosity", "D", "Re", "domain_width", "domain_height")},
                "raw_params": case["params"],
                "raw_shapes": {"u": list(case["u"].shape), "v": list(case["v"].shape)},
                "selected_frame_indices": built[int(case["case_id"])]["frame_index"].tolist(),
            }
        )
    write_json(PROJECT_ROOT / "metadata" / "case_audit_v2.json", {"cases": case_audit, "normalization": normalization})
    print(json.dumps({"output": str(output), "split_manifest": split_manifest, "normalization": normalization}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
