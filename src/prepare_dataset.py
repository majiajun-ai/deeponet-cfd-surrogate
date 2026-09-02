"""Convert selected CFDBench cylinder cases to a unified HDF5 OperatorDataset."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from cfd_pretrain.common import PROJECT_ROOT, read_json, resolve_path, write_json, write_json_attr


BRANCH_FEATURES = ["vel_in", "density", "viscosity", "D", "Re", "domain_width", "domain_height"]


def bilinear_resample(field: np.ndarray, x_fraction: np.ndarray, y_fraction: np.ndarray) -> np.ndarray:
    """Bilinearly resample [T,H,W] fields at fractional grid locations."""

    t_count, height, width = field.shape
    x = np.clip(x_fraction * (width - 1), 0.0, width - 1.0)
    y = np.clip(y_fraction * (height - 1), 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    a = field[:, y0, x0]
    b = field[:, y0, x1]
    c = field[:, y1, x0]
    d = field[:, y1, x1]
    return (
        (1 - wx)[None, :] * (1 - wy)[None, :] * a
        + wx[None, :] * (1 - wy)[None, :] * b
        + (1 - wx)[None, :] * wy[None, :] * c
        + wx[None, :] * wy[None, :] * d
    ).astype(np.float32)


def load_cases(raw_root: Path, case_ids: list[int], dt: float) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        case_dir = raw_root / f"case{case_id:04d}"
        if not case_dir.exists():
            raise FileNotFoundError(f"Missing raw case directory: {case_dir}")
        params = read_json(case_dir / "case.json")
        u = np.load(case_dir / "u.npy", mmap_mode="r")
        v = np.load(case_dir / "v.npy", mmap_mode="r")
        if u.shape != v.shape or u.ndim != 3:
            raise ValueError(f"Expected u/v [T,H,W] with equal shapes, got {u.shape} / {v.shape}")
        radius = float(params["radius"])
        diameter = 2.0 * radius
        vel_in = float(params["vel_in"])
        density = float(params["density"])
        viscosity = float(params["viscosity"])
        reynolds = density * vel_in * diameter / max(viscosity, 1e-12)
        domain_width = float(params["x_max"]) - float(params["x_min"])
        domain_height = float(params["y_max"]) - float(params["y_min"])
        cases.append(
            {
                "case_id": case_id,
                "case_dir": case_dir,
                "params": params,
                "u": u,
                "v": v,
                "dt": float(dt),
                "dimension": 2,
                "solver_fidelity": "interpolated CFD benchmark; upstream solver generation is ANSYS Fluent",
                "fidelity_class": "RANS/CFD unknown from released interpolated files; do not label as DNS/LES",
                "vel_in": vel_in,
                "density": density,
                "viscosity": viscosity,
                "D": diameter,
                "Re": reynolds,
                "domain_width": domain_width,
                "domain_height": domain_height,
                "x_min": float(params["x_min"]),
                "x_max": float(params["x_max"]),
                "y_min": float(params["y_min"]),
                "y_max": float(params["y_max"]),
            }
        )
    return cases


def make_query_grid(cases: list[dict[str, Any]], nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # CFDBench case domains are comparable but each case.json remains the source of truth.
    x_min = min(case["x_min"] for case in cases)
    x_max = max(case["x_max"] for case in cases)
    y_min = min(case["y_min"] for case in cases)
    y_max = max(case["y_max"] for case in cases)
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    y = np.linspace(y_min, y_max, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    query_xy = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return query_xy, xx, yy


def build_case_samples(
    case: dict[str, Any],
    query_xy: np.ndarray,
    frame_stride: int,
    max_frames: int,
    dt: float,
    time_input_scale: float = 1.0,
) -> dict[str, Any]:
    u = np.asarray(case["u"], dtype=np.float32)
    v = np.asarray(case["v"], dtype=np.float32)
    t_count, height, width = u.shape
    params = case["params"]
    x_frac = (query_xy[:, 0] - float(params["x_min"])) / (float(params["x_max"]) - float(params["x_min"]))
    y_frac = (query_xy[:, 1] - float(params["y_min"])) / (float(params["y_max"]) - float(params["y_min"]))
    u_query = bilinear_resample(u, x_frac, y_frac)
    v_query = bilinear_resample(v, x_frac, y_frac)
    u_ref = max(float(case["vel_in"]), 1e-12)
    fields = np.stack([u_query / u_ref, v_query / u_ref], axis=-1)  # [T,Q,2]
    radius = float(params["radius"])
    center_x = float(params.get("center_x", 0.0))
    center_y = float(params.get("center_y", 0.0))
    valid = ((query_xy[:, 0] - center_x) ** 2 + (query_xy[:, 1] - center_y) ** 2 > radius**2).astype(np.float32)
    frame_indices = list(range(0, t_count, max(1, frame_stride)))[:max_frames]
    branch_raw = np.array(
        [
            case["vel_in"],
            case["density"],
            case["viscosity"],
            case["D"],
            case["Re"],
            case["domain_width"],
            case["domain_height"],
        ],
        dtype=np.float32,
    )
    x_star = (query_xy[:, 0] - center_x) / case["D"]
    y_star = (query_xy[:, 1] - center_y) / case["D"]
    # The released CFDBench files expose no physical time unit in case.json.
    # The upstream loader defines dt=0.1; with D and U this is the consistent t*.
    trunk_base = np.stack([x_star, y_star], axis=-1).astype(np.float32)
    trunk = np.stack(
        [np.concatenate([trunk_base, np.full((len(query_xy), 1), i * dt * u_ref / case["D"] / time_input_scale, dtype=np.float32)], axis=-1) for i in frame_indices],
        axis=0,
    )
    targets = fields[frame_indices]
    return {
        "branch_raw": np.repeat(branch_raw[None, :], len(frame_indices), axis=0),
        "trunk": trunk,
        "target_raw": targets.astype(np.float32),
        "mask": np.repeat(valid[None, :], len(frame_indices), axis=0),
        "case_id": np.full(len(frame_indices), case["case_id"], dtype=np.int64),
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "fields": fields,
        "query_xy": query_xy,
    }


def robust_stats(values: np.ndarray, axis: tuple[int, ...] | int | None = None) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=axis).astype(np.float32)
    std = np.std(values, axis=axis).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini.yaml")
    parser.add_argument("--raw-root", default="raw/CFDBench/cylinder_bc")
    parser.add_argument("--output", default="processed/operator_dataset.h5")
    args = parser.parse_args()
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    raw_root = resolve_path(args.raw_root)
    output = resolve_path(args.output)
    case_ids = [int(x) for x in config["data"]["case_ids"]]
    cases = load_cases(raw_root, case_ids, float(config["data"]["dt"]))
    query_xy, xx, yy = make_query_grid(cases, int(config["data"]["query_nx"]), int(config["data"]["query_ny"]))

    built = [
        build_case_samples(
            case,
            query_xy,
            frame_stride=int(config["data"]["frame_stride"]),
            max_frames=int(config["data"]["max_frames"]),
            dt=float(config["data"]["dt"]),
            time_input_scale=float(config["data"].get("time_input_scale", 1.0)),
        )
        for case in cases
    ]
    n_cases = len(cases)
    n_train = max(1, int(math.ceil(n_cases * float(config["data"].get("train_fraction", 0.6)))))
    n_val = max(1, int(math.ceil(n_cases * float(config["data"].get("val_fraction", 0.2)))))
    if n_train + n_val >= n_cases:
        n_train = max(1, n_cases - 2)
        n_val = 1
    split_case_ids = {
        "train": case_ids[:n_train],
        "val": case_ids[n_train : n_train + n_val],
        "test": case_ids[n_train + n_val :],
    }
    if not split_case_ids["test"]:
        raise ValueError("Need at least one held-out test case")

    def concat(name: str, split: str) -> np.ndarray:
        selected = [built[i][name] for i, case_id in enumerate(case_ids) if case_id in split_case_ids[split]]
        return np.concatenate(selected, axis=0)

    train_branch_raw = concat("branch_raw", "train")
    train_target_raw = concat("target_raw", "train")
    train_mask = concat("mask", "train")
    branch_mean, branch_std = robust_stats(train_branch_raw, axis=0)
    # Per-channel field scaling is computed after physical nondimensionalization and only on training cases.
    field_values = train_target_raw[train_mask.astype(bool)]
    field_mean, field_std = robust_stats(field_values, axis=0)

    normalization = {
        "coordinate": {
            "x_star": "(x - cylinder_center_x) / D",
            "y_star": "(y - cylinder_center_y) / D",
            "time_star": "t * Uref / D; dt from upstream CFDBench loader = 0.1",
            "time_network_input": f"time_star / {float(config['data'].get('time_input_scale', 1.0))}; physical t* is retained in metadata",
            "query_grid_source": "regular grid spanning the union of case.json domains; bilinear from 64x64 release",
        },
        "field_physical": {
            "u_star": "u / Uref",
            "v_star": "v / Uref",
            "pressure": "not present in selected CFDBench interpolated members",
        },
        "branch_features": BRANCH_FEATURES,
        "branch_mean": branch_mean.tolist(),
        "branch_std": branch_std.tolist(),
        "field_mean": field_mean.tolist(),
        "field_std": field_std.tolist(),
        "note": "Branch/field standardization is global over training cases only; it does not replace physical nondimensionalization.",
    }
    split_manifest = {
        "case_splits": split_case_ids,
        "case_count": n_cases,
        "sample_count": {split: int(sum(len(built[i]["frame_index"]) for i, case_id in enumerate(case_ids) if case_id in ids)) for split, ids in split_case_ids.items()},
        "leakage_policy": "whole CFD cases are assigned to exactly one split; time/space points from a case never cross splits",
    }
    write_json(PROJECT_ROOT / "splits" / "case_splits.json", split_manifest)
    write_json(PROJECT_ROOT / "metadata" / "normalization.json", normalization)

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["schema"] = "OperatorDataset-v1"
        handle.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["source_dataset"] = "CFDBench interpolated cylinder/bc subset"
        handle.attrs["dimension"] = 2
        handle.attrs["target_variables"] = write_json_attr(["u", "v"])
        handle.attrs["branch_features"] = write_json_attr(BRANCH_FEATURES)
        handle.attrs["normalization"] = write_json_attr(normalization)
        handle.attrs["split_manifest"] = write_json_attr(split_manifest)
        handle.attrs["query_shape"] = write_json_attr([int(config["data"]["query_ny"]), int(config["data"]["query_nx"])])
        handle.create_dataset("query_xy", data=query_xy.astype(np.float32), compression="gzip")
        for split in ("train", "val", "test"):
            group = handle.create_group(split)
            branch_raw = concat("branch_raw", split)
            target_raw = concat("target_raw", split)
            group.create_dataset("branch", data=((branch_raw - branch_mean) / branch_std).astype(np.float32), compression="gzip")
            group.create_dataset("trunk", data=concat("trunk", split).astype(np.float32), compression="gzip")
            group.create_dataset("target", data=((target_raw - field_mean) / field_std).astype(np.float32), compression="gzip")
            group.create_dataset("mask", data=concat("mask", split).astype(np.float32), compression="gzip")
            group.create_dataset("case_id", data=concat("case_id", split), compression="gzip")
            group.create_dataset("frame_index", data=concat("frame_index", split), compression="gzip")
        handle.flush()

    case_metadata = []
    for case in cases:
        case_metadata.append(
            {
                "case_id": case["case_id"],
                "parameters": {key: case[key] for key in ("vel_in", "density", "viscosity", "D", "Re", "domain_width", "domain_height")},
                "raw_params": case["params"],
                "raw_shapes": {"u": list(case["u"].shape), "v": list(case["v"].shape)},
            }
        )
    write_json(PROJECT_ROOT / "metadata" / "case_audit.json", {"cases": case_metadata, "normalization": normalization})
    print(json.dumps({"output": str(output), "splits": split_manifest, "normalization": normalization}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
