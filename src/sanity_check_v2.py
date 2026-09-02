"""Run v2 dataset, split, time-coverage, and checkpoint reload checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for OPTIONAL_DEPS in (PROJECT_ROOT / "work" / "pydeps", PROJECT_ROOT.parent.parent / "work" / "pydeps"):
    if OPTIONAL_DEPS.exists():
        sys.path.append(str(OPTIONAL_DEPS))

import h5py
import numpy as np
import torch

from cfd_pretrain.common import read_json_attr, resolve_path, write_json
from cfd_pretrain.model_v2 import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="processed_v2/operator_dataset_v2.h5")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--output", default="reports_v2/sanity_report_v2.json")
    args = parser.parse_args()

    data_path = resolve_path(args.data)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else None
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    with h5py.File(data_path, "r") as handle:
        schema = str(handle.attrs["schema"])
        checks["schema_v2"] = schema == "OperatorDataset-v2"
        split_manifest = read_json_attr(handle.attrs["split_manifest"])
        normalization = read_json_attr(handle.attrs["normalization"])
        details["schema"] = schema
        details["split_manifest"] = split_manifest
        split_arrays: dict[str, dict[str, np.ndarray]] = {}
        arrays_ok = True
        for split in ("train", "val", "test"):
            group = handle[split]
            split_arrays[split] = {name: group[name][:] for name in ("branch", "trunk", "target", "mask", "case_id", "frame_index", "time_norm", "time_star")}
            for name, array in split_arrays[split].items():
                if not np.isfinite(array).all():
                    arrays_ok = False
                    details.setdefault("nonfinite", []).append(f"{split}/{name}")
            if group["branch"].shape[0] != group["target"].shape[0] or group["trunk"].shape[0] != group["target"].shape[0]:
                arrays_ok = False
        checks["no_nan_inf_and_shapes"] = arrays_ok

        manifest_sets = {split: set(int(value) for value in split_manifest["case_splits"][split]) for split in ("train", "val", "test")}
        actual_sets = {split: set(int(value) for value in split_arrays[split]["case_id"].tolist()) for split in ("train", "val", "test")}
        checks["case_split_disjoint"] = not bool(manifest_sets["train"] & manifest_sets["val"] or manifest_sets["train"] & manifest_sets["test"] or manifest_sets["val"] & manifest_sets["test"])
        checks["case_split_matches_h5"] = actual_sets == manifest_sets
        checks["test_type_partition"] = set(split_manifest["interpolation_test_case_ids"]) | set(split_manifest["extrapolation_test_case_ids"]) == manifest_sets["test"] and not (set(split_manifest["interpolation_test_case_ids"]) & set(split_manifest["extrapolation_test_case_ids"]))

        time_ok = True
        time_details: dict[str, dict[str, float]] = {}
        for split, arrays in split_arrays.items():
            for case_id in sorted(set(int(value) for value in arrays["case_id"].tolist())):
                selected = arrays["case_id"] == case_id
                time_min = float(np.min(arrays["time_norm"][selected]))
                time_max = float(np.max(arrays["time_norm"][selected]))
                time_details[str(case_id)] = {"min": time_min, "max": time_max, "split": split}
                if time_min > 1e-6 or time_max < 1.0 - 1e-6:
                    time_ok = False
        checks["time_covers_full_case_interval"] = time_ok
        details["time_coverage"] = time_details
        query_xy = handle["query_xy"][:]
        checks["coordinates_finite"] = bool(np.isfinite(query_xy).all() and np.isfinite(split_arrays["train"]["trunk"]).all())
        checks["mask_binary"] = bool(np.isin(split_arrays["train"]["mask"], [0.0, 1.0]).all())
        field_mean = np.asarray(normalization["field_mean"], dtype=np.float32)
        field_std = np.asarray(normalization["field_std"], dtype=np.float32)
        recovered = split_arrays["train"]["target"][0] * field_std + field_mean
        checks["normalization_inverse_finite"] = bool(np.isfinite(recovered).all())
        details["query_shape"] = list(handle["train/trunk"].shape[1:])
        details["branch_features"] = read_json_attr(handle.attrs["branch_features"])
        details["trunk_features"] = read_json_attr(handle.attrs["trunk_features"])

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        checks["checkpoint_reload"] = True
        details["checkpoint_model_config"] = checkpoint["model_config"]
        details["checkpoint_parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
    else:
        checks["checkpoint_reload"] = not args.require_checkpoint
        details["checkpoint_missing"] = str(checkpoint_path) if checkpoint_path else "not requested"

    report = {"checks": checks, "all_pass": bool(all(checks.values())), "details": details}
    output = resolve_path(args.output)
    write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
