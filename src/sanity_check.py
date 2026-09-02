"""Run data, split, normalization, and checkpoint sanity tests."""

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

from cfd_pretrain.common import PROJECT_ROOT as PACKAGE_ROOT, read_json_attr, resolve_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="processed/operator_dataset.h5")
    parser.add_argument("--checkpoint", default="checkpoints/public_pretrain/best.pt")
    parser.add_argument("--require-checkpoint", action="store_true")
    args = parser.parse_args()
    data_path = resolve_path(args.data)
    checkpoint_path = resolve_path(args.checkpoint)
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    with h5py.File(data_path, "r") as handle:
        splits = read_json_attr(handle.attrs["split_manifest"])
        normalization = read_json_attr(handle.attrs["normalization"])
        details["schema"] = handle.attrs["schema"]
        details["splits"] = splits
        arrays_ok = True
        for split in ("train", "val", "test"):
            group = handle[split]
            for name in ("branch", "trunk", "target", "mask", "case_id", "frame_index"):
                array = group[name][:]
                if not np.isfinite(array).all():
                    arrays_ok = False
                    details.setdefault("nonfinite", []).append(f"{split}/{name}")
            if group["branch"].shape[0] != group["target"].shape[0] or group["trunk"].shape[0] != group["target"].shape[0]:
                arrays_ok = False
        checks["no_nan_inf"] = arrays_ok
        case_sets = {split: set(int(x) for x in splits["case_splits"][split]) for split in ("train", "val", "test")}
        checks["case_split_disjoint"] = not bool(case_sets["train"] & case_sets["val"] or case_sets["train"] & case_sets["test"] or case_sets["val"] & case_sets["test"])
        checks["case_split_complete"] = case_sets["train"] | case_sets["val"] | case_sets["test"] == set(int(x) for x in splits["case_splits"]["train"] + splits["case_splits"]["val"] + splits["case_splits"]["test"])
        query_xy = handle["query_xy"][:]
        checks["coordinate_finite"] = bool(np.isfinite(query_xy).all())
        checks["mask_binary"] = bool(np.isin(handle["train/mask"][:], [0.0, 1.0]).all())
        target = handle["train/target"][0]
        mean = np.asarray(normalization["field_mean"], dtype=np.float32)
        std = np.asarray(normalization["field_std"], dtype=np.float32)
        recovered = target * std + mean
        checks["normalization_inverse_finite"] = bool(np.isfinite(recovered).all())
        details["target_inverse_sample_minmax"] = [float(recovered.min()), float(recovered.max())]
        details["query_xy_minmax"] = [query_xy.min(axis=0).tolist(), query_xy.max(axis=0).tolist()]
    if checkpoint_path.exists():
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checks["checkpoint_reload"] = True
        details["checkpoint_model_config"] = checkpoint.get("model_config")
    else:
        checks["checkpoint_reload"] = not args.require_checkpoint
        details["checkpoint_missing"] = str(checkpoint_path)
    report = {"checks": checks, "all_pass": bool(all(checks.values())), "details": details}
    write_json(PACKAGE_ROOT / "reports" / "sanity_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
