"""Download a small, auditable CFDBench cylinder subset via HTTP ranges."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cfd_pretrain.common import PROJECT_ROOT, resolve_path, sha256_file, write_json
from cfd_pretrain.zip_range import RangeZipReader


SUBSETS = {
    "bc": {
        "archive": "https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cylinder/bc.zip",
        "description": "cylinder flow with varied boundary-condition/inlet settings",
    },
    "prop": {
        "archive": "https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cylinder/prop.zip",
        "description": "cylinder flow with varied physical properties",
    },
    "geo": {
        "archive": "https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cylinder/geo.zip",
        "description": "cylinder flow with varied geometry settings",
    },
}


def parse_case_ids(value: str) -> list[int]:
    ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = token.split("-", 1)
            ids.extend(range(int(first), int(last) + 1))
        else:
            ids.append(int(token))
    return sorted(set(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=sorted(SUBSETS), default="bc")
    parser.add_argument("--case-ids", default="0,1,2,3,4", help="comma/range list, e.g. 0-4")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--raw-root", default=None, help="override raw extraction directory")
    parser.add_argument("--manifest-output", default=None, help="override provenance manifest path")
    args = parser.parse_args()

    case_ids = parse_case_ids(args.case_ids)
    if not case_ids:
        raise SystemExit("No case ids selected")
    spec = SUBSETS[args.subset]
    reader = RangeZipReader(spec["archive"])
    entries = reader.entries()

    pattern = re.compile(r"^case(\d{4})/(case\.json|u\.npy|v\.npy)$")
    selected: dict[int, dict[str, object]] = {}
    for entry in entries:
        match = pattern.match(entry.name)
        if match:
            case_id = int(match.group(1))
            selected.setdefault(case_id, {})[match.group(2)] = entry
    available = sorted(selected)
    if args.list_only:
        print(json.dumps({"subset": args.subset, "archive": spec["archive"], "available_case_ids": available}, indent=2))
        return
    missing = [case_id for case_id in case_ids if case_id not in selected]
    if missing:
        raise SystemExit(f"Requested cases are not present: {missing}; first available: {available[:20]}")

    raw_root = resolve_path(args.raw_root) if args.raw_root else PROJECT_ROOT / "raw" / "CFDBench" / f"cylinder_{args.subset}"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": "CFDBench interpolated cylinder subset",
        "dataset_url": "https://huggingface.co/datasets/chen-yingfa/CFDBench",
        "archive_url": spec["archive"],
        "subset": args.subset,
        "subset_description": spec["description"],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "Apache-2.0 as declared by the Hugging Face dataset card; retain upstream citation",
        "citation": "Luo, Yining; Chen, Yingfa; Zhang, Zhen. CFDBench, arXiv:2310.05963 (2023).",
        "archive_audit": reader.audit(),
        "archive_entries_total": len(entries),
        "case_ids": case_ids,
        "members": [],
    }

    for case_id in case_ids:
        case_dir = raw_root / f"case{case_id:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for member_name in ("case.json", "u.npy", "v.npy"):
            entry = selected[case_id][member_name]
            target = case_dir / member_name
            reused = False
            if target.exists() and target.stat().st_size == entry.uncompressed_size:
                existing_crc = zlib.crc32(target.read_bytes()) & 0xFFFFFFFF
                if existing_crc == entry.crc32:
                    reused = True
            if not reused:
                payload = reader.read_entry(entry)  # type: ignore[arg-type]
                target.write_bytes(payload)
            shape = None
            dtype = None
            if target.suffix == ".npy":
                array = np.load(target, mmap_mode="r")
                shape = list(array.shape)
                dtype = str(array.dtype)
            manifest["members"].append(
                {
                    "case_id": case_id,
                    "name": member_name,
                    "archive_member": entry.name,
                    "compressed_size": entry.compressed_size,
                    "uncompressed_size": entry.uncompressed_size,
                    "sha256": sha256_file(target),
                    "shape": shape,
                    "dtype": dtype,
                    "reused_after_crc_check": reused,
                }
            )

    manifest_output = resolve_path(args.manifest_output) if args.manifest_output else PROJECT_ROOT / "raw" / "CFDBench" / "source_manifest.json"
    write_json(manifest_output, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
