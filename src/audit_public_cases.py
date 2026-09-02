"""Audit all case metadata in a CFDBench cylinder archive via HTTP ranges."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for OPTIONAL_DEPS in (PROJECT_ROOT / "work" / "pydeps", PROJECT_ROOT.parent.parent / "work" / "pydeps"):
    if OPTIONAL_DEPS.exists():
        sys.path.append(str(OPTIONAL_DEPS))

from cfd_pretrain.common import resolve_path, write_json
from cfd_pretrain.zip_range import RangeZipReader


SUBSET_URLS = {
    "bc": "https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cylinder/bc.zip",
    "prop": "https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cylinder/prop.zip",
    "geo": "https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cylinder/geo.zip",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=sorted(SUBSET_URLS), default="bc")
    parser.add_argument("--output", default="metadata/cfdb_case_metadata_audit.json")
    args = parser.parse_args()

    reader = RangeZipReader(SUBSET_URLS[args.subset])
    entries = reader.entries()
    pattern = re.compile(r"^case(\d{4})/(case\.json|u\.npy|v\.npy)$")
    grouped: dict[int, dict[str, object]] = {}
    for entry in entries:
        match = pattern.match(entry.name)
        if match:
            grouped.setdefault(int(match.group(1)), {})[match.group(2)] = entry

    cases: list[dict[str, object]] = []
    for case_id in sorted(grouped):
        case_json_entry = grouped[case_id].get("case.json")
        if case_json_entry is None:
            continue
        params = json.loads(reader.read_entry(case_json_entry).decode("utf-8"))
        radius = float(params["radius"])
        vel_in = float(params["vel_in"])
        density = float(params["density"])
        viscosity = float(params["viscosity"])
        diameter = 2.0 * radius
        derived = {
            "U_inlet": vel_in,
            "D": diameter,
            "Re": density * vel_in * diameter / max(viscosity, 1e-12),
            "domain_width": float(params["x_max"]) - float(params["x_min"]),
            "domain_height": float(params["y_max"]) - float(params["y_min"]),
        }
        members = {
            name: {
                "archive_member": entry.name,
                "compressed_size": entry.compressed_size,
                "uncompressed_size": entry.uncompressed_size,
                "compression": entry.compression,
            }
            for name, entry in grouped[case_id].items()
        }
        cases.append({"case_id": case_id, "params": params, "derived": derived, "members": members})

    audit = {
        "source": "CFDBench interpolated cylinder archive metadata audit",
        "subset": args.subset,
        "archive_url": SUBSET_URLS[args.subset],
        "archive_audit": reader.audit(),
        "archive_entries_total": len(entries),
        "case_count": len(cases),
        "case_ids": [int(case["case_id"]) for case in cases],
        "cases": cases,
    }
    output = resolve_path(args.output)
    write_json(output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
