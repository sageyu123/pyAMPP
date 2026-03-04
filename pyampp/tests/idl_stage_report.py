#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from scipy.io import readsav


STAGE_ORDER = ["NONE", "POT", "BND", "NAS", "GEN", "CHR"]


def stage_from_name(path: Path) -> str:
    name = path.name
    for stage in reversed(STAGE_ORDER):
        if f".{stage}." in name or name.endswith(f".{stage}.sav"):
            return stage
    return "UNKNOWN"


def get_box(data: dict) -> tuple[str | None, Any | None]:
    if "box" in data:
        return "box", data["box"].flat[0]
    if "pbox" in data:
        return "pbox", data["pbox"].flat[0]
    return None, None


def field_info(val: Any) -> Dict[str, Any]:
    if isinstance(val, np.ndarray) and val.dtype.names:
        return {"type": "struct", "shape": list(val.shape), "fields": list(val.dtype.names)}
    if isinstance(val, np.ndarray):
        return {"type": "array", "shape": list(val.shape), "dtype": str(val.dtype)}
    return {"type": "scalar", "dtype": type(val).__name__}


def struct_fields(val: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(val, np.ndarray) and val.dtype.names and val.size:
        v0 = val.flat[0]
        for f in val.dtype.names:
            out[f] = field_info(v0[f])
    return out


def summarize_box(box: Any) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for field in box.dtype.names:
        info = field_info(box[field])
        if info["type"] == "struct":
            info["members"] = struct_fields(box[field])
        fields[field] = info
    return fields


def diff_fields(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Any]:
    prev_keys = set(prev.keys())
    curr_keys = set(curr.keys())
    added = sorted(curr_keys - prev_keys)
    removed = sorted(prev_keys - curr_keys)
    changed = []
    for k in sorted(curr_keys & prev_keys):
        if prev[k] != curr[k]:
            changed.append(k)
    return {"added": added, "removed": removed, "changed": changed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a JSON report for IDL AMPP stage .sav files.")
    parser.add_argument("--dir", required=True, help="Directory containing .sav stage files.")
    parser.add_argument("--out", required=True, help="Output JSON file.")
    args = parser.parse_args()

    folder = Path(args.dir)
    files = sorted(folder.glob("*.sav"))
    files = sorted(
        files,
        key=lambda p: STAGE_ORDER.index(stage_from_name(p)) if stage_from_name(p) in STAGE_ORDER else 99,
    )

    report: Dict[str, Any] = {"stages": []}
    prev_fields = None
    prev_stage = None

    for path in files:
        data = readsav(str(path), verbose=False)
        box_name, box = get_box(data)
        entry: Dict[str, Any] = {
            "file": path.name,
            "stage": stage_from_name(path),
            "box_name": box_name,
        }
        if box is None:
            entry["error"] = "no_box_or_pbox"
            report["stages"].append(entry)
            continue
        fields = summarize_box(box)
        entry["fields"] = fields
        if prev_fields is not None:
            entry["diff_from_prev"] = diff_fields(prev_fields, fields)
            entry["prev_stage"] = prev_stage
        report["stages"].append(entry)
        prev_fields = fields
        prev_stage = entry["stage"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
