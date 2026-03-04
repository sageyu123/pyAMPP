#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

try:
    import h5py
except Exception as exc:  # pragma: no cover - ad-hoc script
    print("Missing dependency: h5py. Install with `python -m pip install h5py`.", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - ad-hoc script
    print("Missing dependency: numpy. Install with `python -m pip install numpy`.", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    from scipy.io import readsav
except Exception as exc:  # pragma: no cover - ad-hoc script
    print("Missing dependency: scipy. Install with `python -m pip install scipy`.", file=sys.stderr)
    raise SystemExit(1) from exc


CHR_IDL_TO_H5 = {
    "DR": "chromo/dr",
    "STARTIDX": "chromo/start_idx",
    "ENDIDX": "chromo/end_idx",
    "AVFIELD": "chromo/av_field",
    "PHYSLENGTH": "chromo/phys_length",
    "BCUBE": "chromo/bcube",
    "CHROMO_IDX": "chromo/chromo_idx",
    "CHROMO_BCUBE": "chromo/chromo_bcube",
    "N_HTOT": "chromo/n_htot",
    "N_HI": "chromo/n_hi",
    "N_P": "chromo/n_p",
    "DZ": "chromo/dz",
    "CHROMO_N": "chromo/chromo_n",
    "CHROMO_T": "chromo/chromo_t",
    "CHROMO_LAYERS": "chromo/chromo_layers",
    "TR": "chromo/tr",
    "TR_H": "chromo/tr_h",
    "CORONA_BASE": "chromo/corona_base",
}

NAS_IDL_TO_H5 = {
    "BX": "corona/bx",
    "BY": "corona/by",
    "BZ": "corona/bz",
}

POT_IDL_TO_H5 = {
    "BX": "corona/bx",
    "BY": "corona/by",
    "BZ": "corona/bz",
}


@dataclass
class FieldInfo:
    shape: Tuple[int, ...]
    dtype: str


def _h5_fields(path: str) -> Dict[str, FieldInfo]:
    fields: Dict[str, FieldInfo] = {}
    with h5py.File(path, "r") as f:
        for group in f.keys():
            for name, obj in f[group].items():
                if isinstance(obj, h5py.Dataset):
                    fields[f"{group}/{name}"] = FieldInfo(obj.shape, str(obj.dtype))
    return fields


def _load_idl_box(data: Dict[str, Any]):
    if "box" in data:
        return data["box"].flat[0]
    if "pbox" in data:
        return data["pbox"].flat[0]
    raise KeyError("No 'box' or 'pbox' found in SAV file.")


def _sav_fields(path: str) -> Dict[str, FieldInfo]:
    data = readsav(path, verbose=False)
    box = _load_idl_box(data)
    fields: Dict[str, FieldInfo] = {}
    for field in box.dtype.names:
        val = box[field]
        if isinstance(val, np.ndarray):
            fields[field] = FieldInfo(val.shape, str(val.dtype))
        else:
            fields[field] = FieldInfo((), type(val).__name__)
    return fields


def _sav_base_fields(path: str) -> Dict[str, FieldInfo]:
    data = readsav(path, verbose=False)
    box = _load_idl_box(data)
    base = box["BASE"]
    fields: Dict[str, FieldInfo] = {}
    if isinstance(base, np.ndarray) and base.size:
        b0 = base.flat[0]
        for field in base.dtype.names:
            val = b0[field]
            if isinstance(val, np.ndarray):
                fields[field] = FieldInfo(val.shape, str(val.dtype))
            else:
                fields[field] = FieldInfo((), type(val).__name__)
    return fields


def _note_shape(idl_shape: Tuple[int, ...], h5_shape: Tuple[int, ...]) -> str:
    if idl_shape == h5_shape:
        return ""
    if int(np.prod(idl_shape)) == int(np.prod(h5_shape)) and len(idl_shape) == 3 and len(h5_shape) == 1:
        return "flattened in HDF5"
    if len(idl_shape) == 4 and len(h5_shape) == 4 and sorted(idl_shape) == sorted(h5_shape):
        return "axis order differs"
    if len(idl_shape) == 3 and len(h5_shape) == 3 and sorted(idl_shape) == sorted(h5_shape):
        return "axis order differs"
    if int(np.prod(idl_shape)) == int(np.prod(h5_shape)):
        return "same size, different shape"
    return "shape differs"

def _infer_stage(h5_fields: Dict[str, FieldInfo], sav_fields: Dict[str, FieldInfo]) -> str:
    h5_keys = set(h5_fields.keys())
    if any(k.startswith("chromo/") for k in h5_keys):
        return "chr"
    if any(k.startswith("corona/") for k in h5_keys):
        return "nas"
    if any(k.startswith("nlfff/") for k in h5_keys):
        return "nas"
    if any(k.startswith("pot/") for k in h5_keys):
        return "pot"
    # Fallback to IDL fields if HDF5 is minimal
    if "BX" in sav_fields and "BY" in sav_fields and "BZ" in sav_fields:
        return "nas"
    return "unknown"

def _mapping_for_stage(stage: str) -> Dict[str, str]:
    if stage == "chr":
        return CHR_IDL_TO_H5
    if stage == "nas":
        return NAS_IDL_TO_H5
    if stage == "pot":
        return POT_IDL_TO_H5
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare IDL .sav BOX fields to HDF5 datasets.")
    parser.add_argument("--h5", required=True, help="Path to HDF5 file.")
    parser.add_argument("--sav", required=True, help="Path to IDL .sav file.")
    parser.add_argument("--out", help="Write JSON report to this path.")
    parser.add_argument("--stage", choices=["auto", "nas", "pot", "chr"], default="auto",
                        help="Stage mapping to use (default: auto-detect).")
    args = parser.parse_args()

    h5_fields = _h5_fields(args.h5)
    sav_fields = _sav_fields(args.sav)
    base_fields = _sav_base_fields(args.sav)

    stage = _infer_stage(h5_fields, sav_fields) if args.stage == "auto" else args.stage
    mapping_table = _mapping_for_stage(stage)

    mapping: List[Dict[str, Any]] = []
    for idl_field, h5_path in mapping_table.items():
        idl = sav_fields.get(idl_field)
        h5 = h5_fields.get(h5_path)
        note = ""
        if idl and h5:
            note = _note_shape(idl.shape, h5.shape)
        mapping.append(
            {
                "idl_field": idl_field,
                "h5_path": h5_path,
                "idl_shape": idl.shape if idl else None,
                "h5_shape": h5.shape if h5 else None,
                "idl_dtype": idl.dtype if idl else None,
                "h5_dtype": h5.dtype if h5 else None,
                "note": note,
            }
        )

    h5_only = sorted(set(h5_fields.keys()) - set(mapping_table.values()))
    idl_only = sorted(set(sav_fields.keys()) - set(mapping_table.keys()))

    report = {
        "stage": stage,
        "mapping": mapping,
        "hdf5_only": {k: h5_fields[k].__dict__ for k in h5_only},
        "idl_only": {k: sav_fields[k].__dict__ for k in idl_only},
        "idl_base_fields": {k: v.__dict__ for k, v in base_fields.items()},
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    else:
        print(json.dumps(report, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
