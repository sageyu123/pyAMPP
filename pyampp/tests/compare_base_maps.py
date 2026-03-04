#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from scipy.io import readsav


@dataclass
class Metrics:
    mae: float
    rmse: float
    max_abs: float
    corr: float
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float


def _as_ndarray(value) -> np.ndarray:
    out = np.asarray(value)
    if out.dtype.byteorder not in ("=", "|"):
        out = out.byteswap().view(out.dtype.newbyteorder("="))
    return out


def _metrics(a: np.ndarray, b: np.ndarray) -> Metrics:
    da = _as_ndarray(a).astype(np.float64, copy=False)
    db = _as_ndarray(b).astype(np.float64, copy=False)
    diff = da - db
    finite = np.isfinite(da) & np.isfinite(db)
    if not np.any(finite):
        return Metrics(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    aa = da[finite]
    bb = db[finite]
    dd = diff[finite]
    corr = np.nan
    if aa.size > 1 and np.nanstd(aa) > 0 and np.nanstd(bb) > 0:
        corr = float(np.corrcoef(aa, bb)[0, 1])
    return Metrics(
        mae=float(np.nanmean(np.abs(dd))),
        rmse=float(np.sqrt(np.nanmean(dd * dd))),
        max_abs=float(np.nanmax(np.abs(dd))),
        corr=corr,
        mean_a=float(np.nanmean(aa)),
        mean_b=float(np.nanmean(bb)),
        std_a=float(np.nanstd(aa)),
        std_b=float(np.nanstd(bb)),
    )


def _load_h5_base(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        if "base" not in f:
            raise KeyError(f"'base' group missing in {path}")
        out = {}
        for key in ("bx", "by", "bz", "ic", "chromo_mask"):
            if key in f["base"]:
                out[key] = _as_ndarray(f["base"][key][:])
        return out


def _load_sav_base(path: Path) -> Dict[str, np.ndarray]:
    data = readsav(str(path), verbose=False)
    if "box" in data:
        box = data["box"].flat[0]
    elif "pbox" in data:
        box = data["pbox"].flat[0]
    else:
        raise KeyError(f"No 'box' or 'pbox' structure found in {path}")

    base = box["base"]
    if isinstance(base, np.ndarray) and base.size:
        b0 = base.flat[0]
    else:
        raise KeyError(f"Invalid BASE structure in {path}")

    out = {}
    for key in ("BX", "BY", "BZ", "IC", "CHROMO_MASK"):
        if key in b0.dtype.names:
            out[key.lower()] = _as_ndarray(b0[key])
    return out


def _best_simple_transform(h5_map: np.ndarray, sav_map: np.ndarray, signed: bool) -> Tuple[str, Metrics]:
    variants = {
        "direct": sav_map,
        "T": sav_map.T,
        "flipud": np.flipud(sav_map),
        "fliplr": np.fliplr(sav_map),
        "T_flipud": np.flipud(sav_map.T),
        "T_fliplr": np.fliplr(sav_map.T),
    }
    if signed:
        variants.update({f"{name}_neg": -arr for name, arr in list(variants.items())})

    best_name = "none"
    best_metrics = Metrics(np.inf, np.inf, np.inf, np.nan, np.nan, np.nan, np.nan, np.nan)
    for name, arr in variants.items():
        if arr.shape != h5_map.shape:
            continue
        m = _metrics(h5_map, arr)
        if m.mae < best_metrics.mae:
            best_name, best_metrics = name, m
    return best_name, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare HDF5 base maps against IDL SAV BASE maps.")
    parser.add_argument("--h5", required=True, type=Path, help="Path to CHR/NAS/POT HDF5 file.")
    parser.add_argument("--sav", required=True, type=Path, help="Path to IDL SAV model file.")
    parser.add_argument("--out", type=Path, help="Optional output JSON report path.")
    args = parser.parse_args()

    h5_base = _load_h5_base(args.h5)
    sav_base = _load_sav_base(args.sav)

    report = {
        "h5": str(args.h5),
        "sav": str(args.sav),
        "direct": {},
        "best_simple_transform": {},
        "horizontal_magnitude": {},
        "notes": [
            "best_simple_transform searches transpose/flip/sign variants of each SAV base map independently.",
            "Large Bx/By mismatch with good Bz/IC agreement often points to horizontal-vector convention or disambiguation differences.",
        ],
    }

    for key in ("bx", "by", "bz", "ic", "chromo_mask"):
        if key not in h5_base or key not in sav_base:
            continue
        if h5_base[key].shape != sav_base[key].shape:
            report["direct"][key] = {
                "h5_shape": list(h5_base[key].shape),
                "sav_shape": list(sav_base[key].shape),
                "error": "shape mismatch",
            }
        else:
            m = _metrics(h5_base[key], sav_base[key])
            report["direct"][key] = m.__dict__

        tname, tm = _best_simple_transform(
            h5_base[key],
            sav_base[key],
            signed=key in ("bx", "by", "bz"),
        )
        report["best_simple_transform"][key] = {
            "transform": tname,
            **tm.__dict__,
        }

    if all(k in h5_base for k in ("bx", "by")) and all(k in sav_base for k in ("bx", "by")):
        hmag = np.sqrt(h5_base["bx"] ** 2 + h5_base["by"] ** 2)
        smag = np.sqrt(sav_base["bx"] ** 2 + sav_base["by"] ** 2)
        if hmag.shape == smag.shape:
            report["horizontal_magnitude"] = _metrics(hmag, smag).__dict__
        else:
            report["horizontal_magnitude"] = {
                "h5_shape": list(hmag.shape),
                "sav_shape": list(smag.shape),
                "error": "shape mismatch",
            }

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote report: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
