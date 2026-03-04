#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from typing import Dict, Tuple

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


IDL_TO_H5 = {
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


def _load_box(sav_path: str):
    data = readsav(sav_path, verbose=False)
    if "box" not in data:
        raise ValueError("IDL .sav does not contain a 'box' variable.")
    return data["box"].flat[0]


def _prepare_array(idl_val, h5_shape: Tuple[int, ...], *, field_name: str) -> np.ndarray:
    arr = np.asarray(idl_val)
    if arr.shape == h5_shape:
        return arr

    # Flatten 3D to 1D (same size)
    if arr.ndim == 3 and len(h5_shape) == 1 and arr.size == int(np.prod(h5_shape)):
        return arr.reshape(h5_shape)

    # Move vector axis from first to last (IDL: (3, X, Y, Z) -> H5: (X, Y, Z, 3))
    if arr.ndim == 4 and len(h5_shape) == 4:
        if arr.shape[0] == h5_shape[-1] and arr.shape[1:] == h5_shape[:3]:
            return np.moveaxis(arr, 0, -1)

    # Reorder DZ: IDL (Z, X, Y) -> H5 (X, Y, Z)
    if arr.ndim == 3 and len(h5_shape) == 3 and arr.shape[::-1] == h5_shape:
        return np.transpose(arr, (1, 2, 0))

    if arr.size == int(np.prod(h5_shape)):
        return arr.reshape(h5_shape)

    # Handle 1D length mismatch by truncating to match HDF5 target
    if arr.ndim == 1 and len(h5_shape) == 1:
        if arr.size > h5_shape[0]:
            print(
                f"Warning: truncating {field_name} from {arr.size} to {h5_shape[0]} elements.",
                file=sys.stderr,
            )
            return arr[: h5_shape[0]]
        if arr.size < h5_shape[0]:
            raise ValueError(
                f"{field_name} has {arr.size} elements, smaller than HDF5 target {h5_shape[0]}."
            )

    raise ValueError(f"Cannot reshape IDL array {arr.shape} to HDF5 shape {h5_shape}.")


def _collect_idl_arrays(box) -> Dict[str, np.ndarray]:
    arrays = {}
    for idl_field, _ in IDL_TO_H5.items():
        arrays[idl_field] = box[idl_field]
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a pyAMPP-style HDF5 by copying a template and filling with IDL .sav data."
    )
    parser.add_argument("--template", required=True, help="Path to existing pyAMPP HDF5 template.")
    parser.add_argument("--sav", required=True, help="Path to IDL .sav file containing BOX.")
    parser.add_argument("--out", required=True, help="Output HDF5 path.")
    args = parser.parse_args()

    box = _load_box(args.sav)
    idl_arrays = _collect_idl_arrays(box)

    # Copy template to output (copy each top-level object)
    with h5py.File(args.template, "r") as src, h5py.File(args.out, "w") as dst:
        for name in src.keys():
            src.copy(name, dst)

    # Fill datasets in output
    with h5py.File(args.out, "r+") as f:
        for idl_field, h5_path in IDL_TO_H5.items():
            if h5_path not in f:
                print(f"Skip missing HDF5 dataset: {h5_path}", file=sys.stderr)
                continue
            ds = f[h5_path]
            arr = _prepare_array(idl_arrays[idl_field], ds.shape, field_name=idl_field)
            ds[...] = arr

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
