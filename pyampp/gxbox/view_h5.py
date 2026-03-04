#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import warnings
try:
    from pyvista import PyVistaDeprecationWarning
except Exception:
    PyVistaDeprecationWarning = DeprecationWarning
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import Heliocentric, get_earth
from sunpy.sun import constants as sun_consts

from pyampp.gxbox.boxutils import read_b3d_h5
from pyampp.gxbox.magfield_viewer import MagFieldViewer
from PyQt5.QtWidgets import QApplication, QFileDialog


@dataclass
class SimpleBox:
    dims_pix: np.ndarray
    res: u.Quantity
    b3d: dict
    _frame_obs: object
    _center: SkyCoord

    @property
    def grid_coords(self):
        dims = self.dims_pix
        dx = self.res
        x = np.linspace(-dims[0] / 2, dims[0] / 2, dims[0]) * dx
        y = np.linspace(-dims[1] / 2, dims[1] / 2, dims[1]) * dx
        z = np.linspace(-dims[2] / 2, dims[2] / 2, dims[2]) * dx
        return {"x": x, "y": y, "z": z, "frame": self._frame_obs}


def infer_dims(b3d: dict) -> np.ndarray:
    for key in ("corona", "nlfff", "pot"):
        if key in b3d and "bx" in b3d[key]:
            return np.array(b3d[key]["bx"].shape, dtype=int)
    if "chromo" in b3d:
        if "bcube" in b3d["chromo"]:
            return np.array(b3d["chromo"]["bcube"].shape[:3], dtype=int)
        if "chromo_bcube" in b3d["chromo"]:
            return np.array(b3d["chromo"]["chromo_bcube"].shape[1:4], dtype=int)
    raise ValueError("Unable to infer dimensions from HDF5.")


def infer_time(b3d: dict) -> Time:
    if "chromo" in b3d and "attrs" in b3d["chromo"]:
        attrs = b3d["chromo"]["attrs"]
        if "obs_time" in attrs:
            try:
                return Time(attrs["obs_time"])
            except Exception:
                pass
    return Time.now()


def infer_res(b3d: dict) -> u.Quantity:
    if "corona" in b3d and "dr" in b3d["corona"]:
        dr = b3d["corona"]["dr"]
        if dr is not None and np.size(dr) >= 1:
            return (dr[0] * sun_consts.radius.to(u.km).value) * u.km
    if "chromo" in b3d and "dr" in b3d["chromo"]:
        dr = b3d["chromo"]["dr"]
        if dr is not None and np.size(dr) >= 1:
            return (dr[0] * sun_consts.radius.to(u.km).value) * u.km
    return 1.0 * u.Mm


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a saved HDF5 model in the 3D viewer without recomputing.")
    parser.add_argument("h5_path", nargs="?", help="Path to the HDF5 model file (positional).")
    parser.add_argument("--h5", dest="h5_opt", help="Path to the HDF5 model file.")
    parser.add_argument("--dir", dest="start_dir", help="Initial directory for file picker when no model path is given.")
    parser.add_argument("--pick", action="store_true", help="Open file picker even when model path is provided.")
    args = parser.parse_args()

    h5_arg = args.h5_opt or args.h5_path
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication([])
        owns_app = True

    if args.pick or not h5_arg:
        start_dir = Path(args.start_dir).expanduser() if args.start_dir else Path.cwd()
        if not start_dir.exists() or not start_dir.is_dir():
            start_dir = Path.cwd()
        dialog = QFileDialog(None, "Open HDF5 Model")
        # Native macOS picker may ignore selectFile(); use Qt dialog for reliable preselection.
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("HDF5 Files (*.h5);;All Files (*)")
        if h5_arg:
            candidate = Path(h5_arg).expanduser()
            dialog.setDirectory(str(candidate.parent if candidate.parent.exists() else start_dir))
            dialog.selectFile(str(candidate.name))
        else:
            dialog.setDirectory(str(start_dir))
        if not dialog.exec_():
            return 0
        selected = dialog.selectedFiles()
        if not selected:
            return 0
        h5_arg = selected[0]

    h5_path = Path(h5_arg).expanduser().resolve()
    b3d = read_b3d_h5(str(h5_path))

    dims = infer_dims(b3d)
    obs_time = infer_time(b3d)
    res = infer_res(b3d)

    frame = Heliocentric(observer=get_earth(obs_time), obstime=obs_time)
    center = SkyCoord(0 * u.Mm, 0 * u.Mm, 0 * u.Mm, frame=frame)
    box = SimpleBox(dims_pix=dims, res=res.to(u.Mm), b3d=b3d, _frame_obs=frame, _center=center)

    if "corona" in b3d:
        b3dtype = "corona"
    elif "nlfff" in b3d:
        b3dtype = "corona"
        b3d["corona"] = b3d.pop("nlfff")
    elif "pot" in b3d:
        b3dtype = "corona"
        b3d["corona"] = b3d.pop("pot")
    elif "chromo" in b3d:
        b3dtype = "chromo"
        chromo = b3d.get("chromo", {})
        if "bx" not in chromo and "bcube" in chromo:
            bcube = chromo["bcube"]
            if bcube.ndim == 4 and bcube.shape[-1] == 3:
                chromo["bx"] = bcube[:, :, :, 0]
                chromo["by"] = bcube[:, :, :, 1]
                chromo["bz"] = bcube[:, :, :, 2]
                b3d["chromo"] = chromo
    else:
        raise ValueError("No known model types found in HDF5 (expected corona/chromo).")

    warnings.filterwarnings("ignore", category=PyVistaDeprecationWarning)
    viewer = MagFieldViewer(box, time=obs_time, b3dtype=b3dtype, parent=None)
    if hasattr(viewer, "app_window"):
        viewer.app_window.setWindowTitle(f"GxBox 3D viewer - {h5_path}")
    viewer.show()
    if owns_app:
        app.exec_()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
