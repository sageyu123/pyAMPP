from __future__ import annotations

import time
import numpy as np

from .utils import idl_where


def sfq_step1(mag: dict, pot: dict, silent: bool = False, acute: bool = False) -> dict:
    t0 = time.time()
    if not silent:
        print("starting preliminary SFQ  disambiguation")

    out = dict(mag)
    out["t1"] = np.asarray(out["t1"], dtype=float).copy()
    out["t2"] = np.asarray(out["t2"], dtype=float).copy()
    p_t1 = np.asarray(pot["t1"], dtype=float)
    p_t2 = np.asarray(pot["t2"], dtype=float)

    ind = idl_where(out["t2"] < 0)
    if ind[0] != -1:
        idx = ind.astype(np.int64)
        t1f = out["t1"].ravel()
        t2f = out["t2"].ravel()
        t1f[idx] = -t1f[idx]
        t2f[idx] = -t2f[idx]
        out["t1"] = t1f.reshape(out["t1"].shape)
        out["t2"] = t2f.reshape(out["t2"].shape)

    if acute:
        au = out["t1"] * p_t1 + out["t2"] * p_t2
        ind = idl_where(au < 0)
        if ind[0] != -1:
            idx = ind.astype(np.int64)
            t1f = out["t1"].ravel()
            t2f = out["t2"].ravel()
            t1f[idx] = -t1f[idx]
            t2f[idx] = -t2f[idx]
            out["t1"] = t1f.reshape(out["t1"].shape)
            out["t2"] = t2f.reshape(out["t2"].shape)
        if not silent:
            print(f"preliminary SFQ  disambiguation complete in {int(round(time.time() - t0))} seconds")
        return out

    bpy_y = np.roll(p_t1, shift=-1, axis=0) - p_t1
    by_y = np.roll(out["t1"], shift=-1, axis=0) - out["t1"]
    au = (bpy_y - by_y) ** 2
    au_ = (bpy_y + by_y) ** 2

    bpy_y = np.roll(p_t1, shift=-1, axis=1) - p_t1
    by_y = np.roll(out["t1"], shift=-1, axis=1) - out["t1"]
    au = au + (bpy_y - by_y) ** 2
    au_ = au_ + (bpy_y + by_y) ** 2

    bpy_y = np.roll(p_t2, shift=-1, axis=0) - p_t2
    by_y = np.roll(out["t2"], shift=-1, axis=0) - out["t2"]
    au = au + (bpy_y - by_y) ** 2
    au_ = au_ + (bpy_y + by_y) ** 2

    bpy_y = np.roll(p_t2, shift=-1, axis=1) - p_t2
    by_y = np.roll(out["t2"], shift=-1, axis=1) - out["t2"]
    au = np.sqrt(au + (bpy_y - by_y) ** 2)
    au_ = np.sqrt(au_ + (bpy_y + by_y) ** 2)

    ind = idl_where(au > au_)
    if ind[0] != -1:
        idx = ind.astype(np.int64)
        t1f = out["t1"].ravel()
        t2f = out["t2"].ravel()
        t1f[idx] = -t1f[idx]
        t2f[idx] = -t2f[idx]
        out["t1"] = t1f.reshape(out["t1"].shape)
        out["t2"] = t2f.reshape(out["t2"].shape)

    if not silent:
        print(f"preliminary SFQ  disambiguation complete in {int(round(time.time() - t0))} seconds")
    return out
