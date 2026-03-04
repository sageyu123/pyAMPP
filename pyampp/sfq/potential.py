from __future__ import annotations

import time
import numpy as np

from .utils import idl_where, norm_vec, u_grid, u_grid_box, u_str_add


def get_str_mag(bx: np.ndarray, by: np.ndarray, bz: np.ndarray, apos: np.ndarray, rsun: float) -> dict:
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    bz = np.asarray(bz, dtype=float)
    n = bx.shape
    gr = u_grid(pos=np.asarray(apos, dtype=float), n=n)

    alf = float(rsun) * np.pi / 180.0 / 3600.0
    dsun = 1.0 / np.sin(alf)
    tx = np.tan(gr["x"] * np.pi / 180.0 / 3600.0)
    ty = np.tan(gr["y"] * np.pi / 180.0 / 3600.0)
    txy2 = tx * tx + ty * ty
    denom = txy2 + 1.0

    a = dsun * txy2 / denom
    b = (dsun * dsun * txy2 - 1.0) / denom
    det = a * a - b

    indin = idl_where(det >= 0.0)
    indout = idl_where(det < 0.0)

    x = np.zeros_like(gr["x"], dtype=float)
    if indin[0] != -1:
        in_mask = det >= 0.0
        x[in_mask] = a[in_mask] + np.sqrt(det[in_mask])
    if indout[0] != -1:
        out_mask = det < 0.0
        t = np.sqrt(txy2[out_mask])
        tan_alf = np.tan(alf)
        x[out_mask] = tan_alf * dsun * t / (tan_alf * t + 1.0)

    qs = {"l": np.float32(0.0), "b": np.float32(0.0), "p": np.float32(0.0), "r": np.float32(rsun), "type": "arc"}
    return {
        "x0": x.astype(np.float32),
        "x1": gr["x"].astype(np.float32),
        "x2": gr["y"].astype(np.float32),
        "t0": bz.astype(np.float32),
        "t1": bx.astype(np.float32),
        "t2": by.astype(np.float32),
        "pos": np.asarray(apos, dtype=np.float32),
        "qs": qs,
        "ts": dict(qs),
        "indin": indin,
        "indout": indout,
        "visible": idl_where(np.ones_like(x, dtype=bool)),
        "unvisible": np.int64(-1),
    }


def pex_bl_(si: dict, sh: bool = False, silent: bool = False, bin_value: int | float | None = None, pot_vmag_func=None) -> dict:
    if pot_vmag_func is None:
        raise NotImplementedError("pex_bl_ requires pot_vmag_func.")

    s = dict(si)
    sh_i = 1 if sh else 0
    bin_i = 0 if bin_value is None else float(bin_value)

    t0 = np.asarray(s["t0"], dtype=float)
    nx, ny = t0.shape
    sx = 256.0 if bin_i == 0 else 256.0 / bin_i
    sy = 256.0 if bin_i == 0 else 256.0 / bin_i
    npx = max(int(np.ceil(nx / sx)), 10)
    npy = max(int(np.ceil(ny / sy)), 10)
    sx = nx / npx
    sy = ny / npy

    indout_global = np.asarray(s.get("indout", np.array([-1], dtype=np.int64)))
    if indout_global.size > 0 and indout_global[0] != -1:
        t0f = t0.ravel()
        t0f[indout_global.astype(np.int64)] = 0.0
        t0 = t0f.reshape(t0.shape)
    s["t0"] = t0

    pot = dict(s)
    pot["t0"] = np.zeros_like(t0)
    pot = u_str_add(pot, "t1", np.zeros_like(t0))
    pot = u_str_add(pot, "t2", np.zeros_like(t0))

    ss_base = u_str_add(dict(s), "t1")
    ss_base = u_str_add(ss_base, "t2")
    wgt = np.zeros_like(t0)
    nparts = 2 * (npx + 2) * (npy + 2)
    oldcomplete = 0

    for i in range(-1, npx + 1):
        for j in range(-1, npy + 1):
            if not silent:
                complete = int(np.round(100.0 * float(j + 1 + (i + 1) * (npx + 2) + sh_i * nparts / 2.0) / nparts))
                if (complete > oldcomplete) and (complete % 10 == 0):
                    print(f"potential field calculating: {complete} % complete")
                oldcomplete = complete

            xst = min(max(int(i * sx + sx / 2.0 * sh_i), 0), nx - 1)
            xen = min(max(int((i + 1) * sx - 1.0 + sx / 2.0 * sh_i), 0), nx - 1)
            yst = min(max(int(j * sy + sy / 2.0 * sh_i), 0), ny - 1)
            yen = min(max(int((j + 1) * sy - 1.0 + sy / 2.0 * sh_i), 0), ny - 1)
            if xen < xst or yen < yst:
                continue

            ss = dict(ss_base)
            ss = u_str_add(ss, ["t0", "x0", "x1", "x2"], s["t0"][xst:xen + 1, yst:yen + 1], s["x0"][xst:xen + 1, yst:yen + 1], s["x1"][xst:xen + 1, yst:yen + 1], s["x2"][xst:xen + 1, yst:yen + 1])
            ss["pos"] = np.array([s["x1"][xst, yst], s["x2"][xst, yst], s["x1"][xen, yen], s["x2"][xen, yen]], dtype=float)

            dx = (ss["pos"][2] - ss["pos"][0]) / 2.0 or 1.0
            dy = (ss["pos"][3] - ss["pos"][1]) / 2.0 or 1.0
            xc = (ss["pos"][2] + ss["pos"][0]) / 2.0
            yc = (ss["pos"][3] + ss["pos"][1]) / 2.0
            wgts = np.exp(-6.0 * ((ss["x1"] - xc) / dx) ** 2 - 6.0 * ((ss["x2"] - yc) / dy) ** 2)

            radius = float(ss["qs"]["r"])
            alf = radius * np.pi / 180.0 / 3600.0
            dsun = 1.0 / np.sin(alf)
            tx = np.tan(ss["x1"] * np.pi / 180.0 / 3600.0)
            ty = np.tan(ss["x2"] * np.pi / 180.0 / 3600.0)
            tx2ty2 = tx * tx + ty * ty
            a = dsun * tx2ty2 / (tx2ty2 + 1.0)
            b = (dsun * dsun * tx2ty2 - 1.0) / (tx2ty2 + 1.0)
            det = a * a - b
            indin = idl_where(det >= 0.0)
            indout = idl_where(det < 0.0)
            ss = u_str_add(ss, ["indin", "indout"], indin, indout)

            if indout[0] != -1:
                t0_local = ss["t0"].ravel()
                t0_local[indout.astype(np.int64)] = 0.0
                ss["t0"] = t0_local.reshape(ss["t0"].shape)

            if int(np.sum(np.abs(ss["t0"]) > 0)) != 0 and ss["indin"].size > 1:
                ss1 = pot_vmag_func(ss, simple=True)
                if indout[0] != -1 and isinstance(ss1, dict):
                    for key in ("t0", "t1", "t2"):
                        arr = np.asarray(ss1[key], dtype=float).ravel()
                        arr[indout.astype(np.int64)] = 0.0
                        ss1[key] = arr.reshape(np.asarray(ss1[key]).shape)
                    wgtsf = wgts.ravel()
                    wgtsf[indout.astype(np.int64)] = 0.0
                    wgts = wgtsf.reshape(wgts.shape)
                if isinstance(ss1, dict):
                    pot["t0"][xst:xen + 1, yst:yen + 1] = ss1["t0"]
                    pot["t1"][xst:xen + 1, yst:yen + 1] = ss1["t1"]
                    pot["t2"][xst:xen + 1, yst:yen + 1] = ss1["t2"]
                    wgt[xst:xen + 1, yst:yen + 1] = wgts

    s["t0"] = pot["t0"]
    s["t1"] = pot["t1"]
    s["t2"] = pot["t2"]
    s = u_str_add(s, "wgt", wgt)
    return s


def pex_bl(s0: dict, silent: bool = False, pot_vmag_func=None) -> dict:
    if pot_vmag_func is None:
        raise NotImplementedError("pex_bl requires pot_vmag_func.")

    t0 = time.time()
    if not silent:
        print("starting precise potential field calculating in parts")
    s1 = pex_bl_(s0, silent=silent, pot_vmag_func=pot_vmag_func)
    s2 = pex_bl_(s0, sh=True, silent=silent, pot_vmag_func=pot_vmag_func)

    wgt = s1["wgt"] + s2["wgt"]
    tt0 = s1["t0"] * s1["wgt"] + s2["t0"] * s2["wgt"]
    tt1 = s1["t1"] * s1["wgt"] + s2["t1"] * s2["wgt"]
    tt2 = s1["t2"] * s1["wgt"] + s2["t2"] * s2["wgt"]

    ind = idl_where(wgt != 0)
    if ind[0] != -1:
        idx = ind.astype(np.int64)
        fw = wgt.ravel()
        f0, f1, f2 = tt0.ravel(), tt1.ravel(), tt2.ravel()
        f0[idx] = f0[idx] / fw[idx]
        f1[idx] = f1[idx] / fw[idx]
        f2[idx] = f2[idx] / fw[idx]
        tt0, tt1, tt2 = f0.reshape(tt0.shape), f1.reshape(tt1.shape), f2.reshape(tt2.shape)

    if not silent:
        print(f"Potential field calculation complete in {int(round(time.time() - t0))} seconds")
    return {"t1": tt1, "t2": tt2}


def pot_vmag(
    s: dict,
    outbpos: dict | None = None,
    fcenter: int = 1,
    simple: bool = False,
    alpha: float = 0.0,
    normal: bool = True,
    bpos: dict | None = None,
    sol_crd_func=None,
    qs_crd_func=None,
    a_field_func=None,
    get_fftplane_func=None,
    return_outbpos: bool = False,
):
    if sol_crd_func is None or qs_crd_func is None or a_field_func is None or get_fftplane_func is None:
        raise NotImplementedError("pot_vmag requires sol_crd_func, qs_crd_func, a_field_func, and get_fftplane_func.")

    l0, b0, p0 = (0.0, 0.0, 0.0) if normal else (float(s["qs"]["l"]), float(s["qs"]["b"]), float(s["qs"]["p"]))
    rad = float(s["qs"]["r"])
    bpos_local = None if bpos is None else dict(bpos)

    if fcenter != 0:
        indin = np.asarray(s.get("indin", np.array([-1], dtype=np.int64)))
        if indin.size == 0 or indin[0] == -1:
            raise ValueError("pot_vmag requires non-empty s['indin'] for fcenter != 0.")
        idx = indin.astype(np.int64)

        ua = {
            "x0": np.asarray(s["x0"], dtype=float).ravel()[idx],
            "x1": np.asarray(s["x1"], dtype=float).ravel()[idx],
            "x2": np.asarray(s["x2"], dtype=float).ravel()[idx],
            "qs": {"l": l0, "b": b0, "p": p0, "r": rad, "type": "arc"},
        }
        ub = {"qs": dict(ua["qs"])}
        ub["qs"]["type"] = "dec"
        u = sol_crd_func(ua, ub, crd=True)

        xc, yc, zc = float(np.mean(u["x0"])), float(np.mean(u["x1"])), float(np.mean(u["x2"]))
        if fcenter == 2:
            if "t2" in s and "t1" in s:
                t0m = np.asarray(s["t0"], dtype=float).ravel()[idx]
                t1m = np.asarray(s["t1"], dtype=float).ravel()[idx]
                t2m = np.asarray(s["t2"], dtype=float).ravel()[idx]
                b_mod = np.sqrt(t0m ** 2 + t1m ** 2 + t2m ** 2)
            else:
                b_mod = np.abs(np.asarray(s["t0"], dtype=float).ravel()[idx])
            denom = float(np.sum(b_mod))
            yc = float(np.sum(b_mod * np.asarray(u["x1"], dtype=float)) / denom) if denom else yc
            zc = float(np.sum(b_mod * np.asarray(u["x2"], dtype=float)) / denom) if denom else zc
            xc = float(np.sqrt(max(0.0, 1.0 - yc ** 2 - zc ** 2)))

        ad = qs_crd_func({"x0": xc, "x1": yc, "x2": zc}, l0, b0, p0, inv=True, tosph=True)
        lc = float(ad["x2"] * 180.0 / np.pi)
        bc = float(90.0 - ad["x1"] * 180.0 / np.pi)

        ad = qs_crd_func({"x0": u["x0"], "x1": u["x1"], "x2": u["x2"]}, l0, b0, p0, inv=True)
        sph = qs_crd_func(ad, lc, bc, 0.0, tosph=True)
        l = np.asarray(sph["x2"], dtype=float) * 180.0 / np.pi
        b = 90.0 - np.asarray(sph["x1"], dtype=float) * 180.0 / np.pi
        hs = np.array([np.max(np.abs(l)), np.max(np.abs(b))], dtype=float)

        rm = np.sqrt(np.asarray(u["x1"], dtype=float) ** 2 + np.asarray(u["x2"], dtype=float) ** 2)
        srt = np.argsort(rm)[:2]
        u0 = np.array([u["x0"][srt[0]], u["x1"][srt[0]], u["x2"][srt[0]]], dtype=float)
        u1 = np.array([u["x0"][srt[1]], u["x1"][srt[1]], u["x2"][srt[1]]], dtype=float)
        rob0, rob1 = norm_vec(u0), norm_vec(u1)
        pixs = float(np.arccos(np.clip(float(np.dot(u0, u1) / (rob0 * rob1)), -1.0, 1.0)))

        n_vec = np.maximum(np.fix(2.0 * hs * np.pi / 180.0 / pixs).astype(int), 5)
        hs = (n_vec - 1) * pixs * 180.0 / np.pi / 2.0
        bpos_local = {"l": lc, "b": bc, "hs": hs, "n": np.array([n_vec[0], n_vec[1], float(np.min(n_vec)) / 2.0], dtype=float)}

    if bpos_local is None:
        raise ValueError("bpos must be provided when fcenter == 0.")
    if outbpos is not None:
        outbpos.clear()
        outbpos.update(bpos_local)

    n = np.asarray(bpos_local["n"], dtype=float)[:2].astype(int)
    hs = np.asarray(bpos_local["hs"], dtype=float)
    grb = u_grid_box(np.array([-hs[0], -hs[1]], dtype=float), np.array([2.0 * hs[0], 2.0 * hs[1]], dtype=float), n)
    ph = grb["x"] * np.pi / 180.0
    th = grb["y"] * np.pi / 180.0
    r = np.zeros_like(ph)

    sa = {
        "a": {"x0": s["x0"], "x1": s["x1"], "x2": s["x2"], "t0": s["t0"], "qs": {"l": l0, "b": b0, "p": p0, "r": rad, "type": "arc"}, "pos": s.get("pos")},
        "b": {"qs": {"l": bpos_local["l"], "b": bpos_local["b"], "p": 0.0, "type": "sbox"}},
        "proc": "b_spl",
        "vec": 0,
    }
    q = a_field_func(ph, th, r, set=sa)
    if isinstance(q, dict) and q.get("err"):
        return 0

    set_obj = {"bl": q["t0"], "spos": {"L": l0, "b": b0, "p": p0}, "bpos": {"l": bpos_local["l"], "b": bpos_local["b"], "hs": hs}, "alfa": alpha}
    get_fftplane_func(set=set_obj, simple=simple)

    pos = np.array([-hs[0], -hs[1], hs[0], hs[1]], dtype=float) * np.pi / 180.0
    uu = get_fftplane_func(pos, n, 0.0, simple=simple)

    sa2 = {
        "a": {"x0": uu["x0"], "x1": uu["x1"], "x2": np.asarray(uu["x0"], dtype=float) * 0.0, "t0": uu["t0"], "t1": uu["t1"], "t2": uu["t2"], "qs": sa["b"]["qs"], "ts": sa["b"]["qs"], "pos": pos},
        "b": {"qs": sa["a"]["qs"], "ts": sa["a"]["qs"]},
        "proc": "b_spl",
        "vec": 1,
    }
    q1 = a_field_func(s["x0"], s["x1"], s["x2"], set=sa2)

    s1 = dict(s)
    s1["t0"] = q1["t0"]
    s1["t1"] = q1["t1"]
    s1["t2"] = q1["t2"]
    s1["type"] = f"vecp {s1['type']}" if "type" in s1 else "vecp"
    return (s1, bpos_local) if return_outbpos else s1
