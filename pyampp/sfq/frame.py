from __future__ import annotations

import time
import numpy as np


def sfq_frame(
    s: dict,
    solis: int | None = None,
    hmi: bool = False,
    silent: bool = False,
    acute: bool = False,
    pot_vmag_func=None,
    sfq_step1_func=None,
    sfq_clean_func=None,
) -> dict:
    if pot_vmag_func is None or sfq_step1_func is None or sfq_clean_func is None:
        raise NotImplementedError("sfq_frame requires pot_vmag_func, sfq_step1_func, and sfq_clean_func.")

    t0 = time.time()
    if not silent:
        print("starting precise potential field calculating")
    pot = pot_vmag_func(s, simple=True)
    if not silent:
        print(f"Potential field calculation complete in {int(round(time.time() - t0))} seconds")

    s = sfq_step1_func(s, pot, silent=silent, acute=acute)

    by = np.asarray(s["t1"], dtype=float)
    bz = np.asarray(s["t2"], dtype=float)
    bo = 10
    ny, nx = by.shape
    by_pad = np.zeros((ny + 2 * bo, nx + 2 * bo), dtype=float)
    bz_pad = np.zeros((ny + 2 * bo, nx + 2 * bo), dtype=float)
    by_pad[bo:bo + ny, bo:bo + nx] = by
    bz_pad[bo:bo + ny, bo:bo + nx] = bz

    solis_flag = 0 if solis is None else int(solis)
    if hmi:
        solis_flag = 1
    if solis_flag != 0:
        solis_flag = 1

    cleaned = sfq_clean_func(by_pad, bz_pad, mode=(solis_flag == 1))
    if cleaned is not None:
        by_pad, bz_pad = cleaned

    s["t1"] = by_pad[bo:bo + ny, bo:bo + nx]
    s["t2"] = bz_pad[bo:bo + ny, bo:bo + nx]
    return s
