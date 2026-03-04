from __future__ import annotations

import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _conv1d_edge(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("_conv1d_edge expects a 2D array.")
    k = np.asarray(kernel, dtype=float)
    ksize = int(k.size)
    pad_l = ksize // 2
    pad_r = ksize - 1 - pad_l
    if axis == 0:
        padded = np.pad(arr, ((pad_l, pad_r), (0, 0)), mode="edge")
        out = np.empty_like(arr, dtype=float)
        for j in range(arr.shape[1]):
            out[:, j] = np.convolve(padded[:, j], k, mode="valid")
        return out
    if axis == 1:
        padded = np.pad(arr, ((0, 0), (pad_l, pad_r)), mode="edge")
        out = np.empty_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            out[i, :] = np.convolve(padded[i, :], k, mode="valid")
        return out
    raise ValueError("axis must be 0 or 1.")


def _box_smooth_edge(arr: np.ndarray, size: int) -> np.ndarray:
    kernel = np.ones(int(size), dtype=float) / float(size)
    return _conv1d_edge(_conv1d_edge(arr, kernel, axis=0), kernel, axis=1)


def _gaussf(n: int, sigma: float) -> np.ndarray:
    x = np.arange(int(n), dtype=float) - (int(n) - 1) / 2.0
    sigma = float(sigma) if sigma > 0 else 1.0
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    s = ker.sum()
    return ker / s if s > 0 else ker


def _pwf_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    n = int(np.ceil(float(sigma) * 3.0) * 2 + 1)
    ker = _gaussf(n, sigma)
    return _conv1d_edge(_conv1d_edge(arr, ker, axis=0), ker, axis=1)


def _median_filter_edge(arr: np.ndarray, size: int) -> np.ndarray:
    size = int(size)
    pad = size // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="edge")
    win = sliding_window_view(padded, (size, size))
    return np.median(win, axis=(-2, -1))


def sfq_clean(
    bx: np.ndarray,
    by: np.ndarray,
    s: int | float | None = None,
    gauss: bool = False,
    median: bool = False,
    show: bool = False,
    mode: int = 0,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    bxx = np.asarray(bx, dtype=float).copy()
    byy = np.asarray(by, dtype=float).copy()
    if bxx.shape != byy.shape:
        raise ValueError("bx and by must have the same shape.")

    if s is None:
        t0 = time.time()
        if not silent:
            print("Sarting SFQ cleaning")
        bxx, byy = sfq_clean(bxx, byy, 3, median=True, show=show, silent=True)
        mode_i = 1 if mode != 0 else 0
        if mode_i == 0:
            if min(bxx.shape) > 150:
                bxx, byy = sfq_clean(bxx, byy, 19, gauss=gauss, median=median, show=show, silent=True)
            if min(bxx.shape) > 100:
                bxx, byy = sfq_clean(bxx, byy, 9, gauss=gauss, median=median, show=show, silent=True)
        bxx, byy = sfq_clean(bxx, byy, 5, gauss=gauss, median=median, show=show, silent=True)
        bxx, byy = sfq_clean(bxx, byy, 3, median=True, show=show, silent=True)
        if not silent:
            dt = int(round(time.time() - t0))
            print(f"SFQ cleaning complete in {dt} seconds")
        return bxx, byy

    s_val = float(s)
    n = int(np.ceil(s_val * 3.0) * 2 + 1)
    ker = _gaussf(n, s_val)
    gaussk = float(ker[n // 2] ** 2)
    threshold = max(bxx.size * 0.0001, 5.0)

    for _ in range(301):
        if gauss:
            mbx = _pwf_smooth(bxx, s_val) - bxx * gaussk
            mby = _pwf_smooth(byy, s_val) - byy * gaussk
        else:
            if not median:
                sval_i = int(max(round(s_val), 1))
                mbx = _box_smooth_edge(bxx, sval_i) - bxx / float(sval_i ** 2)
                mby = _box_smooth_edge(byy, sval_i) - byy / float(sval_i ** 2)
            else:
                sval_i = int(max(round(s_val), 1))
                mbx = _median_filter_edge(bxx, sval_i)
                mby = _median_filter_edge(byy, sval_i)

        flip = (mbx * bxx + byy * mby) < 0
        if int(np.count_nonzero(flip)) < threshold:
            break
        bxx[flip] = -bxx[flip]
        byy[flip] = -byy[flip]
        if show:
            pass

    return bxx, byy
