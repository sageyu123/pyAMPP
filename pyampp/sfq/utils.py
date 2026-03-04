from __future__ import annotations

import numpy as np


def idl_where(mask: np.ndarray) -> np.ndarray:
    """IDL-like WHERE: return flat indices; return [-1] if no match."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.array([-1], dtype=np.int64)
    return idx.astype(np.int64)


def u_grid(pos: np.ndarray, n: tuple[int, int]) -> dict[str, np.ndarray]:
    """Build an IDL-like pixel grid centered at ``pos``."""
    if len(n) != 2:
        raise ValueError("n must be a 2-element shape tuple.")
    if np.asarray(pos).size != 2:
        raise ValueError("pos must have 2 elements.")
    x0, y0 = float(pos[0]), float(pos[1])
    ii, jj = np.indices((int(n[0]), int(n[1])), dtype=float)
    return {"x": ii - x0, "y": jj - y0}


def u_grid_box(start: np.ndarray, extent: np.ndarray, n: np.ndarray) -> dict[str, np.ndarray]:
    """Build a regular 2D grid from start/extents/sample counts."""
    start = np.asarray(start, dtype=float).reshape(2)
    extent = np.asarray(extent, dtype=float).reshape(2)
    n = np.asarray(n, dtype=int).reshape(2)
    nx, ny = int(n[0]), int(n[1])
    if nx < 2 or ny < 2:
        raise ValueError("grid dimensions must be >= 2 in both axes.")
    x = start[0] + np.arange(nx, dtype=float) * extent[0] / float(nx - 1)
    y = start[1] + np.arange(ny, dtype=float) * extent[1] / float(ny - 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return {"x": xx, "y": yy}


def u_str_add(s: dict, tags, *values):
    """Minimal IDL-like struct tag add/update helper."""
    out = dict(s)
    if isinstance(tags, (list, tuple, np.ndarray)):
        if len(values) != len(tags):
            raise ValueError("When tags is a sequence, values length must match tags length.")
        for k, v in zip(tags, values):
            out[str(k)] = v
        return out

    key = str(tags)
    if len(values) == 0:
        if "t0" in out and isinstance(out["t0"], np.ndarray):
            out[key] = np.zeros_like(out["t0"])
        else:
            out[key] = None
    else:
        out[key] = values[0]
    return out


def norm_vec(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.asarray(v, dtype=float) ** 2)))
