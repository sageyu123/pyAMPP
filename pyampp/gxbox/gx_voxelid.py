from __future__ import annotations

import numpy as np


def gx_voxelid(
    test_id: int | np.ndarray | None = None,
    *,
    chromo: bool = False,
    tr: bool = False,
    corona: bool = False,
    euvtr: bool = False,
    in_: bool = False,
    nw: bool = False,
    enw: bool = False,
    fa: bool = False,
    pl: bool = False,
    pen: bool = False,
    umb: bool = False,
    tube: bool = False,
    layer: bool = False,
    mask: bool = False,
) -> np.uint32 | np.ndarray:
    """IDL-equivalent voxel bitmask builder/tester.

    Port of ``gx_voxelid.pro`` from GX Simulator.

    If ``test_id`` is provided, returns ``id & uint32(test_id)``.
    Otherwise returns the constructed uint32 ``id``.
    """
    vid = np.uint32(0)

    if chromo:
        vid += np.uint32(1)
    if tr:
        vid += np.uint32(2)
    if corona:
        vid += np.uint32(4)
    if euvtr:
        vid += np.uint32(8)

    if tube:
        vid += np.uint32(255 << 8)
    if layer:
        vid += np.uint32(255 << 16)
    if mask:
        vid += np.uint32(255 << 24)

    if in_:
        vid += np.uint32(1 << 24)
    if nw:
        vid += np.uint32(2 << 24)
    if enw:
        vid += np.uint32(3 << 24)
    if fa:
        vid += np.uint32(4 << 24)
    if pl:
        vid += np.uint32(5 << 24)
    if pen:
        vid += np.uint32(6 << 24)
    if umb:
        vid += np.uint32(7 << 24)

    if test_id is None:
        return vid
    return vid & np.asarray(test_id, dtype=np.uint32)

