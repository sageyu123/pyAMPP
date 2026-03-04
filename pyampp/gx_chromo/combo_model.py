import numpy as np
from scipy.interpolate import interp1d
import pdb
from .decompose import decompose
from .populate_chromo import populate_chromo

def combo_model_idl(bndbox, box):
    base_bz = bndbox["base"][0]["bz"][0].T
    base_ic = bndbox["base"][0]["ic"][0].T
    dr = bndbox["dr"][0]

    #box2 = {}

    #for k in ("bx", "by", "bz"):
    #    box2[k] = box[k].transpose((1, 2, 0))

    return combo_model(box, dr, base_bz, base_ic)

def combo_model(box, dr, base_bz, base_ic, chromo_mask=None):
    if chromo_mask is None:
        chromo_mask = decompose(base_bz, base_ic)
    chromo = populate_chromo(chromo_mask)

    csize = chromo['nh'].shape

    bx, by, bz = (box[k] for k in ("bx", "by", "bz"))

    msize = bx.shape
    box_bcube = np.zeros((*msize, 3), dtype=np.float32)
    box_bcube[:, :, :, 0] = bx[:,:,::-1,:]
    box_bcube[:, :, :, 1] = by[:,:,::-1,:]
    box_bcube[:, :, :, 2] = bz[:,:,::-1,:]

    dz = np.ones(msize, dtype=np.float64) * dr[2]
    z = np.zeros(msize, dtype=np.float64)
    z[:, :, 1:msize[2]] = np.cumsum(dz, axis=2)[:, :, 0:msize[2]-1]

    dh_flat = chromo['dh'].flatten(order="F")
    chromo_idx = np.where(dh_flat != 1)[0]
    chromo['dh'][chromo['dh'] == 1] = 0

    tr_h = np.sum(chromo['dh'], axis=2) / 696000.0

    max_tr_h = np.max(tr_h)

    corona_base_idx = np.min(np.where(z[0, 0, :] >= max_tr_h)[0])
    corona_base_height = z[0, 0, corona_base_idx]
    dh = chromo['dh'] / 696000.0

    tr_idx = np.zeros((csize[0], csize[1]), dtype=np.int64)

    for i in range(csize[0]):
        for j in range(csize[1]):
            tr_idx[i, j] = np.max(np.where(chromo["dh"][i, j, :] != 0)[0])+1
            if tr_idx[i, j] < csize[2]:
                dh[i,j,tr_idx[i,j]:] = (corona_base_height - tr_h[i,j]) / dh[i,j,tr_idx[i,j]:].size
            else:
                dz[i,j,corona_base_idx] += corona_base_height - tr_h[i,j]

    dz = dz[:, :, corona_base_idx:]

    size_dz = dz.shape
    big_size = csize[2] + size_dz[2]
    big_dh = np.zeros((csize[0], csize[1], big_size))
    big_dh[:, :, 0:csize[2]] = dh[:, :, 0:csize[2]]
    big_dh[:, :, csize[2]:] = dz
    big_h = np.zeros((csize[0], csize[1], big_size), dtype=np.float64)
    big_h[:, :, 1:big_size] = np.cumsum(big_dh, axis=2)[:, :, 0:big_size-1]

    max_chromo_idx = np.max(tr_idx)

    h = big_h[:, :, 0:max_chromo_idx]

    nx, ny, nz = z.shape
    nh = h.shape[2]

    z_flat = z.reshape(-1, nz)  # (Ncols, nz)
    h_flat = h.reshape(-1, nh)  # (Ncols, nh)
    bcube_flat = np.zeros((nx * ny, nh, 3), dtype=np.float32)

    for k in range(3):  # bx, by, bz
        fp_flat = box_bcube[..., k].reshape(-1, nz)  # (Ncols, nz)
        out = np.empty((nx * ny, nh), dtype=np.float32)
        for r in range(nx * ny):
            # np.interp clamps outside-range values to edge fp by default
            out[r] = np.interp(h_flat[r], z_flat[r], fp_flat[r])
        bcube_flat[:, :, k] = out

    bcube = bcube_flat.reshape(nx, ny, nh, 3)

    t  = chromo['temp'].T.flat[chromo_idx]
    n   = chromo['nne'].T.flat[chromo_idx]
    nh   = chromo['nh'].T.flat[chromo_idx]
    nhi = chromo['nhi'].T.flat[chromo_idx]
    n_p  = chromo['np'].T.flat[chromo_idx]

    return {
        'chromo_idx': chromo_idx,
        'bcube': box_bcube,
        'chromo_bcube': bcube,
        'n_htot': nh,
        'n_hi': nhi,
        'n_p': n_p,
        'dz': big_dh,
        'chromo_n': n,
        'chromo_t': t,
        'chromo_layers': max_chromo_idx,
        'tr': tr_idx,
        'tr_h': tr_h,
        'corona_base': corona_base_idx,
        'dr': dr,
        'chromo_mask': chromo_mask
    }
