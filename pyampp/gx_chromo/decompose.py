import numpy as np

def decompose(mag, cont):
    mag_qs = 10  # 10 Gauss for QS
    thr_plage = 3  # MF in plage is thr_plage times stronger than QS

    sub = np.abs(mag) < mag_qs
    count = np.count_nonzero(sub)
    cutoff_qs = np.sum(cont[sub]) / count
    print("cutoff_qs: ", cutoff_qs, count)

    pdf = np.histogram(cont.flat, bins=cont.size)
    pdf_sub = np.histogram(cont[sub].flat, bins=cont.size)

    cutoff_b = pdf_sub[1][-1]
    cutoff_f = pdf_sub[1][-1]
    pdf_sub = np.histogram(cont[sub].flat, bins=cont.size)

    # exclude sunspots
    sub = (cont > (cutoff_qs * 0.9))
    cutoff_b = np.quantile(cont[sub], 0.75)
    cutoff_f = np.quantile(cont[sub], 0.97)

    print("cutoff_b: ", cutoff_b)
    print("cutoff_f: ", cutoff_f)
    
    # creating decomposition mask
    model_mask = np.zeros(cont.shape, dtype=np.int32)
    abs_mag = np.abs(mag)

    # umbra
    sub = cont <= (0.65 * cutoff_qs)
    n_umbra = np.count_nonzero(sub)
    if n_umbra != 0:
        print("umbra: nelem= ", n_umbra, " abs(B) range: ", np.min(abs_mag[sub]), np.max(abs_mag[sub]))
        model_mask[sub] = 7

    # penumbra
    sub = (cont > (0.65 * cutoff_qs)) & (cont <= (0.9 * cutoff_qs))
    n_penumbra = np.count_nonzero(sub)
    if n_penumbra != 0:
        print("penumbra: nelem= ", n_penumbra, " abs(B) range: ", np.min(abs_mag[sub]), np.max(abs_mag[sub]))
        model_mask[sub] = 6

    # enhanced NW
    sub = (cont > cutoff_f) & (cont <= (1.19 * cutoff_qs))
    n_enw = np.count_nonzero(sub)
    if n_enw != 0:
        print("eNW: nelem= ", n_enw, " abs(B) range: ", np.min(abs_mag[sub]), np.max(abs_mag[sub]))
        model_mask[sub] = 3

    # NW lane
    sub = (cont > cutoff_b) & (cont <= cutoff_f)
    n_nw = np.count_nonzero(sub)
    if n_nw != 0:
        print("NW: nelem= ", n_nw, " abs(B) range: ", np.min(abs_mag[sub]), np.max(abs_mag[sub]))
        model_mask[sub] = 2

    # IN
    sub = (cont > (0.9 * cutoff_qs)) & (cont <= cutoff_b)
    n_in = np.count_nonzero(sub)
    if n_in != 0:
        print("IN: nelem= ", n_in, " abs(B) range: ", np.min(abs_mag[sub]), np.max(abs_mag[sub]))
        model_mask[sub] = 1

    # plage
    sub = (cont > (0.95 * cutoff_qs)) & (cont <= cutoff_f) & (abs_mag > (thr_plage * mag_qs))
    n_plage = np.count_nonzero(sub)
    if n_plage != 0:
        print("plage: nelem= ", n_plage, " abs(B) range: ", np.min(abs_mag[sub]), np.max(abs_mag[sub]))
        model_mask[sub] = 4

    # facula
    sub = (cont > (1.01 * cutoff_qs)) & (abs_mag > (thr_plage * mag_qs))
    n_facula = np.count_nonzero(sub)
    if n_facula != 0:
        print("facula: nelem= ", n_facula, " abs(B) range: ", np.min(abs_mag[sub]), np.max(abs_mag[sub]))
        model_mask[sub] = 5

    n_tot = n_in + n_nw + n_enw + n_plage + n_facula + n_penumbra + n_umbra
    print("Total elements: ", n_tot)
    print("Number of elements in cont: ", np.shape(cont)[0] * np.shape(cont)[1])

    return model_mask