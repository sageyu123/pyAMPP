import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicCarrington, HeliographicStonyhurst
from sunpy.map import all_coordinates_from_map, Map

import numpy as np
from astropy.io import fits

def hmi_disambig(azimuth_map, disambig_map, method=2):
    """
    Combine HMI disambiguation result with azimuth.

    :param azimuth_map: sunpy.map.Map, The azimuth map.
    :type azimuth_map: sunpy.map.Map
    :param disambig_map: sunpy.map.Map, The disambiguation map.
    :type disambig_map: sunpy.map.Map
    :param method: int, Method index (0: potential acute, 1: random, 2: radial acute). Default is 2.
    :type method: int, optional

    :return: map_azimuth: sunpy.map.Map, The azimuth map with disambiguation applied.
    :rtype: sunpy.map.Map
    """
    # Load data from FITS files
    azimuth = azimuth_map.data
    disambig = disambig_map.data

    # Check dimensions of the arrays
    if azimuth.shape != disambig.shape:
        raise ValueError("Dimension of two images do not agree")

    # Fix disambig to ensure it is integer
    disambig = disambig.astype(int)

    # Validate method index
    if method < 0 or method > 2:
        method = 2
        print("Invalid disambiguation method, set to default method = 2")

    # Apply disambiguation method
    disambig_corr = disambig // (2 ** method)  # Move the corresponding bit to the lowest
    index = disambig_corr % 2 != 0  # True where bits indicate flip

    # Update azimuth where flipping is needed
    azimuth[index] += 180
    azimuth = azimuth % 360  # Ensure azimuth stays within [0, 360]

    map_azimuth = Map(azimuth, azimuth_map.meta)
    return map_azimuth


def hmi_b2ptr(map_field, map_inclination, map_azimuth):
    sz = map_field.data.shape
    ny, nx = sz

    field = map_field.data
    gamma = np.deg2rad(map_inclination.data)
    psi = np.deg2rad(map_azimuth.data)

    b_xi = -field * np.sin(gamma) * np.sin(psi)
    b_eta = field * np.sin(gamma) * np.cos(psi)
    b_zeta = field * np.cos(gamma)

    foo = all_coordinates_from_map(map_field).transform_to(HeliographicStonyhurst)
    phi = foo.lon
    lambda_ = foo.lat

    b = np.deg2rad(map_field.fits_header["crlt_obs"])
    p = np.deg2rad(-map_field.fits_header["crota2"])

    phi, lambda_ = np.deg2rad(phi), np.deg2rad(lambda_)

    sinb, cosb = np.sin(b), np.cos(b)
    sinp, cosp = np.sin(p), np.cos(p)
    sinphi, cosphi = np.sin(phi), np.cos(phi)  # nx*ny
    sinlam, coslam = np.sin(lambda_), np.cos(lambda_)  # nx*ny

    k11 = coslam * (sinb * sinp * cosphi + cosp * sinphi) - sinlam * cosb * sinp
    k12 = - coslam * (sinb * cosp * cosphi - sinp * sinphi) + sinlam * cosb * cosp
    k13 = coslam * cosb * cosphi + sinlam * sinb
    k21 = sinlam * (sinb * sinp * cosphi + cosp * sinphi) + coslam * cosb * sinp
    k22 = - sinlam * (sinb * cosp * cosphi - sinp * sinphi) - coslam * cosb * cosp
    k23 = sinlam * cosb * cosphi - coslam * sinb
    k31 = - sinb * sinp * sinphi + cosp * cosphi
    k32 = sinb * cosp * sinphi + sinp * cosphi
    k33 = - cosb * sinphi

    bptr = np.zeros((3, ny, nx))

    bptr[0, :, :] = k31 * b_xi + k32 * b_eta + k33 * b_zeta
    bptr[1, :, :] = k21 * b_xi + k22 * b_eta + k23 * b_zeta
    bptr[2, :, :] = k11 * b_xi + k12 * b_eta + k13 * b_zeta

    header = map_field.fits_header
    map_bp = Map(bptr[0, :, :], header)
    map_bt = Map(bptr[1, :, :], header)
    map_br = Map(bptr[2, :, :], header)

    return map_bp, map_bt, map_br
