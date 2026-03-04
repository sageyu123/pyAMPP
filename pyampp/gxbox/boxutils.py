import numpy as np
import astropy.units as u
from astropy.io import fits
from sunpy.map import Map

def _bilinear_sample(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, arr.shape[0] - 1)
    x1 = np.clip(x1, 0, arr.shape[0] - 1)
    y0 = np.clip(y0, 0, arr.shape[1] - 1)
    y1 = np.clip(y1, 0, arr.shape[1] - 1)

    wx = x - x0
    wy = y - y0

    f00 = arr[x0, y0]
    f10 = arr[x1, y0]
    f01 = arr[x0, y1]
    f11 = arr[x1, y1]

    return (f00 * (1 - wx) * (1 - wy) +
            f10 * wx * (1 - wy) +
            f01 * (1 - wx) * wy +
            f11 * wx * wy)


def _vxv(a: np.ndarray, b: np.ndarray, norm: bool = False) -> np.ndarray:
    c = np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ], dtype=float)
    if norm:
        n = np.sqrt((c * c).sum())
        if n > 0:
            c /= n
    return c


def compute_vertical_current(b0: np.ndarray,
                             b1: np.ndarray,
                             b2: np.ndarray,
                             wcs_header: str,
                             rsun_arcsec: float,
                             crpix1: float | None = None,
                             crpix2: float | None = None,
                             cdelt1_arcsec: float | None = None,
                             cdelt2_arcsec: float | None = None) -> np.ndarray:
    if crpix1 is None or crpix2 is None or cdelt1_arcsec is None or cdelt2_arcsec is None:
        header = fits.Header.fromstring(wcs_header, sep="\n")
        cdelt1 = header.get("CDELT1")
        cdelt2 = header.get("CDELT2")
        cunit1 = header.get("CUNIT1", "deg")
        cunit2 = header.get("CUNIT2", "deg")
        crpix1 = header.get("CRPIX1")
        crpix2 = header.get("CRPIX2")

        if cdelt1 is None or cdelt2 is None or crpix1 is None or crpix2 is None:
            raise ValueError("WCS header missing CDELT/CRPIX values.")

        cdelt1_arcsec = (cdelt1 * u.Unit(cunit1)).to_value(u.arcsec)
        cdelt2_arcsec = (cdelt2 * u.Unit(cunit2)).to_value(u.arcsec)

    jr = np.zeros_like(b0, dtype=float)
    nx, ny = b0.shape

    delta = 0.5 * cdelt1_arcsec / rsun_arcsec

    for i1 in range(1, nx - 1):
        for i2 in range(1, ny - 1):
            x1 = (i1 - crpix1) * cdelt1_arcsec / rsun_arcsec
            x2 = (i2 - crpix2) * cdelt2_arcsec / rsun_arcsec
            if np.sqrt(x1 * x1 + x2 * x2) >= 0.95:
                continue
            x0 = np.sqrt(1.0 - x1 * x1 - x2 * x2)

            e0 = np.array([x0, x1, x2], dtype=float)
            e1 = _vxv(np.array([0.0, 0.0, 1.0], dtype=float), e0, norm=True)
            e2 = _vxv(e0, e1, norm=False)

            x1s = np.array([-e1[1], e1[1], -e2[1], e2[1]]) * delta + x1
            x2s = np.array([-e1[2], e1[2], -e2[2], e2[2]]) * delta + x2

            xx1 = x1s * rsun_arcsec / cdelt1_arcsec + crpix1
            xx2 = x2s * rsun_arcsec / cdelt2_arcsec + crpix2

            b0_int = _bilinear_sample(b0, xx1, xx2) * 1e-4
            b1_int = _bilinear_sample(b1, xx1, xx2) * 1e-4
            b2_int = _bilinear_sample(b2, xx1, xx2) * 1e-4

            term_e2 = (b0_int[1] * e2[0] + b1_int[1] * e2[1] + b2_int[1] * e2[2]) - \
                      (b0_int[0] * e2[0] + b1_int[0] * e2[1] + b2_int[0] * e2[2])
            term_e1 = (b0_int[3] * e1[0] + b1_int[3] * e1[1] + b2_int[3] * e1[2]) - \
                      (b0_int[2] * e1[0] + b1_int[2] * e1[1] + b2_int[2] * e1[2])

            jr[i1, i2] = (term_e2 - term_e1)
            jr[i1, i2] /= 2 * (delta * 696e6) * (4 * np.pi * 1e-7)

    return jr


def _sanitize_unit_like_header_values(header: fits.Header) -> fits.Header:
    """
    Normalize non-standard escaped unit strings found in some IDL-written FITS.
    Example: 'DN\\/s' -> 'DN/s'.
    """
    for key in ("BUNIT", "CUNIT1", "CUNIT2"):
        if key in header:
            value = header.get(key)
            if isinstance(value, str):
                fixed = value.replace("\\/", "/").replace("\\", "")
                if fixed != value:
                    header[key] = fixed
    return header


def _first_image_hdu(hdulist):
    for hdu in hdulist:
        if getattr(hdu, "data", None) is not None:
            return hdu
    raise ValueError("No image HDU with data found in FITS file.")


def load_sunpy_map_compat(path_or_data, header=None):
    """
    Load a SunPy map while tolerating IDL-style escaped FITS unit strings.

    If a direct `Map(path)` fails due to unit parsing, this falls back to opening
    the FITS file, sanitizing unit-like header fields, and building the map from
    `(data, header)`.
    """
    if header is not None:
        if isinstance(header, fits.Header):
            safe_header = _sanitize_unit_like_header_values(header.copy())
        else:
            safe_header = _sanitize_unit_like_header_values(fits.Header(header))
        return Map(path_or_data, safe_header)

    try:
        return Map(path_or_data)
    except Exception as exc:
        msg = str(exc)
        if ("did not parse as unit" not in msg) and ("not a valid unit" not in msg):
            raise

        with fits.open(path_or_data) as hdul:
            image_hdu = _first_image_hdu(hdul)
            data = image_hdu.data
            safe_header = _sanitize_unit_like_header_values(image_hdu.header.copy())
        return Map(data, safe_header)


from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicCarrington, HeliographicStonyhurst
from sunpy.map import all_coordinates_from_map
from PyQt5.QtWidgets import  QMessageBox

import numpy as np
from astropy.io import fits
import h5py

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
    '''
    Converts the magnetic field components from SDO/HMI
    data from field strength, inclination, and azimuth into components of the
    magnetic field in the local heliographic coordinate system (B_phi, B_theta, B_r).

    This function transforms the magnetic field vector in the local (xi, eta, zeta)
    system to the heliographic (phi, theta, r) system. The transformation accounts for
    the observer's position and orientation relative to the Sun.

    :param map_field: `sunpy.map.Map`,
        The magnetic field strength map from HMI, given in Gauss.

    :param map_inclination: `sunpy.map.Map`,
        The magnetic field inclination angle map from HMI, in degrees, where 0 degrees is
        parallel to the radial direction.

    :param map_azimuth: `sunpy.map.Map`,
        The magnetic field azimuth angle map from HMI, in degrees, measured counterclockwise
        from the north in the plane perpendicular to the radial direction.

    :return: tuple,
        A tuple containing the three magnetic field component maps in the heliographic
        coordinate system:

        - `map_bp` (`sunpy.map.Map`): The magnetic field component in the phi direction (B_phi).
        - `map_bt` (`sunpy.map.Map`): The magnetic field component in the theta direction (B_theta).
        - `map_br` (`sunpy.map.Map`): The magnetic field component in the radial direction (B_r).

    :example:

    .. code-block:: python

        # Load the HMI field, inclination, and azimuth maps
        map_field = sunpy.map.Map('hmi_field.fits')
        map_inclination = sunpy.map.Map('hmi_inclination.fits')
        map_azimuth = sunpy.map.Map('hmi_azimuth.fits')

        # Convert to heliographic coordinates
        map_bp, map_bt, map_br = hmi_b2ptr(map_field, map_inclination, map_azimuth)

    '''
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


def validate_number(func):
    """
    Decorator to validate if the input in the widget is a number.

    :param func: function,
        The function to wrap.
    :return: function ,
        The wrapped function.
    """

    def wrapper(self, widget, *args, **kwargs):
        try:
            if hasattr(widget, "value"):
                float(widget.value())
            else:
                float(widget.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")
            return
        return func(self, widget, *args, **kwargs)

    return wrapper


def set_QLineEdit_text_pos(line_edit, text):
    """
    Sets the text of the QLineEdit and moves the cursor to the beginning.

    :param line_edit: QLineEdit,
        The QLineEdit widget.
    :param text: str,
        The text to set.
    """
    line_edit.setText(text)
    line_edit.setCursorPosition(0)


def read_gxsim_b3d_sav(savfile):
    """
    Read B3D data from a ``.sav`` file and save it as a ``.gxbox`` file.

    :param savfile: str,
        The path to the ``.sav`` file.
    """
    from scipy.io import readsav
    import pickle
    bdata = readsav(savfile)
    bx = bdata['box']['bx'][0]
    by = bdata['box']['by'][0]
    bz = bdata['box']['bz'][0]
    # boxfile = 'hmi.sharp_cea_720s.8088.20220330_172400_TAI.Br.gxbox'
    # with open(boxfile, 'rb') as f:
    #     gxboxdata = pickle.load(f)
    gxboxdata = {}
    gxboxdata['b3d'] = {}
    gxboxdata['b3d']['corona'] = {}
    gxboxdata['b3d']['corona']['bx'] = bx.swapaxes(0,2)
    gxboxdata['b3d']['corona']['by'] = by.swapaxes(0,2)
    gxboxdata['b3d']['corona']['bz'] = bz.swapaxes(0,2)
    gxboxdata['b3d']['corona']["attrs"] = {"model_type": "nlfff"}
    boxfilenew = savfile.replace('.sav', '.gxbox')
    with open(boxfilenew, 'wb') as f:
        pickle.dump(gxboxdata, f)
    print(f'{savfile} is saved as {boxfilenew}')
    return boxfilenew

def read_b3d_h5(filename):
    """
    Read B3D data from an HDF5 file and populate a dictionary.

    The resulting dictionary will contain keys corresponding to different
    magnetic field models (e.g., 'corona' for coronal fields and 'chromo' for
    chromospheric fields), and each model will have sub-keys for the magnetic
    field components (e.g., 'bx', 'by', 'bz').

    :param filename: str,
        The path to the HDF5 file.
    :return: dict,
        A dictionary containing the B3D data.

    :example:

    .. code-block:: python

        b3dbox = read_b3d_h5('path_to_file.h5')

        # Get the coronal field components
        bx_cor = b3dbox['corona']['bx']
        by_cor = b3dbox['corona']['by']
        bz_cor = b3dbox['corona']['bz']
    """
    box_b3d = {}
    with h5py.File(filename, 'r') as hdf_file:
        for model_type in hdf_file.keys():
            group = hdf_file[model_type]
            target_type = model_type
            model_attr = None
            if model_type in ("nlfff", "pot", "potential", "bounds"):
                target_type = "corona"
                if model_type == "potential":
                    model_attr = "pot"
                elif model_type in ("bounds",):
                    model_attr = "bnd"
                else:
                    model_attr = model_type
            if target_type not in box_b3d:
                box_b3d[target_type] = {}
            for component in group.keys():
                ds = group[component]
                if isinstance(ds, h5py.Group):
                    sub = {}
                    for sub_key in ds.keys():
                        sub_ds = ds[sub_key]
                        if sub_ds.shape == ():
                            sub[sub_key] = sub_ds[()]
                        else:
                            sub[sub_key] = sub_ds[:]
                    box_b3d[target_type][component] = sub
                else:
                    if ds.shape == ():
                        box_b3d[target_type][component] = ds[()]
                    else:
                        box_b3d[target_type][component] = ds[:]
            if len(group.attrs.keys()) > 0 or model_attr is not None:
                attrs = dict(group.attrs)
                if model_attr is not None and "model_type" not in attrs:
                    attrs["model_type"] = model_attr
                if "attrs" in box_b3d[target_type]:
                    box_b3d[target_type]["attrs"].update(attrs)
                else:
                    box_b3d[target_type]["attrs"] = attrs
    return box_b3d

def write_b3d_h5(filename, box_b3d):
    """
    Write B3D data to an HDF5 file from a dictionary.

    :param filename: str,
        The path to the HDF5 file.
    :param box_b3d: dict,
        A dictionary containing the B3D data to be written.
    """

    with h5py.File(filename, 'w') as hdf_file:
        for model_type, components in box_b3d.items():
            if components is None:
                print(f"Warning: {model_type} components are None, skipping.")
                continue
            if model_type == "metadata":
                group = hdf_file.create_group("metadata")
                for key, value in components.items():
                    if isinstance(value, str):
                        group.create_dataset(key, data=np.bytes_(value))
                    else:
                        group.create_dataset(key, data=value)
                continue
            target_type = model_type
            attrs = components.get("attrs", {}) if isinstance(components, dict) else {}
            if model_type in ("nlfff", "pot", "potential", "bounds"):
                target_type = "corona"
                if "model_type" not in attrs:
                    attrs = dict(attrs)
                    if model_type == "potential":
                        attrs["model_type"] = "pot"
                    elif model_type == "bounds":
                        attrs["model_type"] = "bnd"
                    else:
                        attrs["model_type"] = model_type
            if target_type in hdf_file:
                # Avoid collisions if multiple legacy groups map to corona
                continue
            group = hdf_file.create_group(target_type)
            for component, data in components.items():
                if component == "attrs":
                    continue
                if isinstance(data, dict):
                    sub = group.create_group(component)
                    for sub_key, sub_val in data.items():
                        if target_type in ("chromo", "lines") and sub_key == "voxel_status":
                            sub.create_dataset(sub_key, data=np.asarray(sub_val, dtype=np.uint8))
                        else:
                            sub.create_dataset(sub_key, data=sub_val)
                else:
                    if target_type in ("chromo", "lines") and component == "voxel_status":
                        group.create_dataset(component, data=np.asarray(data, dtype=np.uint8))
                    else:
                        group.create_dataset(component, data=data)
            if attrs:
                group.attrs.update(attrs)


# Backward-compatible SFQ exports.
from pyampp.sfq import get_str_mag as get_str_mag
from pyampp.sfq import sfq_frame as sfq_frame
from pyampp.sfq import sfq_step1 as sfq_step1
from pyampp.sfq import sfq_clean as sfq_clean
from pyampp.sfq import pex_bl_ as pex_bl_
from pyampp.sfq import pex_bl as pex_bl
from pyampp.sfq import pot_vmag as pot_vmag
