import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicCarrington, HeliographicStonyhurst
from sunpy.map import all_coordinates_from_map, Map
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
    '''
    Read the B3D data from the .sav file and save it as a .gxbox file
    :param savfile: str,
        The path to the .sav file.


    '''
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
    gxboxdata['b3d']['nlfff'] = {}
    gxboxdata['b3d']['nlfff']['bx'] = bx.swapaxes(0,2)
    gxboxdata['b3d']['nlfff']['by'] = by.swapaxes(0,2)
    gxboxdata['b3d']['nlfff']['bz'] = bz.swapaxes(0,2)
    boxfilenew = savfile.replace('.sav', '.gxbox')
    with open(boxfilenew, 'wb') as f:
        pickle.dump(gxboxdata, f)
    print(f'{savfile} is saved as {boxfilenew}')
    return boxfilenew

def read_b3d_h5(filename):
    """
    Read B3D data from an HDF5 file and populate a dictionary.

    The resulting dictionary will contain keys corresponding to different
    magnetic field models (e.g., 'pot' for potential fields and 'nlfff' for
    nonlinear force-free fields), and each model will have sub-keys for
    the magnetic field components (e.g., 'bx', 'by', 'bz').

    :param filename: str,
        The path to the HDF5 file.
    :return: dict,
        A dictionary containing the B3D data.

    :example:

    .. code-block:: python

        b3dbox = read_b3d_h5('path_to_file.h5')

        # Get the potential field components
        bx_pot = b3dbox['pot']['bx']
        by_pot = b3dbox['pot']['by']
        bz_pot = b3dbox['pot']['bz']

        # Get the NLFFF field components
        bx_nlf = b3dbox['nlfff']['bx']
        by_nlf = b3dbox['nlfff']['by']
        bz_nlf = b3dbox['nlfff']['bz']
    """
    box_b3d = {}
    with h5py.File(filename, 'r') as hdf_file:
        for model_type in hdf_file.keys():
            group = hdf_file[model_type]
            box_b3d[model_type] = {}
            for component in group.keys():
                box_b3d[model_type][component] = group[component][:]
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
            group = hdf_file.create_group(model_type)
            for component, data in components.items():
                group.create_dataset(component, data=data)