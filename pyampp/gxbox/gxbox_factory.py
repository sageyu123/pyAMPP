import itertools
import locale
import pickle
from pathlib import Path

import astropy.units as u
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QComboBox, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, \
    QMainWindow, \
    QPushButton, QVBoxLayout, QWidget
from PyQt5 import uic

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from matplotlib import colormaps as mplcmaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pyAMaFiL.mag_field_proc import MagFieldProcessor
from pyAMaFiL.mag_field_lin_fff import MagFieldLinFFF
from sunpy.coordinates import Heliocentric, HeliographicStonyhurst, HeliographicCarrington, Helioprojective, get_earth
from sunpy.sun import constants as sun_consts
from sunpy.map import Map, coordinate_is_on_solar_disk, make_fitswcs_header

import pyampp
from pyampp.data import downloader
from pyampp.gxbox.box import Box
from pyampp.gx_chromo.decompose import decompose
from pyampp.gxbox.boxutils import hmi_b2ptr, hmi_disambig, read_b3d_h5, write_b3d_h5, compute_vertical_current
from pyampp.gxbox.magfield_viewer import MagFieldViewer
from pyampp.util.config import *

import typer
import shlex
from typing import Tuple, Optional

app = typer.Typer(help="Run GxBox with specified parameters.")

os.environ['OMP_NUM_THREADS'] = '16'  # number of parallel threads
locale.setlocale(locale.LC_ALL, "C");


def _build_index_header(bottom_wcs_header, source_map: Map) -> str:
    """
    Build an IDL-GX compatible INDEX-like FITS header string.
    """
    header = fits.Header(bottom_wcs_header).copy()

    ctype1 = str(header.get("CTYPE1", ""))
    ctype2 = str(header.get("CTYPE2", ""))
    if ctype1.startswith("HGLN-"):
        header["CTYPE1"] = "CRLN-" + ctype1.split("-", 1)[1]
    if ctype2.startswith("HGLT-"):
        header["CTYPE2"] = "CRLT-" + ctype2.split("-", 1)[1]

    header["SIMPLE"] = int(bool(header.get("SIMPLE", 1)))
    header["BITPIX"] = int(header.get("BITPIX", 8))
    header["NAXIS"] = int(header.get("NAXIS", 2))

    if source_map is not None:
        date_obs = source_map.date.isot if source_map.date is not None else None
        if date_obs:
            header["DATE-OBS"] = date_obs
            header["DATE_OBS"] = date_obs
            header["DATE"] = date_obs
        elif "DATE-OBS" in header and "DATE_OBS" not in header:
            header["DATE_OBS"] = header["DATE-OBS"]

        if hasattr(source_map, "dsun") and source_map.dsun is not None:
            header["DSUN_OBS"] = float(source_map.dsun.to_value(u.m))

        obs = source_map.observer_coordinate
        if obs is not None:
            obs_hgs = obs.transform_to(HeliographicStonyhurst(obstime=source_map.date))
            header["HGLN_OBS"] = float(obs_hgs.lon.to_value(u.deg))
            header["HGLT_OBS"] = float(obs_hgs.lat.to_value(u.deg))
            header["SOLAR_B0"] = float(obs_hgs.lat.to_value(u.deg))
            try:
                obs_hgc = obs.transform_to(HeliographicCarrington(observer="earth", obstime=source_map.date))
                header["CRLN_OBS"] = float(obs_hgc.lon.to_value(u.deg))
                header["CRLT_OBS"] = float(obs_hgc.lat.to_value(u.deg))
            except Exception:
                # Carrington observer transforms can fail for some observer metadata;
                # keep header generation robust and proceed with available keys.
                pass

        if getattr(source_map, "rsun_meters", None) is not None:
            header["RSUN_REF"] = float(source_map.rsun_meters.to_value(u.m))

        tel = source_map.meta.get("telescop")
        if tel:
            header["TELESCOP"] = str(tel)
        instr = source_map.meta.get("instrume")
        if instr:
            header["INSTRUME"] = str(instr)
        if "WCSNAME" not in header:
            header["WCSNAME"] = "Carrington-Heliographic"

    return header.tostring(sep="\n", endcard=True)


## todo add chrom mask to the tool
class GxBox(QMainWindow):
    def __init__(self, time, observer, box_orig, box_dims=u.Quantity([100, 100, 100]) * u.pix,
                 box_res=1.4 * u.Mm, pad_frac=0.10, data_dir=DOWNLOAD_DIR, gxmodel_dir=GXMODEL_DIR,
                 external_box=None, execute_cmd: str | None = None,
                 save_empty_box: bool = False,
                 save_bounds: bool = False,
                 save_potential: bool = False,
                 save_nas: bool = False,
                 save_gen: bool = False,
                 save_chr: bool = False,
                 stop_after: str | None = None,
                 auto_visualize_last: bool = True,
                 euv: bool = True,
                 uv: bool = True,
                 hmifiles: str | None = None,
                 entry_box: str | None = None,
                 empty_box_only: bool = False,
                 potential_only: bool = False,
                 use_potential: bool = False,
                 jump2potential: bool = False,
                 jump2nlfff: bool = False,
                 jump2lines: bool = False,
                 jump2chromo: bool = False):
        """
        Main application window for visualizing and interacting with solar data in a 3D box.

        :param time: Observation time.
        :type time: `~astropy.time.Time`
        :param observer: Observer location.
        :type observer: `~astropy.coordinates.SkyCoord`
        :param box_orig: The origin of the box (center of the box bottom).
        :type box_orig: `~astropy.coordinates.SkyCoord`
        :param box_dims: Dimensions of the box in pixels (x, y, z) in the specified units. x and y are in the hpc frame, z is the height above the solar surface.
        :type box_dims: `~astropy.units.Quantity`
        :param box_res: Spatial resolution of the box, defaults to 1.4 Mm.
        :type box_res: `~astropy.units.Quantity`
        :param pad_frac: Fractional padding applied to each side of the box, expressed as a decimal, defaults to 0.25.
        :type pad_frac: float
        :param data_dir: Directory for storing data.
        :type data_dir: str
        :param gxmodel_dir: Directory for storing model outputs.
        :type gxmodel_dir: str
        :param external_box: Path to external box file (optional).
        :type external_box: str

        Methods
        -------
        loadmap(mapname, fov_coords=None)
            Loads a map from the available data.
        init_ui()
            Initializes the user interface.
        update_bottom_map(map_name)
            Updates the bottom map displayed in the UI.
        update_context_map(map_name)
            Updates the context map displayed in the UI.
        update_plot()
            Updates the plot with the current data and settings.
        create_lines_of_sight()
            Creates lines of sight for the visualization.
        visualize()
            Visualizes the data in the UI.

        Example
        -------
        >>> from astropy.time import Time
        >>> from astropy.coordinates import SkyCoord
        >>> import astropy.units as u
        >>> from pyampp.gxbox import GxBox
        >>> time = Time('2024-05-09T17:12:00')
        >>> observer = SkyCoord(0 * u.deg, 0 * u.deg, obstime=time, frame='heliographic_carrington')
        >>> box_orig = SkyCoord(450 * u.arcsec, -256 * u.arcsec, obstime=time, observer="earth", frame='helioprojective')
        >>> box_dims = u.Quantity([100, 100, 100], u.pix)
        >>> box_res = 1.4 * u.Mm
        >>> gxbox = GxBox(time, observer, box_orig, box_dims, box_res)
        >>> gxbox.show()
        """

        super(GxBox, self).__init__()
        self.time = time
        self.observer = observer
        self.box_dims = box_dims
        self.box_res = box_res
        self.pad_frac = pad_frac
        ## this is the origin of the box, i.e., the center of the box bottom
        self.box_origin = box_orig
        self.gxmodel_dir = gxmodel_dir
        self.auto_save_stages = True
        self.execute_cmd = execute_cmd
        self.save_empty_box = save_empty_box
        self.save_bounds = save_bounds
        self.save_potential = save_potential
        self.save_nas = save_nas
        self.save_gen = save_gen
        self.save_chr = save_chr
        self.stop_after = stop_after
        self.auto_visualize_last = auto_visualize_last
        self.euv = euv
        self.uv = uv
        self.hmifiles = hmifiles
        self.entry_box = entry_box
        self.empty_box_only = empty_box_only
        self.potential_only = potential_only
        self.use_potential = use_potential
        self.jump2potential = jump2potential
        self.jump2nlfff = jump2nlfff
        self.jump2lines = jump2lines
        self.jump2chromo = jump2chromo
        self._visualizing_last = False
        self.sdofitsfiles = None
        print('observer:', self.box_origin)
        self.frame_hcc = Heliocentric(observer=self.box_origin, obstime=self.time)
        self.frame_obs = Helioprojective(observer=self.observer, obstime=self.time)
        self.frame_hgs = HeliographicStonyhurst(obstime=self.time)
        self.lines_of_sight = []
        self.edge_coords = []
        self.axes = None
        self.fig = None
        self.axes_world_coords = None
        self.axes_world_coords_init = None
        self.init_map_context_name = '171'
        self.init_map_bottom_name = 'field'
        self.external_box = external_box
        self.fieldlines_coords = []
        self.fieldlines_line_collection = []  # Initialize an empty list to store LineCollections
        self.fieldlines_show_status = True  # Initial status of the fieldlines visibility
        self.map_context_im = None
        self.map_bottom_im = None
        self._base_cache = None
        self._refmaps_cache = None

    def _stage_output_dir(self) -> Path:
        date_str = self.time.to_datetime().strftime("%Y-%m-%d")
        out_dir = Path(self.gxmodel_dir) / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _format_coord_tag(self) -> str:
        try:
            coord = self.box_origin.transform_to(HeliographicCarrington(obstime=self.time))
            suffix = "CR"
        except Exception:
            coord = self.box_origin.transform_to(HeliographicStonyhurst(obstime=self.time))
            suffix = "HG"
        lon = coord.lon.to_value(u.deg)
        lat = coord.lat.to_value(u.deg)
        lon = (lon + 180.0) % 360.0 - 180.0
        lon_dir = "W" if lon >= 0 else "E"
        lat_dir = "N" if lat >= 0 else "S"
        lon_val = f"{abs(int(round(lon))):02d}"
        lat_val = f"{abs(int(round(lat))):02d}"
        return f"{lon_dir}{lon_val}{lat_dir}{lat_val}{suffix}"

    def _stage_file_base(self) -> str:
        time_tag = self.time.to_datetime().strftime("%Y%m%d_%H%M%S")
        coord_tag = self._format_coord_tag()
        return f"hmi.M_720s.{time_tag}.{coord_tag}.CEA"

    def _stage_filename(self, stage_tag: str) -> Path:
        out_dir = self._stage_output_dir()
        base = self._stage_file_base()
        return out_dir / f"{base}.{stage_tag}.h5"

    def _make_gen_chromo(self, chromo_box: dict) -> dict:
        gen_keys = [
            "dr",
            "bcube",
            "start_idx",
            "end_idx",
            "av_field",
            "phys_length",
            "voxel_status",
            "apex_idx",
            "codes",
            "seed_idx",
        ]
        gen = {k: chromo_box[k] for k in gen_keys if k in chromo_box}
        if "attrs" in chromo_box:
            gen["attrs"] = chromo_box["attrs"]
        return gen

    def _last_stage_tag(self) -> str:
        if self.stop_after:
            stop = self.stop_after.lower()
            if stop in ("none", "empty", "empty_box"):
                return "NONE"
            if stop in ("bnd", "bounds"):
                return "BND"
            if stop == "pot":
                return "POT"
            if stop == "nas":
                return "NAS"
            if stop == "gen":
                return "NAS.GEN"
            if stop == "chr":
                return "NAS.CHR"
        return "NAS.CHR"

    def _should_save_stage(self, stage_tag: str) -> bool:
        if stage_tag == self._last_stage_tag():
            return True
        if stage_tag == "POT":
            return self.save_potential
        if stage_tag == "NAS":
            return self.save_nas
        if stage_tag == "NAS.GEN":
            return self.save_gen
        if stage_tag == "NAS.CHR":
            return self.save_chr
        if stage_tag == "NONE":
            return self.save_empty_box
        if stage_tag == "BND":
            return self.save_bounds
        return False

    def _rsun_km(self) -> u.Quantity:
        rsun = None
        if self.map_context is not None:
            rsun = self.map_context.meta.get("rsun_ref")
        if rsun is None and self.map_bottom is not None:
            rsun = self.map_bottom.meta.get("rsun_ref")
        if rsun is not None:
            return (rsun * u.m).to(u.km)
        return sun_consts.radius.to(u.km)

    def _open_last_viewer(self, stage_tag: str) -> None:
        if self._visualizing_last:
            return
        self._visualizing_last = True
        try:
            if self.box.b3d.get("corona") is None and self.box.b3d.get("chromo") is None:
                return
            if stage_tag == "POT":
                self.b3dModelSelector.setCurrentText("pot")
                b3dtype = "pot"
            elif stage_tag in ("NAS", "NAS.GEN"):
                self.b3dModelSelector.setCurrentText("nlfff")
                b3dtype = "nlfff"
            else:
                b3dtype = "chromo"
            box_norm_direction = self.box_norm_direction()
            box_view_up = self.box_view_up()
            self.visualizer = MagFieldViewer(self.box, parent=self, box_norm_direction=box_norm_direction,
                                             box_view_up=box_view_up, time=self.time, b3dtype=b3dtype)
            self.visualizer.show()
        finally:
            self._visualizing_last = False

    def _save_empty_box(self) -> None:
        obs_dr = self.box_res.to(u.km) / self._rsun_km()
        dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]
        nx, ny, nz = self.box.dims_pix
        empty = np.zeros((nx, ny, nz), dtype=float)
        stage_box = {"corona": {"bx": empty, "by": empty, "bz": empty, "dr": np.array(dr3),
                                "attrs": {"model_type": "none"}}}
        self._save_stage("NONE", stage_box)

    def _save_bounds(self) -> None:
        map_bz = self.loadmap("br")
        map_bx = -self.loadmap("bt")
        map_by = self.loadmap("bp")
        map_bz = map_bz.reproject_to(self.bottom_wcs_header, algorithm="exact")
        map_bx = map_bx.reproject_to(self.bottom_wcs_header, algorithm="exact")
        map_by = map_by.reproject_to(self.bottom_wcs_header, algorithm="exact")
        obs_dr = self.box_res.to(u.km) / self._rsun_km()
        dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]
        stage_box = {
            "bounds": {
                "bx": map_bx.data,
                "by": map_by.data,
                "bz": map_bz.data,
                "dr": np.array(dr3),
            }
        }
        self._save_stage("BND", stage_box)

    def _get_base_group(self) -> dict:
        if self._base_cache is not None:
            return self._base_cache
        map_bx = -self.loadmap("bt")
        map_by = self.loadmap("bp")
        map_bz = self.loadmap("br")
        map_ic = self.loadmap("continuum")
        map_bx = map_bx.reproject_to(self.bottom_wcs_header, algorithm="exact")
        map_by = map_by.reproject_to(self.bottom_wcs_header, algorithm="exact")
        map_bz = map_bz.reproject_to(self.bottom_wcs_header, algorithm="exact")
        map_ic = map_ic.reproject_to(self.bottom_wcs_header, algorithm="exact")
        index = _build_index_header(self.bottom_wcs_header, map_bz)
        chromo_mask = decompose(map_bz.data.T, map_ic.data.T)
        self._base_cache = {
            "bx": map_bx.data,
            "by": map_by.data,
            "bz": map_bz.data,
            "ic": map_ic.data,
            "chromo_mask": chromo_mask,
            "index": index,
        }
        return self._base_cache

    def _get_refmaps(self) -> dict:
        if self._refmaps_cache is not None:
            return self._refmaps_cache
        refmaps = {}

        def add_refmap(ref_id: str, smap: Map) -> None:
            # Use WCS header only to avoid non-ASCII FITS meta from some sources.
            header = smap.wcs.to_header()
            refmaps[ref_id] = {"data": smap.data, "wcs_header": header.tostring(sep="\n", endcard=True)}

        add_refmap("Bz_reference", self.loadmap("magnetogram"))
        add_refmap("Ic_reference", self.loadmap("continuum"))

        vert_current_error = None
        try:
            map_bx = -self.loadmap("bt")
            map_by = self.loadmap("bp")
            map_bz = self.loadmap("br")
            vc_header = map_bx.wcs.to_header().tostring(sep="\n", endcard=True)
            rsun_arcsec = self.loadmap("magnetogram").rsun_obs.to_value(u.arcsec)
            crpix1, crpix2 = map_bx.wcs.wcs.crpix
            cdelt1 = map_bx.scale.axis1.to_value(u.arcsec / u.pix)
            cdelt2 = map_bx.scale.axis2.to_value(u.arcsec / u.pix)
            jz = compute_vertical_current(map_bz.data, map_bx.data, map_by.data,
                                          vc_header, rsun_arcsec,
                                          crpix1=crpix1, crpix2=crpix2,
                                          cdelt1_arcsec=cdelt1, cdelt2_arcsec=cdelt2)
            refmaps["Vert_current"] = {"data": jz, "wcs_header": vc_header}
        except Exception as exc:
            vert_current_error = str(exc)

        for pb in AIA_EUV_PASSBANDS + AIA_UV_PASSBANDS:
            if pb in self.sdofitsfiles:
                add_refmap(f"AIA_{pb}", self.loadmap(pb))

        if vert_current_error:
            refmaps["_vert_current_error"] = {"data": np.array([vert_current_error]),
                                               "wcs_header": ""}
        self._refmaps_cache = refmaps
        return self._refmaps_cache

    def _save_stage(self, stage_tag: str, stage_box: dict):
        if not self.auto_save_stages:
            return
        if not self._should_save_stage(stage_tag):
            return
        out_path = self._stage_filename(stage_tag)
        stage_box = dict(stage_box)
        if "base" not in stage_box:
            stage_box["base"] = self._get_base_group()
        if "refmaps" not in stage_box:
            stage_box["refmaps"] = self._get_refmaps()
        stage_id = f"{self._stage_file_base()}.{stage_tag}"
        if self.execute_cmd:
            stage_box["metadata"] = {"execute": self.execute_cmd, "id": stage_id}
        else:
            stage_box["metadata"] = {"id": stage_id}
        write_b3d_h5(str(out_path), stage_box)
        if self.auto_visualize_last and stage_tag == self._last_stage_tag():
            self._open_last_viewer(stage_tag)
        self.pot_res = None

        box_dimensions = box_dims / u.pix * box_res

        ## this is a dummy map. it should be replaced by a real map from inputs.
        self.instrument_map = self.make_dummy_map(self.box_origin.transform_to(self.frame_obs))

        box_center = box_orig.transform_to(self.frame_hcc)
        box_center = SkyCoord(x=box_center.x,
                              y=box_center.y,
                              z=box_center.z + box_dimensions[2] / 2,
                              frame=box_center.frame)
        ## this is the center of the box
        self.box_center = box_center

        self.box = Box(self.frame_obs, self.box_origin, self.box_center, self.box_dims, self.box_res)
        self.box_bounds = self.box.bounds_coords
        self.bottom_wcs_header = self.box.bottom_cea_header

        self.fov_coords = self.box.bounds_coords_bl_tr(pad_frac=self.pad_frac)
        # print(f"Bottom left: {self.fov_coords[0]}; Top right: {self.fov_coords[1]}")

        if not all([coordinate_is_on_solar_disk(coord) for coord in self.fov_coords]):
            print("Warning: Some of the box corners are not on the solar disk. Please check the box dimensions.")

        if self.hmifiles:
            data_dir = self.hmifiles
        download_sdo = downloader.SDOImageDownloader(time, data_dir=data_dir, euv=self.euv, uv=self.uv)
        self.sdofitsfiles = download_sdo.download_images()
        self.sdomaps = {}

        self.sdomaps[self.init_map_context_name] = self.loadmap(self.init_map_context_name)
        self.map_context = self.sdomaps[self.init_map_context_name]
        self.bottom_wcs_header['rsun_ref'] = self.map_context.meta['rsun_ref']
        self.sdomaps[self.init_map_bottom_name] = self.loadmap(self.init_map_bottom_name)

        # print(self.bottom_wcs_header)
        self.map_bottom = self.sdomaps[self.init_map_bottom_name].reproject_to(self.bottom_wcs_header,
                                                                               algorithm="adaptive",
                                                                               roundtrip_coords=False)

        if self.save_empty_box:
            self._save_empty_box()
        if self.save_bounds:
            self._save_bounds()

        self.init_ui()

        if self.entry_box:
            if os.path.isfile(self.entry_box):
                self.load_gxbox(self.entry_box)
            if self.jump2potential:
                self.stop_after = "pot"
                self.calc_potential_field()
            elif self.jump2nlfff:
                self.stop_after = "nas"
                self.calc_nlfff()
            elif self.jump2lines:
                self.stop_after = "gen"
                self.calc_nlfff()
            elif self.jump2chromo:
                self.stop_after = "chr"
                self.calc_nlfff()

    def box_norm_direction(self):
        cartesian_coords = self.box_origin.transform_to(
            Heliocentric(observer=self.observer, obstime=self.time)).cartesian.xyz.value
        normal_vector = cartesian_coords / np.linalg.norm(cartesian_coords)
        return normal_vector

    # def box_norm_direction(self):
    #     cartesian_coords = np.diff(self.box.box_norm_direction.transform_to(Heliocentric(observer=self.observer, obstime=self.time)).cartesian.xyz).value
    #     normal_vector = np.squeeze(cartesian_coords / np.linalg.norm(cartesian_coords))
    #     normal_vector = normal_vector[1]/abs(normal_vector[1])*normal_vector
    #     return normal_vector

    def box_view_up(self):
        cartesian_coords = np.diff(self.box.box_view_up.transform_to(
            Heliocentric(observer=self.observer, obstime=self.time)).cartesian.xyz).value
        normal_vector = np.squeeze(cartesian_coords / np.linalg.norm(cartesian_coords))
        normal_vector = normal_vector[1] / abs(normal_vector[1]) * normal_vector
        return normal_vector

    def load_gxbox(self, boxfile):
        if os.path.basename(boxfile).endswith('.gxbox'):
            with open(boxfile, 'rb') as f:
                gxboxdata = pickle.load(f)
                b3d = gxboxdata.get('b3d', {})
                if "corona" in b3d:
                    self.box.b3d["corona"] = b3d["corona"]
                    self.box.corona_type = b3d.get("corona", {}).get("attrs", {}).get("model_type")
                elif "nlfff" in b3d or "pot" in b3d:
                    for model_type in ("nlfff", "pot"):
                        if model_type in b3d:
                            self.box.corona_models[model_type] = b3d[model_type]
                            self.box.b3d["corona"] = b3d[model_type]
                            self.box.corona_type = model_type
                            break
        elif os.path.basename(boxfile).endswith('.h5'):
            self.box.b3d = read_b3d_h5(boxfile)
            if "corona" in self.box.b3d:
                self.box.corona_type = self.box.b3d["corona"].get("attrs", {}).get("model_type")
                if self.box.corona_type in ("pot", "nlfff"):
                    self.box.corona_models[self.box.corona_type] = self.box.b3d["corona"]
        print(self.box.b3d.keys())

    @property
    def avaliable_maps(self):
        """
        Lists the available maps.

        :return: A list of available map keys.
        :rtype: list
        """
        if all(key in self.sdofitsfiles.keys() for key in HMI_B_SEGMENTS):
            return list(self.sdofitsfiles.keys()) + HMI_B_PRODUCTS
        else:
            return self.sdofitsfiles.keys()

    def corr_fov_coords(self, sunpymap, fov_coords):
        '''
        Corrects the field of view coordinates using the given map.
        :param sunpymap: The map to use for correction.
        :type sunpymap: sunpy.map.Map
        :param fov_coords: The field of view coordinates (bottom left and top right) as SkyCoord objects.
        :type fov_coords: list

        :return: Corrected field of view coordinates.
        :rtype: list
        '''
        fov_coords = [SkyCoord(Tx=fov_coords[0].Tx, Ty=fov_coords[0].Ty, frame=sunpymap.coordinate_frame),
                      SkyCoord(Tx=fov_coords[1].Tx, Ty=fov_coords[1].Ty, frame=sunpymap.coordinate_frame)]
        return fov_coords

    def _load_hmi_b_seg_maps(self, mapname, fov_coords):
        """
        Load specific HMI B segment maps required for the magnetic field vector data products.

        :param mapname: Name of the map to load.
        :type mapname: str
        :param fov_coords: The field of view coordinates (bottom left and top right) as SkyCoord objects.
        :type fov_coords: list
        :return: Loaded map object.
        :rtype: sunpy.map.Map
        :raises ValueError: If the map name is not in the expected HMI B segments.
        """
        if mapname not in HMI_B_SEGMENTS:
            raise ValueError(f"mapname: {mapname} must be one of {HMI_B_SEGMENTS}. Use loadmap method for others.")

        if mapname in self.sdomaps.keys():
            return self.sdomaps[mapname]

        print(f'fov_coords: {fov_coords}')
        loaded_map = Map(self.sdofitsfiles[mapname])
        fov_coords = self.corr_fov_coords(loaded_map, fov_coords)
        loaded_map = loaded_map.submap(fov_coords[0], top_right=fov_coords[1])
        # loaded_map = loaded_map.rotate(order=3)
        if mapname in ['azimuth']:
            if 'disambig' not in self.sdomaps.keys():
                self.sdomaps['disambig'] = Map(self.sdofitsfiles['disambig']).submap(fov_coords[0],
                                                                                     top_right=fov_coords[1])
            loaded_map = hmi_disambig(loaded_map, self.sdomaps['disambig'])

        self.sdomaps[mapname] = loaded_map
        return loaded_map

    def loadmap(self, mapname, fov_coords=None):
        """
        Loads a map from the available data.

        :param mapname: Name of the map to load.
        :type mapname: str
        :param fov_coords: Field of view coordinates (bottom left and top right) as SkyCoord objects, optional. Defaults to the entire FOV if not specified.
        :type fov_coords: list, optional
        :return: The requested map.
        :raises ValueError: If the specified map is not available.
        """
        if mapname not in self.avaliable_maps:
            raise ValueError(f"Map {mapname} is not available. mapname must be one of {self.avaliable_maps}")

        if mapname in self.sdomaps.keys():
            return self.sdomaps[mapname]

        if fov_coords is None:
            fov_coords = self.fov_coords

        if mapname in HMI_B_SEGMENTS:
            self._load_hmi_b_seg_maps(mapname, fov_coords)

        if mapname in HMI_B_PRODUCTS:
            if mapname in self.sdomaps.keys():
                return self.sdomaps[mapname]
            for key in HMI_B_SEGMENTS:
                if key not in self.sdomaps.keys():
                    self.sdomaps[key] = self._load_hmi_b_seg_maps(key, fov_coords)

            map_bp, map_bt, map_br = hmi_b2ptr(self.sdomaps['field'], self.sdomaps['inclination'],
                                               self.sdomaps['azimuth'])
            self.sdomaps['bp'] = map_bp
            self.sdomaps['bt'] = map_bt
            self.sdomaps['br'] = map_br
            return self.sdomaps[mapname]

        # Load general maps
        loaded_map = Map(self.sdofitsfiles[mapname])
        fov_coords = self.corr_fov_coords(loaded_map, fov_coords)
        self.sdomaps[mapname] = loaded_map.submap(fov_coords[0], top_right=fov_coords[1])
        return self.sdomaps[mapname]

    def make_dummy_map(self, ref_coord):
        """
        Creates a dummy map for initialization purposes.

        :param ref_coord: Reference coordinate for the map.
        :type ref_coord: `~astropy.coordinates.SkyCoord`
        :return: The created dummy map.
        :rtype: sunpy.map.Map
        """
        instrument_data = np.nan * np.ones((50, 50))
        instrument_header = make_fitswcs_header(instrument_data,
                                                ref_coord,
                                                scale=u.Quantity([10, 10]) * u.arcsec / u.pix)
        return Map(instrument_data, instrument_header)

    def init_ui(self):
        """
        Initializes the user interface for the GxBox application.
        """
        uic.loadUi(Path(__file__).parent / "UI" / "gxbox.ui", self)

        # Matplotlib Figure
        self.fig = plt.Figure(figsize=(9, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvasLayout.addWidget(self.canvas)

        # Add Matplotlib Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvasLayout.addWidget(self.toolbar)

        # self.controlLayout.SetFixedSize(300)

        self.mapBottomSelector.addItems(list(self.avaliable_maps))
        self.mapBottomSelector.setCurrentIndex(self.avaliable_maps.index(self.init_map_bottom_name))
        self.mapBottomSelector.currentTextChanged.connect(self.update_bottom_map)

        self.mapContextSelector.addItems(list(self.avaliable_maps))
        self.mapContextSelector.setCurrentIndex(self.avaliable_maps.index(self.init_map_context_name))
        self.mapContextSelector.currentTextChanged.connect(self.update_context_map)

        self.visualizeButton.setToolTip("Visualize the 3D magnetic field.")
        self.visualizeButton.clicked.connect(self.visualize_3d_magnetic_field)

        self.toggleFieldlinesButton.setToolTip("Toggle the visibility of the field lines.")
        self.toggleFieldlinesButton.clicked.connect(self.toggle_fieldlines_visibility)
        self.clearFieldlinesButton.setToolTip("Clear the field lines.")
        self.clearFieldlinesButton.clicked.connect(self.clear_fieldlines)
        self.saveFieldlinesButton.setToolTip("Save the field lines to a file.")
        self.saveFieldlinesButton.clicked.connect(
            lambda: self.save_fieldlines(f'fieldlines_{self.time.to_datetime().strftime("%Y%m%dT%H%M%S")}.pkl'))

        self.b3dModelSelector.addItems(self.box.b3dtype)
        self.b3dModelSelector.setCurrentIndex(0)

        self.bminClipCheckbox.setChecked(False)
        self.bmaxClipCheckbox.setChecked(False)

        self.cmapSelector.addItems(sorted(list(mplcmaps)))  # List all available colormaps
        self.cmapSelector.setCurrentText("viridis")  # Set a default colormap
        # self.cmapClipCheckbox.setChecked(False)

        if self.external_box is not None:
            if os.path.isfile(self.external_box):
                self.load_gxbox(self.external_box)

        self.update_plot()

        print(self.map_context.dimensions)
        map_context_aspect_ratio = (self.map_context.dimensions[1] / self.map_context.dimensions[0]).value
        window_width = 900
        window_height = int(window_width * map_context_aspect_ratio)

        # Adjust for padding, toolbar, and potential high DPI scaling
        window_width += 200  # Adjust based on your UI needs
        window_height += 0  # Includes space for toolbar and dropdowns

        self.setGeometry(100, 100, int(window_width), int(window_height))
        self.splitter.setSizes([900, 200])

    def calc_potential_field(self):
        import time
        print(f'Starting potential field computation...')
        t0 = time.time()
        if self.mapBottomSelector.currentText() != 'br':
            self.mapBottomSelector.setCurrentIndex(self.avaliable_maps.index('br'))
        maglib_lff = MagFieldLinFFF()
        bnddata = self.map_bottom.data
        bnddata[np.isnan(bnddata)] = 0.0

        maglib_lff.set_field(bnddata)
        ## the axis order in res is y, x, z. so we need to swap the first two axes, so that the order becomes x, y, z.
        self.pot_res = maglib_lff.LFFF_cube(nz=self.box.dims_pix[-1], alpha=0.0)
        print(f'Time taken to compute potential field solution: {time.time() - t0:.1f} seconds')
        obs_dr = self.box._res.to(u.km) / self._rsun_km()
        dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]
        pot_box = {
            "bx": self.pot_res['by'].swapaxes(0, 1),
            "by": self.pot_res['bx'].swapaxes(0, 1),
            "bz": self.pot_res['bz'].swapaxes(0, 1),
            "dr": np.array(dr3),
            "attrs": {"model_type": "pot"},
        }
        self.box.corona_models["pot"] = pot_box
        self.box.b3d["corona"] = pot_box
        self.box.corona_type = "pot"
        self._save_stage("POT", {"corona": pot_box})
        if self.stop_after and self.stop_after.lower() == "pot":
            return

    def calc_nlfff(self):
        import time
        from pyampp.util.compute import cutout2box
        from pyampp.gx_chromo.combo_model import combo_model

        if "pot" not in self.box.corona_models:
            self.calc_potential_field()
        if self.stop_after and self.stop_after.lower() == "pot":
            return
        pot_box = self.box.corona_models["pot"]
        bx_lff, by_lff, bz_lff = [pot_box[k].swapaxes(0, 1) for k in ("by", "bx", "bz")]
        # replace bottom boundary of lff solution with initial boundary conditions
        bvect_bottom = {}
        bvect_bottom['bz'] = self.sdomaps['br'] if 'br' in self.sdomaps.keys() else self.loadmap('br')
        bvect_bottom['bx'] = -self.sdomaps['bt'] if 'bt' in self.sdomaps.keys() else -self.loadmap('bt')
        bvect_bottom['by'] = self.sdomaps['bp'] if 'bp' in self.sdomaps.keys() else self.loadmap('bp')

        self.bvect_bottom = {}
        for k in bvect_bottom.keys():
            self.bvect_bottom[k] = bvect_bottom[k].reproject_to(self.bottom_wcs_header, algorithm="adaptive",
                                                                roundtrip_coords=False)

        self.bvect_bottom_data = {}
        for k in bvect_bottom.keys():
            self.bvect_bottom_data[k] = self.bvect_bottom[k].data
            self.bvect_bottom_data[k][np.isnan(self.bvect_bottom_data[k])] = 0.0
        bx_lff[:, :, 0] = self.bvect_bottom_data['bx']
        by_lff[:, :, 0] = self.bvect_bottom_data['by']
        bz_lff[:, :, 0] = self.bvect_bottom_data['bz']

        print(f'Starting NLFFF computation...')
        t0 = time.time()
        maglib = MagFieldProcessor()
        if self.pot_res is None:
            self.pot_res = {}
            self.pot_res['bx'] = pot_box['by'].swapaxes(0, 1)
            self.pot_res['by'] = pot_box['bx'].swapaxes(0, 1)
            self.pot_res['bz'] = pot_box['bz'].swapaxes(0, 1)
        maglib.load_cube_vars(self.pot_res)

        if self.use_potential:
            bx_nlff, by_nlff, bz_nlff = pot_box["bx"], pot_box["by"], pot_box["bz"]
            obs_dr = self.box._res.to(u.km) / self._rsun_km()
            dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]
            nlfff_box = {
                "bx": bx_nlff,
                "by": by_nlff,
                "bz": bz_nlff,
                "dr": np.array(dr3),
                "attrs": {"model_type": "pot"},
            }
            self.box.b3d["corona"] = nlfff_box
            self.box.corona_type = "pot"
        else:
            res_nlf = maglib.NLFFF()
            print(f'Time taken to compute NLFFF solution: {time.time() - t0:.1f} seconds')

            bx_nlff, by_nlff, bz_nlff = [res_nlf[k].swapaxes(0, 1) for k in ("by", "bx", "bz")]
            obs_dr = self.box._res.to(u.km) / self._rsun_km()
            dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]
            nlfff_box = {
                "bx": bx_nlff,
                "by": by_nlff,
                "bz": bz_nlff,
                "dr": np.array(dr3),
                "attrs": {"model_type": "nlfff"},
            }
            self.box.corona_models["nlfff"] = nlfff_box
            self.box.b3d["corona"] = nlfff_box
            self.box.corona_type = "nlfff"
            self._save_stage("NAS", {"corona": nlfff_box})
        if self.stop_after and self.stop_after.lower() == "nas":
            return

        # ## TODO -------------------
        # t1  = time.time()
        # lines = maglib.lines(seeds=None)

        # def reproj(bottom_map):
        #     return bottom_map.reproject_to(self.bottom_wcs_header, algorithm="adaptive",
        #                                                         roundtrip_coords=False)
        # base_bz = reproj(self.loadmap("magnetogram"))
        # base_ic = reproj(self.loadmap("continuum"))

        # header_field = self.sdomaps["field"].wcs.to_header()
        # field_frame = self.sdomaps["field"].center.heliographic_carrington.frame
        # lon, lat = field_frame.lon.value, field_frame.lat.value

        # obs_time = self.box._frame_obs.obstime
        # dsun_obs = header_field["DSUN_OBS"]
        # header = {"lon": lon, "lat": lat, "dsun_obs": dsun_obs, "obs_time": str(obs_time.iso)}

        # obs_dr = self.box._res.to(u.km) / (696000 * u.km)
        # dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]

        # chromo_box = combo_model(self.box.b3d['nlfff'], dr3, base_bz.data.T, base_ic.data.T)
        # chromo_box["avfield"] = lines["av_field"]
        # chromo_box["physlength"] = lines["phys_length"] * dr3[0]
        # chromo_box["status"] = lines["voxel_status"]
        # self.box.b3d["chromo"] = chromo_box
        # import IPython;
        # IPython.embed()

        def reproj(bottom_map):
            return bottom_map.reproject_to(self.bottom_wcs_header, algorithm="adaptive",
                                                                roundtrip_coords=False)
        base_bz = reproj(self.loadmap("magnetogram"))
        base_ic = reproj(self.loadmap("continuum"))

        header_field = self.sdomaps["field"].wcs.to_header()
        field_frame = self.sdomaps["field"].center.heliographic_carrington.frame
        lon, lat = field_frame.lon.value, field_frame.lat.value

        obs_time = self.box._frame_obs.obstime
        dsun_obs = header_field["DSUN_OBS"]
        header = {"lon": lon, "lat": lat, "dsun_obs": dsun_obs, "obs_time": str(obs_time.iso)}

        obs_dr = self.box._res.to(u.km) / self._rsun_km()
        dr3 = [obs_dr.value, obs_dr.value, obs_dr.value]

        chromo_box = combo_model(self.box.b3d['corona'], dr3, base_bz.data.T, base_ic.data.T)
        for k in ["codes", "apex_idx", "start_idx", "end_idx", "seed_idx",\
                  "av_field", "phys_length", "voxel_status"]:
            chromo_box[k] = lines[k]
        chromo_box["phys_length"] *= dr3[0]
        chromo_box["attrs"] = header
        gen_chromo = self._make_gen_chromo(chromo_box)
        self._save_stage("NAS.GEN", {"chromo": gen_chromo})
        if self.stop_after and self.stop_after.lower() == "gen":
            return

        self.box.b3d["chromo"] = chromo_box
        self._save_stage("NAS.CHR", {"chromo": self.box.b3d["chromo"]})
        print(f"Time taken to compute chromosphere model: {time.time() - t1:.1f} seconds")

    def calc_chromo_model(self):
        pass


    def visualize_3d_magnetic_field(self):
        """
        Launches the MagneticFieldVisualizer to visualize the 3D magnetic field data.
        """

        box_norm_direction = self.box_norm_direction()
        box_view_up = self.box_view_up()
        b3dtype = self.b3dModelSelector.currentText()
        # print(f'type of self.box.b3d is {type(self.box.b3d)}')
        # print(f'value of self.box.b3d is {self.box.b3d}')
        # if b3dtype == 'pot':
        if self.box.b3d["corona"] is not None and (self.box.corona_type is None or self.box.corona_type == b3dtype):
            print(f'Using existing {self.box.corona_type or "corona"} solution...')
        elif b3dtype in self.box.corona_models:
            self.box.b3d["corona"] = self.box.corona_models[b3dtype]
            self.box.corona_type = b3dtype
            print(f'Using existing {b3dtype} solution...')
        else:
            if b3dtype == 'pot':
                self.calc_potential_field()
            elif b3dtype == 'nlfff':
                self.calc_nlfff()

        self.visualizer = MagFieldViewer(self.box, parent=self, box_norm_direction=box_norm_direction,
                                         box_view_up=box_view_up, time=self.time, b3dtype=b3dtype)
        self.visualizer.show()

    def update_bottom_map(self, map_name):
        """
        Updates the bottom map displayed in the UI.

        :param map_name: Name of the map to be updated.
        :type map_name: str
        """
        if self.map_bottom_im is not None:
            self.map_bottom_im.remove()
        map_bottom = self.sdomaps[map_name] if map_name in self.sdomaps.keys() else self.loadmap(map_name)
        self.map_bottom = map_bottom.reproject_to(self.bottom_wcs_header, algorithm="adaptive",
                                                  roundtrip_coords=False)

        self.box._dims_pix[0] = self.map_bottom.data.shape[1]
        self.box._dims_pix[1] = self.map_bottom.data.shape[0]

        self.map_bottom_im = self.map_bottom.plot(axes=self.axes, autoalign=True)
        # self.update_plot()
        self.canvas.draw()

    def update_context_map(self, map_name):
        """
        Updates the context map displayed in the UI.

        :param map_name: Name of the map to be updated.
        :type map_name: str
        """
        if self.map_context_im is not None:
            self.map_context_im.remove()
        if map_name in HMI_B_SEGMENTS + HMI_B_PRODUCTS + ['magnetogram', 'continuum']:
            map_context_prev = self.map_context
        self.map_context = self.sdomaps[map_name] if map_name in self.sdomaps.keys() else self.loadmap(map_name)
        if map_name in HMI_B_SEGMENTS + HMI_B_PRODUCTS + ['magnetogram', 'continuum']:
            self.map_context = self.map_context.rotate(order=3)
            self.map_context = self.map_context.reproject_to(map_context_prev.wcs)
        # else:
        self.map_context_im = self.map_context.plot(axes=self.axes)
        self.canvas.draw()

    @property
    def get_axes_world_coords(self):
        # Get pixel bounds
        pixel_coords_x = self.axes.get_xlim()
        pixel_coords_y = self.axes.get_ylim()

        # Convert pixel bounds to world coordinates
        world_coords = self.map_context.wcs.pixel_to_world(pixel_coords_x, pixel_coords_y)

        return world_coords

    def get_axes_pixel_coords(self, coords_world=None):
        if coords_world is None:
            coords_world = self.get_axes_world_coords
        world_coords = SkyCoord(Tx=coords_world.Tx, Ty=coords_world.Ty, frame=self.map_context.coordinate_frame)
        pixel_coords_x, pixel_coords_y = self.map_context.wcs.world_to_pixel(world_coords)
        return pixel_coords_x, pixel_coords_y

    # def toggle_cmap_clip(self):
    #     """
    #     Toggles the clipping of the colormap.
    #     """
    #     if self.cmap_clip_checkbox.isChecked():
    #         self.bmaxClipCheckbox.setChecked(True)
    #         self.bminClipCheckbox.setChecked(True)
    #     else:
    #         self.bmaxClipCheckbox.setChecked(False)
    #         self.bminClipCheckbox.setChecked(False)

    def toggle_fieldlines_visibility(self):
        """
        Toggles the visibility of the fieldlines from the plot.
        """
        self.fieldlines_show_status = not self.fieldlines_show_status
        print("Toggling fieldlines visibility: ", self.fieldlines_show_status)

        if len(self.fieldlines_line_collection) == 0:
            return

        for lc in self.fieldlines_line_collection:
            lc.set_visible(self.fieldlines_show_status)

        if self.fieldlines_show_status:
            self.toggleFieldlinesButton.setText("Hide")
        else:
            self.toggleFieldlinesButton.setText("Show")

        self.canvas.draw()

    def clear_fieldlines(self):
        """
        Clears the fieldlines from the plot.
        """
        if len(self.fieldlines_line_collection) == 0:
            return

        while self.fieldlines_line_collection:
            lc = self.fieldlines_line_collection.pop()
            lc.remove()  # Remove the LineCollection from the axes
        self.canvas.draw()

    def update_plot(self, show_bound_box=True, show_box_outline=True):
        """
        Updates the plot with the current data and settings.
        """
        if self.axes is not None:
            self.axes_world_coords = self.get_axes_world_coords
        self.fig.clear()
        self.axes = self.fig.add_subplot(projection=self.map_context)
        self.axes.set_facecolor('black')
        ax = self.axes
        self.map_context_im = self.map_context.plot(axes=ax)
        self.map_context.draw_grid(axes=ax, color='w', lw=0.5)
        self.map_context.draw_limb(axes=ax, color='w', lw=1.0)

        # for edge in self.simbox.bottom_edges:
        #     ax.plot_coord(edge, color='r', ls='-', marker='', lw=1.0)
        # for edge in self.simbox.non_bottom_edges:
        #     ax.plot_coord(edge, color='r', ls='--', marker='', lw=0.5)
        if show_box_outline:
            for edge in self.box.bottom_edges:
                ax.plot_coord(edge, color='tab:red', ls='--', marker='', lw=1.0)
            for edge in self.box.non_bottom_edges:
                ax.plot_coord(edge, color='tab:red', ls='-', marker='', lw=1.0)
        # self.map_context.draw_quadrangle(self.map_bottom.bottom_left_coord, axes=ax,
        #                                  width=self.map_bottom.top_right_coord.lon - self.map_bottom.bottom_left_coord.lon,
        #                                  height=self.map_bottom.top_right_coord.lat - self.map_bottom.bottom_left_coord.lat,
        #                                  edgecolor='tab:red', linestyle='--', linewidth=0.5)
        # ax.plot_coord(self.box_center, color='r', marker='+')
        # ax.plot_coord(self.box_origin, mec='r', mfc='none', marker='o')
        if show_bound_box:
            self.map_context.draw_quadrangle(
                self.box.bounds_coords,
                axes=ax,
                edgecolor="tab:blue",
                linestyle="--",
                linewidth=0.5,
            )
        self.map_bottom_im = self.map_bottom.plot(axes=ax, autoalign=True)
        ax.set_title(ax.get_title(), pad=45)
        if self.axes_world_coords_init is None:
            self.axes_world_coords_init = self.get_axes_world_coords
        if self.axes_world_coords is not None:
            axes_pixel_coords = self.get_axes_pixel_coords()
            ax.set_xlim(axes_pixel_coords[0])
            ax.set_ylim(axes_pixel_coords[1])
        self.fig.tight_layout()
        # Refresh canvas
        self.canvas.draw()

    def extract_streamlines(self, streamlines):
        """
        Extracts individual streamlines from the streamlines data.

        :param streamlines: pyvista.PolyData
            The streamlines data.
        :return: list of numpy.ndarray
            A list of individual streamlines.
        """
        lines = []
        fields = []
        n_lines = streamlines.lines.shape[0]
        i = 0
        while i < n_lines:
            num_points = streamlines.lines[i]
            start_idx = streamlines.lines[i + 1]
            end_idx = start_idx + num_points
            line = streamlines.points[start_idx:end_idx]
            lines.append(line)
            bx = streamlines['bx'][start_idx:end_idx]
            by = streamlines['by'][start_idx:end_idx]
            bz = streamlines['bz'][start_idx:end_idx]
            magnitude = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)

            fields.append({'bx': bx, 'by': by, 'bz': bz, 'magnitude': magnitude})
            i += num_points + 1
        return lines, fields

    def plot_fieldlines(self, streamlines, z_base=0.0):
        """
        Plots the extracted fieldlines with colorization based on their magnitude.

        :param streamlines: pyvista.PolyData
            The streamlines data.
        """
        self.flines = {'coords_hcc': [], 'fields': [], 'frame_obs': self.frame_obs}

        from matplotlib.collections import LineCollection

        ax = self.axes

        # Fetch Bmin and Bmax values from input fields
        try:
            bmin = float(self.bminInput.text())
            bmax = float(self.bmaxInput.text())
        except ValueError:
            bmin = 0
            bmax = 1000

        # Normalize the magnitude values for colormap

        cmap = plt.get_cmap(self.cmapSelector.currentText())
        # Check if bounds input box is empty
        bounds_text = self.cmapDiscreteBoundsInput.text()
        if bounds_text:
            bounds = list(map(float, bounds_text.split(',')))
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = mcolors.Normalize(vmin=bmin, vmax=bmax)

        for streamlines_subset in streamlines:
            coords_hcc = []

            coords, fields = self.extract_streamlines(streamlines_subset)
            for coord, field in zip(coords, fields):
                # Convert the streamline coordinates to the gxbox frame_obs
                coord_hcc = SkyCoord(x=coord[:, 0] * u.Mm, y=coord[:, 1] * u.Mm, z=(coord[:, 2] + z_base) * u.Mm,
                                     frame=self.frame_hcc)
                coords_hcc.append(coord_hcc)
                coord_hpc = coord_hcc.transform_to(self.frame_obs)
                # ax.plot_coord(coord_hpc, '-', c='tab:blue', lw=0.3, alpha=0.5)
                xpix, ypix = self.map_context.world_to_pixel(coord_hpc)
                x = xpix.value
                y = ypix.value
                magnitude = field['magnitude']
                segments = [((x[i], y[i]), (x[i + 1], y[i + 1])) for i in range(len(x) - 1)]
                colors = [cmap(norm(value)) for value in magnitude]  # Colormap for each segment

                if self.bminClipCheckbox.isChecked() or self.bmaxClipCheckbox.isChecked():
                    bmin = 0.0
                    bmax = 5e6  ## an unrealistic large B field value for solar corona
                    if self.bminClipCheckbox.isChecked():
                        bmin = float(self.bminInput.text())
                    if self.bmaxClipCheckbox.isChecked():
                        bmax = float(self.bmaxInput.text())
                    mask = np.logical_and(magnitude >= bmin, magnitude <= bmax)
                    colors = np.array(colors)[mask]
                    segments = np.array(segments)[mask[:-1]]
                # if self.cmap_clip_checkbox.isChecked():
                #     bmin = float(self.bminInput.text())
                #     bmax = float(self.bmaxInput.text())
                #     mask = np.logical_and(magnitude >= bmin, magnitude <= bmax)
                #     colors = np.array(colors)[mask]
                #     segments = np.array(segments)[mask[:-1]]
                lc = LineCollection(segments, colors=colors, linewidths=float(self.LineWidthInput.text()))
                lc.set_alpha(float(self.LineAlphaInput.text()))
                ax.add_collection(lc)
                self.fieldlines_line_collection.append(lc)
                if not self.fieldlines_show_status:
                    lc.set_visible(False)
            self.flines['coords_hcc'].append(coords_hcc)
            self.flines['fields'].append(fields)
        self.canvas.draw()

    def save_fieldlines(self, default_filename='fieldlines.pkl'):
        """
        Saves the fieldlines data to a file. Prompts the user to select a directory and input a filename.

        :param default_filename: str
            The default name of the file to save the fieldlines data.
        """
        # Open a file dialog to select directory and input filename
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save Field Lines", default_filename, "Pickle Files (*.pkl)",
                                                  options=options)

        # Save the fieldlines if a valid filename is provided
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump(self.flines, f)
            print(f"Field lines saved to {filename}")

    def plot(self):
        """
        Plots the data in the UI.
        """
        self.update_plot()
        return self.fig

    def create_lines_of_sight(self):
        """
        Creates lines of sight for the visualization.
        """
        # The rest of the code for creating lines of sight goes here
        pass

    def visualize(self):
        """
        Visualizes the data in the UI.
        """
        # The rest of the code for visualization goes here
        pass


def _validate_frame(
        hpc: bool,
        hgc: bool,
        hgs: bool,
        coords: list[float],
        observation_time: Time,
        observer_coord: SkyCoord,
) -> SkyCoord:
    """
    Determine the box origin based on which frame flag is set.
    Exactly one of hpc, hgc, or hgs must be True.
    """
    if hpc + hgc + hgs != 1:
        raise typer.BadParameter("Exactly one coordinate frame must be specified: use either --hpc, --hgc, or --hgs.")

    rsun = 696 * u.Mm  # Solar radius in Mm

    if hpc:
        return SkyCoord(
            coords[0] * u.arcsec,
            coords[1] * u.arcsec,
            obstime=observation_time,
            observer=observer_coord,
            rsun=rsun,
            frame="helioprojective",
        )
    elif hgc:
        return SkyCoord(
            lon=coords[0] * u.deg,
            lat=coords[1] * u.deg,
            obstime=observation_time,
            radius=rsun,
            observer=observer_coord,
            frame="heliographic_carrington",
        )
    else:  # hgs
        return SkyCoord(
            lon=coords[0] * u.deg,
            lat=coords[1] * u.deg,
            obstime=observation_time,
            radius=rsun,
            observer=observer_coord,
            frame="heliographic_stonyhurst",
        )


@app.command()
def main(
        time: Optional[str] = typer.Option(
            None,
            "--time",
            help='Observation time in ISO format, e.g., "2024-05-12T00:00:00".'
        ),
        coords: Optional[Tuple[float, float]] = typer.Option(
            None,
            "--coords",
            help="Two floats: [x, y] (arcsec if HPC, degrees if HGC or HGS). Example: 0.0 0.0",
        ),
        hpc: bool = typer.Option(
            False,
            "--hpc",
            help="Use Helioprojective coordinates."
        ),
        hgc: bool = typer.Option(
            False,
            "--hgc",
            help="Use Heliographic Carrington coordinates."
        ),
        hgs: bool = typer.Option(
            False,
            "--hgs",
            help="Use Heliographic Stonyhurst coordinates."
        ),
        box_dims: Optional[Tuple[int, int, int]] = typer.Option(
            (100, 100, 100),
            "--box-dims",
            help="Three ints: box dimensions [nx, ny, nz] in pixels. Example: 100 100 100"
        ),
        box_res: Optional[float] = typer.Option(
            1.4,
            "--box-res",
            help="Box resolution in Mm per pixel."
        ),
        observer: Optional[str] = typer.Option(
            "Earth",
            "--observer",
            help="Observer location (e.g., 'earth' or a named object)."
        ),
        pad_frac: Optional[float] = typer.Option(
            0.25,
            "--pad-frac",
            help="Fractional padding applied to each side of the box (decimal)."
        ),
        data_dir: Optional[str] = typer.Option(
            DOWNLOAD_DIR,
            "--data-dir",
            help="Directory for storing downloaded data."
        ),
        gxmodel_dir: Optional[str] = typer.Option(
            GXMODEL_DIR,
            "--gxmodel-dir",
            help="Directory for storing model outputs."
        ),
        external_box: Optional[str] = typer.Option(
            os.path.abspath(os.getcwd()),
            "--external-box",
            help="Path to an external box file."
        ),
        interactive: bool = typer.Option(
            False,
            "--interactive",
            help="Enable interactive mode (drops into pdb after launching)."
        ),
        euv: bool = typer.Option(
            True,
            "--euv/--no-euv",
            help="Download AIA/EUV context maps (default: on)."
        ),
        uv: bool = typer.Option(
            True,
            "--uv/--no-uv",
            help="Download AIA/UV context maps (default: on)."
        ),
        hmifiles: Optional[str] = typer.Option(
            None,
            "--hmifiles",
            help="Path to a local HMI data directory to bypass downloads."
        ),
        entry_box: Optional[str] = typer.Option(
            None,
            "--entry-box",
            help="Path to a preexisting box to use as starting point."
        ),
        empty_box_only: bool = typer.Option(
            False,
            "--empty-box-only",
            help="Stop after empty box is generated."
        ),
        potential_only: bool = typer.Option(
            False,
            "--potential-only",
            help="Compute only the potential extrapolation and stop."
        ),
        use_potential: bool = typer.Option(
            False,
            "--use-potential",
            help="Skip NLFFF and use the potential field for downstream steps."
        ),
        jump2potential: bool = typer.Option(
            False,
            "--jump2potential",
            help="Start from entry_box and jump to potential stage."
        ),
        jump2nlfff: bool = typer.Option(
            False,
            "--jump2nlfff",
            help="Start from entry_box and jump to NLFFF stage."
        ),
        jump2lines: bool = typer.Option(
            False,
            "--jump2lines",
            help="Start from entry_box and jump to lines/GEN stage."
        ),
        jump2chromo: bool = typer.Option(
            False,
            "--jump2chromo",
            help="Start from entry_box and jump to chromo stage."
        ),
        save_empty_box: bool = typer.Option(
            False,
            "--save-empty-box",
            help="Save the empty box stage (NONE)."
        ),
        save_bounds: bool = typer.Option(
            False,
            "--save-bounds",
            help="Save the boundary stage (BND)."
        ),
        save_potential: bool = typer.Option(
            False,
            "--save-potential",
            help="Save the potential stage (POT)."
        ),
        save_nas: bool = typer.Option(
            False,
            "--save-nas",
            help="Save the NLFFF stage (NAS)."
        ),
        save_gen: bool = typer.Option(
            False,
            "--save-gen",
            help="Save the GEN stage (NAS.GEN)."
        ),
        save_chr: bool = typer.Option(
            False,
            "--save-chr",
            help="Save the CHR stage (NAS.CHR)."
        ),
        stop_after: Optional[str] = typer.Option(
            None,
            "--stop-after",
            help="Stop after stage: pot|nas|gen|chr."
        ),
        auto_visualize_last: bool = typer.Option(
            True,
            "--auto-visualize-last/--no-auto-visualize-last",
            help="Auto-launch the 3D viewer for the last stage."
        ),
        info: bool = typer.Option(
            False,
            "--info",
            help="Print all explicit and implicit options with their meanings and exit."
        ),
):
    """
    Launch the GxBox application with the specified parameters.

    :param time: Observation time in ISO format (e.g., "2024-05-12T00:00:00").
    :type time: str
    :param coords: Two floats representing [x, y] (arcsec if HPC, degrees if HGC or HGS). Example: 0.0 0.0
    :type coords: Tuple[float, float]
    :param hpc: Use Helioprojective coordinates, defaults to False
    :type hpc: bool, optional
    :param hgc: Use Heliographic Carrington coordinates, defaults to False
    :type hgc: bool, optional
    :param hgs: Use Heliographic Stonyhurst coordinates, defaults to False
    :type hgs: bool, optional
    :param box_dims: Three integers [nx, ny, nz] specifying box dimensions in pixels. Example: 100 100 100
                     Defaults to (100, 100, 100).
    :type box_dims: Tuple[int, int, int], optional
    :param box_res: Box resolution in Mm per pixel, defaults to 1.4
    :type box_res: float, optional
    :param observer: Observer location (e.g., 'Earth' or a named object), defaults to "Earth"
    :type observer: str, optional
    :param pad_frac: Fractional padding applied to each side of the box (decimal), defaults to 0.25
    :type pad_frac: float, optional
    :param data_dir: Directory for storing downloaded data, defaults to DOWNLOAD_DIR
    :type data_dir: str, optional
    :param gxmodel_dir: Directory for storing model outputs, defaults to GXMODEL_DIR
    :type gxmodel_dir: str, optional
    :param external_box: Path to an external box file, defaults to current working directory
    :type external_box: str, optional
    :param interactive: Enable interactive mode (drops into pdb after launching), defaults to False
    :type interactive: bool, optional
    :raises typer.BadParameter: If none or multiple coordinate frame flags (--hpc, --hgc, --hgs) are provided
    :return: None
    :rtype: NoneType

    Example:
    --------
    .. code-block:: bash

      gxbox \
        --time "2022-03-30T17:22:37" \
        --coords 34.44988566346035 14.26110705696788 \
        --hgs \
        --box-dims 360 180 200 \
        --box-res 0.729 \
        --pad-frac 0.25 \
        --data-dir /path/to/download_dir \
        --gxmodel-dir /path/to/gx_models_dir
    """

    def _build_execute_cmd() -> str:
        parts = [
            "gxbox",
            "--time",
            time or "<MISSING_TIME>",
            "--coords",
            str(coords[0]) if coords else "<MISSING_X>",
            str(coords[1]) if coords else "<MISSING_Y>",
        ]
        if hpc:
            parts.append("--hpc")
        elif hgc:
            parts.append("--hgc")
        elif hgs:
            parts.append("--hgs")
        parts += [
            "--box-dims",
            str(box_dims[0]) if box_dims else "<MISSING_NX>",
            str(box_dims[1]) if box_dims else "<MISSING_NY>",
            str(box_dims[2]) if box_dims else "<MISSING_NZ>",
            "--box-res",
            str(box_res) if box_res is not None else "<MISSING_BOX_RES>",
            "--pad-frac",
            str(pad_frac) if pad_frac is not None else "<MISSING_PAD_FRAC>",
            "--data-dir",
            data_dir or "<MISSING_DATA_DIR>",
            "--gxmodel-dir",
            gxmodel_dir or "<MISSING_GXMODEL_DIR>",
            "--observer",
            observer or "<MISSING_OBSERVER>",
            "--external-box",
            external_box or "<MISSING_EXTERNAL_BOX>",
        ]
        if interactive:
            parts.append("--interactive")
        if not euv:
            parts.append("--no-euv")
        if not uv:
            parts.append("--no-uv")
        if hmifiles:
            parts += ["--hmifiles", hmifiles]
        if entry_box:
            parts += ["--entry-box", entry_box]
        if empty_box_only:
            parts.append("--empty-box-only")
        if potential_only:
            parts.append("--potential-only")
        if use_potential:
            parts.append("--use-potential")
        if jump2potential:
            parts.append("--jump2potential")
        if jump2nlfff:
            parts.append("--jump2nlfff")
        if jump2lines:
            parts.append("--jump2lines")
        if jump2chromo:
            parts.append("--jump2chromo")
        if save_empty_box:
            parts.append("--save-empty-box")
        if save_bounds:
            parts.append("--save-bounds")
        if save_potential:
            parts.append("--save-potential")
        if save_nas:
            parts.append("--save-nas")
        if save_gen:
            parts.append("--save-gen")
        if save_chr:
            parts.append("--save-chr")
        if stop_after:
            parts += ["--stop-after", stop_after]
        if not auto_visualize_last:
            parts.append("--no-auto-visualize-last")
        return " ".join(shlex.quote(p) for p in parts)

    def _print_info() -> None:
        print("gxbox options (explicit and implicit):")
        print("")
        time_val = time if time is not None else "<MISSING> (required)"
        coords_val = f"{coords[0]} {coords[1]}" if coords is not None else "<MISSING> (required)"
        frame = "--hpc" if hpc else "--hgc" if hgc else "--hgs" if hgs else "<MISSING_FRAME>"
        box_dims_val = f"{box_dims[0]} {box_dims[1]} {box_dims[2]}" if box_dims else "<MISSING>"
        print(f"  --time           Observation time in ISO format. [value: {time_val}]")
        print(f"  --coords         Center coordinates. [value: {coords_val}]")
        print(f"  {frame:<15} Coordinate frame flag (exactly one of --hpc/--hgc/--hgs).")
        print(f"  --box-dims       Box dimensions [nx ny nz] in pixels. [value: {box_dims_val}]")
        print(f"  --box-res        Box resolution in Mm per pixel. [value: {box_res}]")
        print(f"  --pad-frac       Fractional padding applied to each side. [value: {pad_frac}]")
        print(f"  --observer       Observer location. [value: {observer}]")
        print(f"  --data-dir       Directory for downloaded data. [value: {data_dir}]")
        print(f"  --gxmodel-dir    Directory for model outputs. [value: {gxmodel_dir}]")
        print(f"  --external-box   Path to an external box file. [value: {external_box}]")
        print(f"  --interactive    Drop into pdb after launching. [value: {interactive}]")
        print(f"  --euv            Download AIA/EUV context maps. [value: {euv}]")
        print(f"  --uv             Download AIA/UV context maps. [value: {uv}]")
        print(f"  --hmifiles       Local HMI data directory. [value: {hmifiles}]")
        print(f"  --entry-box      Preexisting box path. [value: {entry_box}]")
        print(f"  --empty-box-only Stop after empty box is generated. [value: {empty_box_only}]")
        print(f"  --potential-only Stop after potential extrapolation. [value: {potential_only}]")
        print(f"  --use-potential  Skip NLFFF and use potential downstream. [value: {use_potential}]")
        print(f"  --jump2potential Jump to potential stage (requires --entry-box). [value: {jump2potential}]")
        print(f"  --jump2nlfff     Jump to NLFFF stage (requires --entry-box). [value: {jump2nlfff}]")
        print(f"  --jump2lines     Jump to GEN stage (requires --entry-box). [value: {jump2lines}]")
        print(f"  --jump2chromo    Jump to chromo stage (requires --entry-box). [value: {jump2chromo}]")
        print(f"  --save-empty-box Save NONE stage. [value: {save_empty_box}]")
        print(f"  --save-bounds    Save BND stage. [value: {save_bounds}]")
        print(f"  --save-potential Save POT stage. [value: {save_potential}]")
        print(f"  --save-nas       Save NAS stage. [value: {save_nas}]")
        print(f"  --save-gen       Save NAS.GEN stage. [value: {save_gen}]")
        print(f"  --save-chr       Save NAS.CHR stage. [value: {save_chr}]")
        print(f"  --stop-after     Stop after stage (pot|nas|gen|chr). [value: {stop_after}]")
        print(f"  --auto-visualize-last Auto-launch 3D viewer for last stage. [value: {auto_visualize_last}]")
        print("")
        print("Implicit behaviors:")
        print("  - Exactly one coordinate frame flag must be set (--hpc/--hgc/--hgs).")
        if time is None:
            print("  - WARNING: --time is required for actual execution.")
        if coords is None:
            print("  - WARNING: --coords is required for actual execution.")
        if not (hpc or hgc or hgs):
            print("  - WARNING: one of --hpc/--hgc/--hgs is required for actual execution.")
        if (jump2potential or jump2nlfff or jump2lines or jump2chromo) and not entry_box:
            print("  - WARNING: jump2* flags are ignored unless --entry-box is provided.")
        print("  - The gxbox GUI launches unless --info is used.")
        print("  - The last stage is always saved; additional stages depend on save flags.")
        print("  - The full gxbox command is recorded into HDF5 metadata/execute.")

    if info:
        _print_info()
        raise SystemExit(0)

    if time is None:
        raise typer.BadParameter("--time is required unless --info is used.")
    if coords is None:
        raise typer.BadParameter("--coords is required unless --info is used.")
    if not (hpc or hgc or hgs):
        raise typer.BadParameter("Exactly one coordinate frame must be specified: use either --hpc, --hgc, or --hgs.")
    if empty_box_only:
        save_empty_box = True
        stop_after = "none"
    if potential_only:
        save_potential = True
        stop_after = "pot"

    observation_time = Time(time)

    # Determine observer location
    observer_coord = get_earth(observation_time) if observer.lower() == "earth" else SkyCoord.from_name(observer)

    # Validate and compute box origin
    box_origin = _validate_frame(hpc, hgc, hgs, coords, observation_time, observer_coord)

    # Convert box dimensions and resolution to astropy Quantities
    dims = u.Quantity(box_dims, u.pix)
    resolution = box_res * u.Mm

    # Instantiate Qt application and GxBox
    app_instance = QApplication([])
    execute_cmd = _build_execute_cmd()
    gxbox = GxBox(
        observation_time,
        observer_coord,
        box_origin,
        dims,
        resolution,
        pad_frac=pad_frac,
        data_dir=data_dir,
        gxmodel_dir=gxmodel_dir,
        external_box=external_box,
        execute_cmd=execute_cmd,
        euv=euv,
        uv=uv,
        hmifiles=hmifiles,
        entry_box=entry_box,
        empty_box_only=empty_box_only,
        potential_only=potential_only,
        use_potential=use_potential,
        jump2potential=jump2potential,
        jump2nlfff=jump2nlfff,
        jump2lines=jump2lines,
        jump2chromo=jump2chromo,
        save_empty_box=save_empty_box,
        save_bounds=save_bounds,
        save_potential=save_potential,
        save_nas=save_nas,
        save_gen=save_gen,
        save_chr=save_chr,
        stop_after=stop_after,
        auto_visualize_last=auto_visualize_last,
    )
    gxbox.show()

    if interactive:
        import pdb

        pdb.set_trace()

    app_instance.exec_()


if __name__ == "__main__":
    app()
