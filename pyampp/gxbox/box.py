import itertools
from contextlib import nullcontext

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective, HeliographicStonyhurst
from sunpy.map import make_fitswcs_header
try:
    from sunpy.coordinates.screens import SphericalScreen
except Exception:  # pragma: no cover
    SphericalScreen = None


class Box:
    """
    Represents a 3D box in solar or observer coordinates defined by its origin, center, dimensions, and resolution.

    This class calculates and stores the coordinates of the box's edges, differentiating between bottom edges and other
    edges. It is designed to integrate with solar physics data analysis frameworks such as SunPy and Astropy.
    """

    def __init__(self, frame_obs, box_origin, box_center, box_dims, box_res):
        self._frame_obs = frame_obs
        if hasattr(Helioprojective, "assume_spherical_screen"):
            screen_ctx = Helioprojective.assume_spherical_screen(frame_obs.observer)
        elif SphericalScreen is not None:
            screen_ctx = SphericalScreen(frame_obs.observer)
        else:
            screen_ctx = nullcontext()
        with (screen_ctx or nullcontext()):
            self._origin = box_origin
            self._center = box_center
        self._dims = box_dims / u.pix * box_res
        self._res = box_res
        self._dims_pix = np.int_(box_dims.value)
        self.corners = list(itertools.product(self._dims[0] / 2 * [-1, 1],
                                              self._dims[1] / 2 * [-1, 1],
                                              self._dims[2] / 2 * [-1, 1]))
        self.edges = [edge for edge in itertools.combinations(self.corners, 2)
                      if np.count_nonzero(u.Quantity(edge[0]) - u.Quantity(edge[1])) == 1]
        self._bottom_edges = None
        self._non_bottom_edges = None
        self._calculate_edge_types()
        self.b3dtype = ['pot', 'nlfff']
        self.b3d = {"corona": None, "chromo": None}
        self.corona_models = {}
        self.corona_type = None

    @property
    def dims_pix(self):
        return self._dims_pix

    @property
    def grid_coords(self):
        return self._get_grid_coords(self._center)

    def _get_grid_coords(self, grid_center):
        grid_coords = {}
        grid_coords['x'] = np.linspace(grid_center.x.to(self._dims.unit) - self._dims[0] / 2,
                                       grid_center.x.to(self._dims.unit) + self._dims[0] / 2, self._dims_pix[0])
        grid_coords['y'] = np.linspace(grid_center.y.to(self._dims.unit) - self._dims[1] / 2,
                                       grid_center.y.to(self._dims.unit) + self._dims[1] / 2, self._dims_pix[1])
        grid_coords['z'] = np.linspace(grid_center.z.to(self._dims.unit) - self._dims[2] / 2,
                                       grid_center.z.to(self._dims.unit) + self._dims[2] / 2, self._dims_pix[2])
        grid_coords['frame'] = self._frame_obs
        return grid_coords

    def _get_edge_coords(self, edges, box_center):
        return [SkyCoord(x=box_center.x + u.Quantity([edge[0][0], edge[1][0]]),
                         y=box_center.y + u.Quantity([edge[0][1], edge[1][1]]),
                         z=box_center.z + u.Quantity([edge[0][2], edge[1][2]]),
                         frame=box_center.frame) for edge in edges]

    def _get_bottom_cea_header(self):
        origin = self._origin.transform_to(HeliographicStonyhurst)
        shape = self._dims[:-1][::-1] / self._res.to(self._dims.unit)
        shape = list(shape.value)
        shape = [int(np.ceil(s)) for s in shape]
        rsun = origin.rsun.to(self._res.unit)
        scale = np.arcsin(self._res / rsun).to(u.deg) / u.pix
        scale = u.Quantity((scale, scale))
        bottom_cea_header = make_fitswcs_header(shape, origin,
                                                scale=scale, projection_code='CEA')
        bottom_cea_header['OBSRVTRY'] = 'None'
        return bottom_cea_header

    def _get_bottom_top_header(self, dsun_obs=None):
        origin = self._origin.transform_to(self._frame_obs)
        shape = self._dims[:-1][::-1] / self._res.to(self._dims.unit)
        shape = list(shape.value)
        shape = [int(np.ceil(s)) for s in shape]
        rsun = origin.rsun.to(self._res.unit)
        if dsun_obs is None:
            dsun_obs = origin.observer.radius.to(self._res.unit)
        else:
            dsun_obs = u.Quantity(dsun_obs).to(self._res.unit)
        # Match IDL TOP scaling: dx_arcsec ~ dx_km / (DSUN_OBS - RSUN).
        scale = ((self._res / (dsun_obs - rsun)).to(u.dimensionless_unscaled) * u.rad).to(u.arcsec) / u.pix
        scale = u.Quantity((scale, scale))
        bottom_top_header = make_fitswcs_header(shape, origin, scale=scale, projection_code='TAN')
        bottom_top_header['OBSRVTRY'] = 'None'
        return bottom_top_header

    def _calculate_edge_types(self):
        min_z = min(corner[2] for corner in self.corners)
        bottom_edges, non_bottom_edges = [], []
        for edge in self.edges:
            if edge[0][2] == min_z and edge[1][2] == min_z:
                bottom_edges.append(edge)
            else:
                non_bottom_edges.append(edge)
        self._bottom_edges = self._get_edge_coords(bottom_edges, self._center)
        self._non_bottom_edges = self._get_edge_coords(non_bottom_edges, self._center)

    def _get_bounds_coords(self, edges, bltr=False, pad_frac=0.0):
        xx = []
        yy = []
        for edge in edges:
            xx.append(edge.transform_to(self._frame_obs).Tx)
            yy.append(edge.transform_to(self._frame_obs).Ty)
        unit = xx[0][0].unit
        min_x = np.min(xx)
        max_x = np.max(xx)
        min_y = np.min(yy)
        max_y = np.max(yy)
        if pad_frac > 0:
            _pad = pad_frac * np.max([max_x - min_x, max_y - min_y, 20])
            min_x -= _pad
            max_x += _pad
            min_y -= _pad
            max_y += _pad
        if bltr:
            bottom_left = SkyCoord(min_x * unit, min_y * unit, frame=self._frame_obs)
            top_right = SkyCoord(max_x * unit, max_y * unit, frame=self._frame_obs)
            return [bottom_left, top_right]
        else:
            coords = SkyCoord(Tx=[min_x, max_x] * unit, Ty=[min_y, max_y] * unit,
                              frame=self._frame_obs)
            return coords

    def bounds_coords_bl_tr(self, pad_frac=0.0):
        return self._get_bounds_coords(self.all_edges, bltr=True, pad_frac=pad_frac)

    @property
    def bounds_coords(self):
        return self._get_bounds_coords(self.all_edges)

    @property
    def bottom_bounds_coords(self):
        return self._get_bounds_coords(self.bottom_edges)

    @property
    def bottom_cea_header(self):
        return self._get_bottom_cea_header()

    def bottom_top_header(self, dsun_obs=None):
        return self._get_bottom_top_header(dsun_obs=dsun_obs)

    @property
    def bottom_edges(self):
        return self._bottom_edges

    @property
    def non_bottom_edges(self):
        return self._non_bottom_edges

    @property
    def all_edges(self):
        return self.bottom_edges + self.non_bottom_edges
