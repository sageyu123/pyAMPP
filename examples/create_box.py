import astropy.time
import sunpy.sun.constants
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Heliocentric, Helioprojective,get_earth
import astropy.units as u

from PyQt5.QtWidgets import QApplication
app = QApplication([])
from pyampp.gxbox.gxbox_factory import GxBox

time = astropy.time.Time('2020-12-01T20:00:00')
observer = get_earth(time)
box_origin = SkyCoord(450 * u.arcsec, -320 * u.arcsec, obstime=time, observer=observer, rsun=696 * u.Mm, frame='helioprojective')
box_dimensions = u.Quantity([400, 300, 300]) * u.pix
box_res = 1.4 * u.Mm


gxbox = GxBox(time, observer, box_origin, box_dimensions)
gxbox.show()
app.exec_()