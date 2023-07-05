import unittest
import numpy as np

_CARTOPY_INSTALLED = True
try:
    from cartopy.mpl.geoaxes import GeoAxesSubplot
except ImportError as exception:
    _CARTOPY_INSTALLED = False

from matplotlib.figure import Figure

from terratools import plot


def needs_cartopy(func):
    """
    Decorator which wraps a test function taking an instance of
    `unittest.TestCase` and does not run the test case if Cartopy is not
    installed.
    """
    if _CARTOPY_INSTALLED:
        return func
    else:

        def returns_none(self):
            return None

        return returns_none


class TestLayerGrid(unittest.TestCase):
    def test_layer_grid_no_cartopy(self):
        n = 20
        lon = 360 * np.random.rand(n)
        lat = 180 * (np.random.rand(n) - 0.5)
        radius = 4000
        values = np.random.rand(n)

        if not _CARTOPY_INSTALLED:
            with self.assertRaises(ImportError):
                plot.layer_grid(lon, lat, radius, values, extent=(1, 2, 3))

    @needs_cartopy
    def test_layer_grid_errors(self):
        n = 20
        lon = 360 * np.random.rand(n)
        lat = 180 * (np.random.rand(n) - 0.5)
        radius = 4000
        values = np.random.rand(n)

        with self.assertRaises(ValueError):
            plot.layer_grid(lon, lat, radius, values, extent=(1, 2, 3))
        with self.assertRaises(ValueError):
            plot.layer_grid(lon, lat, radius, values, extent=(2, 1, 1, 2))
        with self.assertRaises(ValueError):
            plot.layer_grid(lon, lat, radius, values, extent=(1, 2, 2, 1))

        with self.assertRaises(ValueError):
            plot.layer_grid(lon, lat, radius, values, delta=-1)

        with self.assertRaises(ValueError):
            plot.layer_grid(lon, lat, radius, values, method="unknown method")

    @needs_cartopy
    def test_layer_grid(self):
        """Just test that no errors are thrown when creating a plot and
        that we return the right things"""
        n = 20
        lon = 360 * np.random.rand(n)
        lat = 180 * (np.random.rand(n) - 0.5)
        radius = 4000
        values = np.random.rand(n)

        fig, ax = plot.layer_grid(lon, lat, radius, values)

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, GeoAxesSubplot)
