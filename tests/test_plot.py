import unittest
import numpy as np

from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib.figure import Figure

from terratools import plot

class TestLayerGrid(unittest.TestCase):
    def test_layer_grid_errors(self):
        n = 20
        lon = 360*np.random.rand(n)
        lat = 180*(np.random.rand(n) - 0.5)
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

    def test_layer_grid(self):
        """Just test that no errors are thrown when creating a plot and
        that we return the right things"""
        n = 20
        lon = 360*np.random.rand(n)
        lat = 180*(np.random.rand(n) - 0.5)
        radius = 4000
        values = np.random.rand(n)

        fig, ax = plot.layer_grid(lon, lat, radius, values)

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, GeoAxesSubplot)
