import unittest

import os
import tempfile
import numpy as np

from tests import test_convert_files as tst

from terratools import convert_files
from terratools import terra_model
from terratools.terra_model import TerraModel

import netCDF4 as nc


class TestReadLayer(unittest.TestCase):
    def test_read_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            oldfilepath = os.path.join(directory, "test_file_old.nc")
            oldfile = tst.make_old_layer(oldfilepath)

            convert_files.convert_layer([oldfilepath], replace=True)

            aa = nc.Dataset(oldfilepath)

            layer = terra_model.read_netcdf([f"{oldfilepath}"])

            self.assertEqual(np.shape(layer.get_lateral_points())[0], 2)
            self.assertEqual(np.shape(layer.get_lateral_points())[1], 64)
            self.assertEqual(len(layer.get_radii()), 1)
            self.assertRaises(
                terra_model.LayerMethodError,
                layer.get_1d_profile,
                "t",
                0,
                0,
            )
            self.assertRaises(
                terra_model.LayerMethodError, layer.plot_section, "t", 0, 0, 0, 10
            )


if __name__ == "__main__":
    unittest.main()
