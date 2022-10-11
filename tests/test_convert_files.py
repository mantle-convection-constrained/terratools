"""
Tests conversion of old to new file types
"""

import unittest
import numpy as np
from netCDF4 import Dataset
from terratools import terra_model
from terratools import convert_files

# from terratools import test_terra_model_read
import tempfile
import os


def make_old_file(filename):
    """
    Make an 'old' netcdf file
    """
    nps = 64
    nlayers = 3
    compositions = 2
    c_hist_name = ["BasaltFrac", "LherzFrac"]

    file = Dataset(filename, mode="w")
    nps_dim = file.createDimension("nps", nps)
    depths_dim = file.createDimension("Depths", nlayers)

    fields = {
        "Temperature": {"units": "K"},
        "Velocity_x": {"units": "Km/s"},
        "Velocity_y": {"units": "Km/s"},
        "Velocity_z": {"units": "Km/s"},
        "BasaltFrac": {"units": ""},
        "LherzFrac": {"units": ""},
    }

    depths_var = file.createVariable("Depths", terra_model.COORDINATE_TYPE, ("Depths",))
    depths_var.units = "Km"

    lon_var = file.createVariable(
        "Longitude", terra_model.COORDINATE_TYPE, ("Depths", "nps")
    )
    lon_var.units = "Degrees"

    lat_var = file.createVariable(
        "Latitude", terra_model.COORDINATE_TYPE, ("Depths", "nps")
    )
    lat_var.units = "Degrees"

    depths_var[:] = np.linspace(0, 2890, nlayers)
    for layer in range(nlayers):
        lon_var[layer, :] = np.linspace(0, 360, nps)
        lat_var[layer, :] = np.linspace(-90, 90, nps)

        for field in fields:
            fields[field]["vals"] = np.random.rand(nlayers, nps).astype(
                terra_model.VALUE_TYPE
            )

    for field in fields:
        this_var = file.createVariable(field, terra_model.VALUE_TYPE, ("Depths", "nps"))
        this_var[:, :] = fields[field]["vals"]
        if len(fields[field]["units"]) > 0:
            this_var.units = fields[field]["units"]

    return file


class TestConvertFiles(unittest.TestCase):
    def test_convert_files(self):
        with tempfile.TemporaryDirectory() as directory:
            oldfilepath = os.path.join(directory, "test_file_old.nc")
            oldfile = make_old_file(oldfilepath)

            convert_files.convert([oldfilepath], test=True)

            newfile = Dataset(oldfilepath)

            self.assertEqual(
                newfile.dimensions["nps"].size, oldfile.dimensions["nps"].size
            )
            self.assertEqual(
                newfile.dimensions["depths"].size, oldfile.dimensions["Depths"].size
            )
            self.assertTrue(
                np.all(newfile["temperature"][:, :] == oldfile["Temperature"][:, :])
            )
            self.assertTrue(
                np.all(newfile["velocity_x"][:, :] == oldfile["Velocity_x"][:, :])
            )
            self.assertTrue(
                np.all(newfile["velocity_y"][:, :] == oldfile["Velocity_y"][:, :])
            )
            self.assertTrue(
                np.all(newfile["velocity_z"][:, :] == oldfile["Velocity_z"][:, :])
            )
            self.assertTrue(
                np.all(
                    newfile["composition_fractions"][1, :, :]
                    == oldfile["LherzFrac"][:, :]
                )
            )
            self.assertTrue(
                np.all(
                    newfile["composition_fractions"][0, :, :]
                    == 1 - oldfile["BasaltFrac"][:, :] - oldfile["LherzFrac"][:, :]
                )
            )


if __name__ == "__main__":

    unittest.main()
