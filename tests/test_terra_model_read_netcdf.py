"""
Test the reading of files in the NetCDF format
"""

import unittest

import numpy as np
import os
import re
import tempfile

from netCDF4 import Dataset
from terratools import terra_model


def select_indices(array, indices):
    """Slice an array with arbitrary indices"""
    for (axis, inds) in enumerate(indices):
        if inds is not None:
            array = np.take(array, inds, axis=axis)
    return array


def write_seismic_netcdf_files(
    filebase, lon, lat, radii, fields, nfiles=3, c_hist_names=None
):
    """
    Write a set of files in the correct format to `filebase`.
    """
    if "c_hist" in fields.keys() and c_hist_names is None:
        raise ValueError("c_hist_names is required when the c_hist field is present")

    # Total number of points
    npts = len(lon)
    assert len(lat) == npts, "Number of lon and lat points must be the same"
    # Number of radii
    nlayers = len(radii)
    # Indices for indexing into fields and radii (reversed as depth is written increasing)
    depth_inds = np.flip(np.arange(0, nlayers))
    depths = np.flip(6370 - radii)

    # Spread points about files, with 1st to (nfiles - 1)th files having...
    nps_first = npts // nfiles
    # ...and the last file having nps_last points
    nps_last = npts - (nfiles - 1) * nps_first

    for ifile in range(nfiles):
        filename = filebase + f"{ifile}.nc"

        # Last file gets the remainder
        nps = nps_last if ifile == (nfiles - 1) else nps_first

        # Lateral point indices for this file
        index1 = nps_first * ifile
        index2 = index1 + nps - 1
        # We add an extra random point from outside the point range
        duplicate_index = index2 + (-nps if ifile == nfiles - 1 else 1)
        # The final index is ignored in Python, so this is right
        lateral_inds = np.append(np.arange(index1, index2 + 1), [duplicate_index])

        # Coordinates for this file, including duplicates
        lon_file = lon[lateral_inds]
        lat_file = lat[lateral_inds]

        # Open NetCDF file for writing
        file = Dataset(filename, mode="w")

        # Set dimensions
        # + 1 because we add a duplicate point at the end
        nps_dim = file.createDimension("nps", nps + 1)
        depths_dim = file.createDimension("Depths", nlayers)

        depths_var = file.createVariable(
            "Depths", terra_model.COORDINATE_TYPE, ("Depths",)
        )
        depths_var.units = "Km"

        lon_var = file.createVariable(
            "Longitude", terra_model.COORDINATE_TYPE, ("Depths", "nps")
        )
        lon_var.units = "Degrees"

        lat_var = file.createVariable(
            "Latitude", terra_model.COORDINATE_TYPE, ("Depths", "nps")
        )
        lat_var.units = "Degrees"

        # Fill in coordinates
        # Depths in decreasing depth order, which is opposite to our convention
        depths_var[:] = depths
        lon_var[:, :] = np.transpose(
            np.reshape(np.repeat(lon_file, nlayers), (-1, nlayers))
        )
        lat_var[:, :] = np.transpose(
            np.reshape(np.repeat(lat_file, nlayers), (-1, nlayers))
        )

        for field_name, field_vals in fields.items():
            is_scalar = terra_model._is_scalar_field(field_name)
            var_names = terra_model._variable_names_from_field(field_name)
            units = re.findall(r"\[(.*)\]", terra_model._ALL_FIELDS[field_name])[0]

            if is_scalar:
                var_name = var_names[0]
                this_var = file.createVariable(
                    var_name, terra_model.VALUE_TYPE, ("Depths", "nps")
                )
                this_var[:, :] = select_indices(field_vals, (depth_inds, lateral_inds))
                if units != "unitless":
                    this_var.units = units.title()

            # Special case the other things for now as it's just for testing
            elif field_name == "u_xyz":
                for (icomp, var_name) in enumerate(var_names):
                    this_var = file.createVariable(
                        var_name, terra_model.VALUE_TYPE, ("Depths", "nps")
                    )
                    this_var[:, :] = select_indices(
                        field_vals[:, :, icomp], (depth_inds, lateral_inds)
                    )
                    this_var.units = units.title()

            elif field_name == "c_hist":
                for (icomp, comp_name) in enumerate(c_hist_names):
                    var_name = comp_name.title() + "Frac"
                    this_var = file.createVariable(
                        var_name, terra_model.VALUE_TYPE, ("Depths", "nps")
                    )
                    this_var[:, :] = select_indices(
                        field_vals[:, :, icomp], (depth_inds, lateral_inds)
                    )

        file.close()

    return [filebase + f"{ifile}.nc" for ifile in range(nfiles)]


def random_model(npts, nlayers):
    """
    Create a random model with fields t, c_hist and u_xyz, with named
    compositions
    """
    coordinate_type = terra_model.COORDINATE_TYPE
    value_type = terra_model.VALUE_TYPE

    lon = 360 * np.random.rand(npts).astype(coordinate_type)
    lat = 180 * (np.random.rand(npts).astype(coordinate_type) - 0.5)
    r = 3480 + (6370 - 3480) * np.sort(np.random.rand(nlayers).astype(coordinate_type))

    fields = {
        "t": np.random.rand(nlayers, npts).astype(value_type),
        "u_xyz": np.random.rand(nlayers, npts, 3).astype(value_type),
        "c_hist": np.random.rand(nlayers, npts, 2).astype(value_type),
    }
    c_hist_names = ["basalt", "lherz"]

    return lon, lat, r, fields, c_hist_names


class TestTerraModelReadNetCDF(unittest.TestCase):
    def test_terra_model_read_netcdf(self):
        npts = 64
        nlayers = 3
        lon, lat, r, fields, c_hist_names = random_model(npts, nlayers)

        with tempfile.TemporaryDirectory() as dir:
            filebase = os.path.join(dir, "test_netcdf_file_")
            filenames = write_seismic_netcdf_files(
                filebase, lon, lat, r, fields, c_hist_names=c_hist_names
            )
            model = terra_model.read_netcdf(filenames)

        mlon, mlat = model.get_lateral_points()
        mr = model.get_radii()
        self.assertEqual(len(mlon), npts)
        self.assertEqual(len(mr), nlayers)
        self.assertTrue(np.all(lon == mlon))
        self.assertTrue(np.all(lat == mlat))
        self.assertTrue(np.all(r == mr))
        self.assertTrue(np.all(fields["t"] == model.get_field("t")))
        self.assertTrue(np.all(fields["u_xyz"] == model.get_field("u_xyz")))
        self.assertTrue(np.all(fields["c_hist"] == model.get_field("c_hist")))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        sys.stderr.write(
            f"""Usage: python {os.path.basename(__file__)} base_file_path nlayers npts

Positional arguments:
    base_file_path : The full path and start of the name to which
                     to write several TERRA model NetCDF files
    nlayers        : Number of layers in model
    npts           : Total number of lateral points in model
"""
        )
        sys.exit(1)

    filebase = sys.argv[1]
    nlayers = int(sys.argv[2])
    npts = int(sys.argv[3])

    lon, lat, r, fields, c_hist_names = random_model(npts, nlayers)

    filenames = write_seismic_netcdf_files(
        filebase, lon, lat, r, fields, c_hist_names=c_hist_names
    )

    print(f"Wrote files: {filenames}")
