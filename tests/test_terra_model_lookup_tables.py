"""
Test the reading and use of seismic lookup tables for the
terra_model module.
"""

import unittest

import numpy as np

from terratools import terra_model
from terratools.terra_model import TerraModel


def random_model(npts, nlayers):
    """
    Return a set of values which can be used to construct a TerraModel.
    """
    coordinate_type = terra_model.COORDINATE_TYPE
    value_type = terra_model.VALUE_TYPE

    lon = 360 * np.random.rand(npts).astype(coordinate_type)
    lat = 180 * (np.random.rand(npts).astype(coordinate_type) - 0.5)
    r = 3480 + (6370 - 3480) * np.sort(np.random.rand(nlayers).astype(coordinate_type))

    fields = {
        "t": np.random.rand(nlayers, npts).astype(value_type),
        "u_xyz": np.random.rand(nlayers, npts, 3).astype(value_type),
        "c_hist": np.random.rand(nlayers, npts, 3).astype(value_type),
    }
    c_hist_names = ["harzburgite", "lherzolite"]

    return lon, lat, r, fields, c_hist_names


class TestTerraModelLookupTablesConstruction(unittest.TestCase):
    """
    Tests for constructing TerraModels with lookup tables
    """

    def test_wrong_lookup_table_keys(self):
        lon, lat, r, fields, c_hist_names = random_model(3, 2)
        lookup_tables = {
            "harzburgite": "file2",
            "lherzolite": "file2",
            "random_other_name": "file3",
        }
        with self.assertRaises(ValueError):
            TerraModel(
                lon,
                lat,
                r,
                fields=fields,
                c_histogram_names=c_hist_names,
                lookup_tables=lookup_tables,
            )

    def test_no_c_hist_names(self):
        lon, lat, r, fields, c_hist_names = random_model(3, 2)
        lookup_tables = {"harzburgite": "file2", "lherzolite": "file2"}
        with self.assertRaises(ValueError):
            TerraModel(lon, lat, r, fields=fields, lookup_tables=lookup_tables)
