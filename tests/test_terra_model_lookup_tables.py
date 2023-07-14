"""
Test the reading and use of seismic lookup tables for the
terra_model module.
"""

import unittest

import numpy as np
import os

from terratools import terra_model, lookup_tables
from terratools.terra_model import TerraModel
from terratools.lookup_tables import TABLE_FIELDS, SeismicLookupTable, MultiTables


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
    # Make sure composition proportions sum to one
    fields["c_hist"][:, :, 2] = (
        1 - fields["c_hist"][:, :, 0] - fields["c_hist"][:, :, 1]
    )
    c_hist_names = ["harzburgite", "lherzolite"]

    return lon, lat, r, fields, c_hist_names


def test_data_path(file):
    """
    Return the path to a file with name ``file`` in the test data directory.
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, "data", file)


def synthetic_lookup_table(
    func, pressures=np.arange(0, 140e9, 10e9), temperatures=np.arange(0, 5000, 500)
):
    """
    Create a synthetic lookup table across a set of pressures and temperatures,
    where ``func(field, pressure, temperature)`` is evaluated at each point
    and takes pressure in Pa and temperature in K.

    E.g.
        >>> synthetic_lookup_table(lambda field, p, t: p*t)
    """
    n_p = len(pressures)
    n_t = len(temperatures)
    grids = {}
    for field in TABLE_FIELDS:
        grids[field] = np.empty((n_t, n_p))
        for it, temperature in enumerate(temperatures):
            for ip, pressure in enumerate(pressures):
                grids[field][it, ip] = func(field, pressure, temperature)

    return SeismicLookupTable(pressure=pressures, temperature=temperatures, **grids)


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

    def test_wrong_number_lookup_tables(self):
        lon, lat, r, fields, c_hist_names = random_model(3, 2)
        lookup_tables = {"harzburgite": "file"}
        with self.assertRaises(ValueError):
            TerraModel(
                lon,
                lat,
                r,
                fields=fields,
                c_histogram_names=c_hist_names,
                lookup_tables=lookup_tables,
            )

    def test_missing_lookup_table_file(self):
        lon, lat, r, fields, c_hist_names = random_model(4, 3)
        lookup_tables = {
            "harzburgite": test_data_path("test_lookup_table.txt"),
            "lherzolite": "nonexistent_file.txt",
        }
        with self.assertRaises(FileNotFoundError):
            TerraModel(
                lon,
                lat,
                r,
                fields=fields,
                c_histogram_names=c_hist_names,
                lookup_tables=lookup_tables,
            )

    def test_construct_with_paths(self):
        lon, lat, r, fields, c_hist_names = random_model(10, 20)
        lookup_tables = {
            "harzburgite": test_data_path("multi_table_test1.txt"),
            "lherzolite": test_data_path("multi_table_test2.txt"),
        }
        model = TerraModel(
            lon,
            lat,
            r,
            fields=fields,
            lookup_tables=lookup_tables,
            c_histogram_names=c_hist_names,
        )
        self.assertIsInstance(model._lookup_tables, MultiTables)
        self.assertEqual(
            model._lookup_tables._lookup_tables.keys(), lookup_tables.keys()
        )

    def test_construct_with_multitables(self):
        lon, lat, r, fields, c_hist_names = random_model(5, 2)
        lookup_tables = {
            "harzburgite": test_data_path("multi_table_test1.txt"),
            "lherzolite": test_data_path("multi_table_test2.txt"),
        }
        tables = MultiTables(lookup_tables)
        model = TerraModel(
            lon,
            lat,
            r,
            fields=fields,
            lookup_tables=lookup_tables,
            c_histogram_names=c_hist_names,
        )
        self.assertIsInstance(model._lookup_tables, MultiTables)
        self.assertEqual(
            model._lookup_tables._lookup_tables.keys(), lookup_tables.keys()
        )


class TestTerraModelLookupTableInquiry(unittest.TestCase):
    def test_lookup_table_inquiry(self):
        lon, lat, r, fields, c_hist_names = random_model(5, 2)
        model = TerraModel(lon, lat, r, fields=fields, c_histogram_names=c_hist_names)
        self.assertFalse(model.has_lookup_tables())

        lookup_tables = MultiTables(
            {
                name: synthetic_lookup_table(lambda field, p, t: 1)
                for name in c_hist_names
            }
        )

        model.add_lookup_tables(lookup_tables)
        self.assertTrue(model.has_lookup_tables())

        self.assertEqual(model.get_lookup_tables(), lookup_tables)


class TestTerraModelLookupTableInterpolation(unittest.TestCase):
    """
    Tests for using lookup tables to predict values of seismic properties
    """

    def test_evaluate(self):
        lon, lat, r, fields, c_hist_names = random_model(5, 2)
        lookup_tables = {
            # Harzburgite file is all 1s
            "harzburgite": test_data_path("multi_table_test1.txt"),
            # Lherzolite file is all 5s
            "lherzolite": test_data_path("multi_table_test2.txt"),
        }
        tables = MultiTables(lookup_tables)
        model = TerraModel(
            lon,
            lat,
            r,
            fields=fields,
            lookup_tables=lookup_tables,
            c_histogram_names=c_hist_names,
        )
        p = 35
        t = 25
        # All-harzburgite case
        props = {"lherzolite": 0, "harzburgite": 1}
        for prop in TABLE_FIELDS:
            self.assertEqual(model._lookup_tables.evaluate(p, t, props, prop), [1])
        # All-lherzolite case
        props = {"lherzolite": 1, "harzburgite": 0}
        for prop in TABLE_FIELDS:
            self.assertEqual(model._lookup_tables.evaluate(p, t, props, prop), [5])
        # Mixed case
        props = {"lherzolite": 0.25, "harzburgite": 0.75}
        for prop in TABLE_FIELDS:
            self.assertEqual(model._lookup_tables.evaluate(p, t, props, prop), [1.25])
