import unittest

import numpy as np
import os

from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib.figure import Figure

from terratools import terra_model
from terratools.terra_model import TerraModel

# Tolerance for match of floating point numbers, given that TerraModel
# coordinates and values have a defined precision
coord_tol = np.finfo(terra_model.COORDINATE_TYPE).eps
value_tol = np.finfo(terra_model.VALUE_TYPE).eps

# Helper functions for the tests
def dummy_model(nlayers=3, npts=4, with_fields=False, **kwargs):
    lon, lat, r = random_coordinates(nlayers, npts)
    if with_fields:
        fields = {
            "t": random_field(nlayers, npts),
            "u_xyz": random_field(nlayers, npts, 3),
        }
    else:
        fields = {}
    return TerraModel(lon, lat, r, fields=fields, **kwargs)


def random_coordinates(nlayers, npts):
    """Return a set of random coordinates (satisfying the constraints)
    which can be passed to construct a TerraModel.
    FIXME: In theory there may be duplicate points as they are random."""
    rmin, rmax = np.sort(7000 * np.random.rand(2))
    r = np.sort(np.random.uniform(rmin, rmax, nlayers))
    lon = 360 * (np.random.rand(npts) - 0.5)
    lat = 180 * (np.random.rand(npts) - 0.5)
    return lon, lat, r


def random_field(nlayers, npts, ncomps=None):
    if ncomps is None:
        return np.random.rand(nlayers, npts)
    else:
        return np.random.rand(nlayers, npts, ncomps)


def read_test_lateral_points():
    """Read a subset of points on the TERRA grid in a box from
    -2° to 2 ° in longitude and latitude"""
    dir = os.path.dirname(__file__)
    file = os.path.join(dir, "data", "TERRA_grid_lon-2_2_lat-2_2.txt")
    data = np.loadtxt(file, dtype=np.float32)
    lon = data[:, 0]
    lat = data[:, 1]
    return lon, lat


def fields_are_equal(field1, field2):
    return np.allclose(field1, field2, atol=value_tol)


def coords_are_equal(coords1, coords2):
    return np.allclose(coords1, coords2, atol=coord_tol)


class TestTerraModelHelpers(unittest.TestCase):
    """Tests for non-class methods"""

    def test_is_valid_field_name(self):
        """Test for validity of a field name"""
        self.assertFalse(terra_model._is_valid_field_name("incorrect field name"))
        self.assertTrue(terra_model._is_valid_field_name("t"))

    def test_variable_name_from_field(self):
        """Translation of field name to NetCDF variable name(s)"""

        self.assertEqual(terra_model._variable_names_from_field("t"), ("temperature",))
        self.assertEqual(
            terra_model._variable_names_from_field("u_xyz"),
            ("velocity_x", "velocity_y", "velocity_z"),
        )
        self.assertEqual(
            terra_model._variable_names_from_field("c_hist"), ("composition_fractions",)
        )

        with self.assertRaises(KeyError):
            terra_model._variable_names_from_field("incorrect field name")

    def test_field_name_from_variable(self):
        """Translation of NetCDF variable name to field name"""
        self.assertEqual(terra_model._field_name_from_variable("temperature"), "t")
        self.assertEqual(terra_model._field_name_from_variable("absent field"), None)

    def test_check_field_name(self):
        self.assertEqual(terra_model._check_field_name("vp"), None)
        with self.assertRaises(terra_model.FieldNameError):
            terra_model._check_field_name("incorrect field name")

    def test_is_scalar_field(self):
        self.assertTrue(terra_model._is_scalar_field("c"))
        self.assertFalse(terra_model._is_scalar_field("c_hist"))

    def test_is_vector_field(self):
        self.assertTrue(terra_model._is_vector_field("u_xyz"))
        self.assertFalse(terra_model._is_vector_field("t"))

    def test_expected_vector_field_ncomps(self):
        self.assertEqual(terra_model._expected_vector_field_ncomps("u_xyz"), 3)
        self.assertEqual(terra_model._expected_vector_field_ncomps("u_geog"), 3)
        self.assertEqual(terra_model._expected_vector_field_ncomps("c_hist"), None)

    def test_compositions_sum_to_one(self):
        comps = np.zeros((4, 3, 2))
        self.assertFalse(terra_model._compositions_sum_to_one(comps))
        comps[:, :, -1] = 1
        self.assertTrue(terra_model._compositions_sum_to_one(comps))
        comps[:, :, -1] = 0.9
        self.assertTrue(terra_model._compositions_sum_to_one(comps, atol=0.2))
        self.assertFalse(terra_model._compositions_sum_to_one(comps, atol=1.0e-7))
        self.assertFalse(terra_model._compositions_sum_to_one(comps))


class TestTerraModelConstruction(unittest.TestCase):
    """Tests for construction and validation of fields"""

    def test_invalid_field_dimensions(self):
        npts = 3
        nlayers = 2
        scalar_field = random_field(nlayers, npts)
        vector_field = random_field(nlayers, npts, 3)
        fields = {"t": scalar_field, "u_xyz": vector_field}

        with self.assertRaises(terra_model.FieldDimensionError):
            lon, lat, r = random_coordinates(nlayers, npts + 1)
            TerraModel(lon, lat, r, fields=fields)

        with self.assertRaises(terra_model.FieldDimensionError):
            lon, lat, r = random_coordinates(nlayers, npts - 1)
            TerraModel(lon, lat, r, fields=fields)

        with self.assertRaises(terra_model.FieldDimensionError):
            lon, lat, r = random_coordinates(nlayers + 1, npts)
            TerraModel(lon, lat, r, fields=fields)

        with self.assertRaises(terra_model.FieldDimensionError):
            lon, lat, r = random_coordinates(nlayers - 1, npts)
            TerraModel(lon, lat, r, fields=fields)

    def test_invalid_field_name(self):
        npts = 10
        nlayers = 3
        field = random_field(nlayers, npts)
        lon, lat, r = random_coordinates(nlayers, npts)
        with self.assertRaises(terra_model.FieldNameError):
            TerraModel(lon, lat, r, fields={"incorrect field name": field})

    def test_invalid_ncomps(self):
        nlayers = 3
        npts = 2
        u_xyz = random_field(nlayers, npts, 2)
        lon, lat, r = random_coordinates(nlayers, npts)
        with self.assertRaises(terra_model.FieldDimensionError):
            TerraModel(lon, lat, r, fields={"u_xyz": u_xyz})

    def test_radii_not_motonic(self):
        with self.assertRaises(ValueError):
            TerraModel([1], [1], [1, 3, 2])

    def test_radii_not_increasing(self):
        with self.assertRaises(ValueError):
            TerraModel([1], [1], [3, 2, 1])

    def test_lon_lat_not_same_length(self):
        with self.assertRaises(ValueError):
            TerraModel([1], [2, 3], [1, 2, 3])

    def test_composition_proportions_do_not_sum_to_one(self):
        nlayers = 3
        npts = 2
        ncomps = 4
        lon, lat, r = random_coordinates(nlayers, npts)
        c_hist_field = random_field(nlayers, npts, ncomps)
        with self.assertRaises(ValueError):
            TerraModel(lon, lat, r, fields={"c_hist": c_hist_field})

    def test_surface_radius_too_small(self):
        lon, lat, r = random_coordinates(3, 2)
        with self.assertRaises(ValueError):
            TerraModel(lon, lat, r, surface_radius=r[-1] - 1)

    def test_construction(self):
        """Ensure the things we pass in are put in the right place"""
        nlayers = 3
        npts = 10
        lon, lat, r = random_coordinates(nlayers, npts)
        scalar_field_names = ("t", "c", "vp", "vs", "density", "p")
        scalar_fields = [random_field(nlayers, npts) for _ in scalar_field_names]
        u_field = random_field(nlayers, npts, 3)
        c_hist_field = random_field(nlayers, npts, 2)
        # Ensure each composition histogram sums to unity
        c_hist_field[:, :, 1] = 1 - c_hist_field[:, :, 0]
        fields = {name: field for name, field in zip(scalar_field_names, scalar_fields)}
        fields["u_xyz"] = u_field
        fields["u_geog"] = u_field
        fields["c_hist"] = c_hist_field
        c_hist_names = ["A", "B"]
        c_hist_values = [1, 2]

        model = TerraModel(
            lon,
            lat,
            r,
            fields=fields,
            c_histogram_names=c_hist_names,
            c_histogram_values=c_hist_values,
        )

        _lon, _lat = model.get_lateral_points()
        self.assertTrue(coords_are_equal(lon, _lon))
        self.assertTrue(coords_are_equal(lat, _lat))

        self.assertTrue(coords_are_equal(model.get_radii(), r))

        for (field_name, field) in zip(scalar_field_names, scalar_fields):
            self.assertTrue(fields_are_equal(model.get_field(field_name), field))

        self.assertTrue(fields_are_equal(model.get_field("c_hist"), c_hist_field))

        for field in ("u_xyz", "u_geog"):
            self.assertTrue(fields_are_equal(model.get_field(field), u_field))

        self.assertTrue(fields_are_equal(model.get_field("c_hist"), c_hist_field))
        self.assertEqual(model.number_of_compositions(), 2)
        self.assertEqual(model.get_composition_names(), ["A", "B"])
        self.assertEqual(model.get_composition_values(), [1, 2])

        # Use set because we don't need to enforce that the fields
        # are in the same order
        self.assertEqual(set(model.field_names()), set(fields.keys()))

        for field in (*scalar_field_names, "u_geog", "u_xyz", "c_hist"):
            self.assertTrue(model.has_field(field))
        self.assertFalse(model.has_field("vs_an"))


class TestTerraModelGetters(unittest.TestCase):
    """Tests for getters"""

    def test_invalid_field_name(self):
        """Check that an error is thrown when asking for an invalid field"""
        model = dummy_model()
        model.new_field("t")
        with self.assertRaises(terra_model.FieldNameError):
            model.evaluate(0, 0, 4000, "incorrect field name")

    def test_missing_field(self):
        model = dummy_model()
        model.new_field("t")
        with self.assertRaises(terra_model.NoFieldError):
            model.evaluate(0, 0, 4000, "u_xyz")

    def test_get_field(self):
        model = dummy_model(with_fields=True)
        self.assertIs(model.get_field("t"), model._fields["t"])

        temp = model.get_field("t")
        temp[0, 0] = 1
        self.assertTrue(np.all(model.get_field("t") == temp))

    def test_get_radii(self):
        r = [1, 2, 3]
        model = TerraModel([1, 2], [3, 4], r)
        self.assertIs(model.get_radii(), model._radius)

    def test_get_lateral_points(self):
        lon, lat, r = random_coordinates(3, 4)
        model = TerraModel(lon, lat, r)
        _lon, _lat = model.get_lateral_points()
        self.assertIs(_lon, model._lon)
        self.assertIs(_lat, model._lat)

    def test_get_composition_names(self):
        model = dummy_model(c_histogram_names=["A", "B"])
        model.new_field("c_hist", 2)
        self.assertEqual(model.get_composition_names(), ["A", "B"])

    def test_nearest_layer(self):
        model = TerraModel([1], [1], [5, 7, 9])
        self.assertCountEqual(model.nearest_layer(4), (0, 5.0), 2)
        self.assertCountEqual(model.nearest_layer(5.1), (0, 5.0), 2)
        self.assertCountEqual(model.nearest_layer(6.5), (1, 7.0), 2)
        self.assertCountEqual(model.nearest_layer(7.0), (1, 7.0), 2)

    def test_get_composition_values(self):
        model = dummy_model(c_histogram_names=["A", "B"], c_histogram_values=[1, 2])
        model.new_field("c_hist", 2)
        self.assertEqual(model.get_composition_values(), [1, 2])


class TestTerraModelNewField(unittest.TestCase):
    def test_wrong_ncomps(self):
        model = dummy_model()
        with self.assertRaises(ValueError):
            model.new_field("t", 1)
        with self.assertRaises(ValueError):
            model.new_field("u_xyz", 4)

    def test_no_ncomps(self):
        model = dummy_model()
        with self.assertRaises(ValueError):
            model.new_field("c_hist")

    def test_new_field(self):
        model = dummy_model()
        nlayers = len(model.get_radii())
        npts = len(model.get_lateral_points()[0])

        self.assertFalse(model.has_field("t"))
        self.assertFalse(model.has_field("u_xyz"))
        self.assertFalse(model.has_field("c_hist"))

        model.new_field("t")
        self.assertTrue(model.has_field("t"))
        self.assertTrue(
            fields_are_equal(model.get_field("t"), np.zeros((nlayers, npts)))
        )

        model.new_field("u_xyz")
        self.assertTrue(model.has_field("u_xyz"))
        self.assertTrue(
            fields_are_equal(model.get_field("u_xyz"), np.zeros((nlayers, npts, 3)))
        )

        model.new_field("c_hist", 2)
        self.assertTrue(model.has_field("c_hist"))
        self.assertTrue(
            fields_are_equal(model.get_field("c_hist"), np.zeros((nlayers, npts, 2)))
        )


class TerraModelDepthConversion(unittest.TestCase):
    def test_to_depth(self):
        model = dummy_model(surface_radius=10000)
        self.assertEqual(model.to_depth(2500), 7500)
        model = dummy_model()
        surface_radius = model.get_radii()[-1]
        self.assertAlmostEqual(model.to_depth(1), surface_radius - 1, 3)

    def test_to_radius(self):
        model = dummy_model(surface_radius=10000)
        self.assertEqual(model.to_radius(2500), 7500)
        model = dummy_model()
        surface_radius = model.get_radii()[-1]
        self.assertAlmostEqual(model.to_radius(1), surface_radius - 1, 3)


class TestTerraModelRepr(unittest.TestCase):
    def test_repr(self):
        npts = 3
        nlayers = 3
        lon = [1, 2, 3]
        lat = [10, 20, 30]
        r = [1000, 1999, 2000]
        t_field = random_field(nlayers, npts)
        c_hist_field = random_field(nlayers, npts, 2)
        # Require compositions sum to 1
        c_hist_field[:, :, 1] = 1 - c_hist_field[:, :, 0]
        cnames = ["a", "b"]
        model = TerraModel(
            lon,
            lat,
            r,
            fields={"t": t_field, "c_hist": c_hist_field},
            c_histogram_names=cnames,
        )

        self.assertEqual(
            model.__repr__(),
            """TerraModel:
           number of radii: 3
             radius limits: (1000.0, 2000.0)
  number of lateral points: 3
                    fields: ['t', 'c_hist']
         composition names: ['a', 'b']
        composition values: None""",
        )


class TestTerraModelNearestIndex(unittest.TestCase):
    def test_nearest_index(self):
        lon = [20, 22, 0.1, 25]
        lat = [20, 22, 0.1, 24]
        r = [10, 20]
        model = TerraModel(lon, lat, r)
        self.assertEqual(model.nearest_index(0, 0), 2)
        self.assertCountEqual(model.nearest_index([0, 0.1], [0, 0.1]), [2, 2], 2)


class TestTerraModelNearestIndices(unittest.TestCase):
    def test_nearest_indices_zero_n(self):
        with self.assertRaises(ValueError):
            dummy_model().nearest_indices(0, 0, 0)

    def test_nearest_indices(self):
        lon = [20, 22, 0.1, 25]
        lat = [20, 22, 0.1, 24]
        r = [10, 20]
        model = TerraModel(lon, lat, r)
        self.assertCountEqual(model.nearest_indices(0, 0, n=3), [2, 0, 1])

        indices = model.nearest_indices([0, 0.1], [0, 0.1], n=4)
        for inds in indices:
            self.assertCountEqual(inds, [2, 0, 1, 3], 4)


class TestTerraModelNearestNeighbors(unittest.TestCase):
    def test_nearest_neighbors_zero_n(self):
        with self.assertRaises(ValueError):
            dummy_model().nearest_neighbors(0, 0, 0)

    def test_nearest_neighbors(self):
        lon = [0, 0, 0, 0]
        lat_radians = [0, 0.1, -0.2, 0.3]
        lat = np.degrees(lat_radians)
        r = [1]
        model = TerraModel(lon, lat, r)
        indices, distances = model.nearest_neighbors(0, 0, 4)
        self.assertTrue(np.allclose(distances, [0, 0.1, 0.2, 0.3], atol=1e-7))
        self.assertTrue(np.allclose(indices, [0, 1, 2, 3], atol=1e-7))

        indices, distances = model.nearest_neighbors([0, 0], [0, np.degrees(0.1)], n=4)
        self.assertTrue(np.allclose(distances[0], [0, 0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(indices[0], [0, 1, 2, 3]))
        self.assertTrue(np.allclose(distances[1], [0.0, 0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(indices[1], [1, 0, 3, 2]))


class TestTerraModelEvaluate(unittest.TestCase):
    def test_wrong_method(self):
        with self.assertRaises(ValueError):
            dummy_model(with_fields=True).evaluate(
                0, 0, 1, "t", method="unsupported method"
            )

    def test_evaluate_depth(self):
        model = dummy_model(with_fields=True)
        radii = model.get_radii()
        mid_radius = np.min(radii) + (np.max(radii) - np.min(radii)) / 2
        surface_radius = np.max(radii)
        mid_depth = surface_radius - mid_radius
        self.assertEqual(
            model.evaluate(0, 0, mid_radius, "t"),
            model.evaluate(0, 0, mid_depth, "t", depth=True),
        )
        self.assertCountEqual(
            model.evaluate(0, 0, mid_radius, "u_xyz"),
            model.evaluate(0, 0, mid_depth, "u_xyz", depth=True),
            3,
        )

    def test_evaluate_radii_interpolation(self):
        model = TerraModel(lon=[-1, 1, 0], lat=[1, 1, -1], r=[1, 2])
        model.new_field("t")
        temp = model.get_field("t")
        temp[0, :] = 10
        temp[1, :] = 20
        self.assertAlmostEqual(model.evaluate(0, 0, 1.1, "t"), 11)
        self.assertAlmostEqual(model.evaluate(0, 0, 0.9, "t", depth=True), 11)

    def test_evaluate_lateral_interpolation(self):
        def test_func(lon, lat):
            return lat

        lon, lat = read_test_lateral_points()
        radii = [3480, 6370]
        model = TerraModel(lon, lat, radii)
        model.new_field("t")
        temp = model.get_field("t")
        # Make temp follow an analytical function
        test_r = radii[0]
        for i in range(len(radii)):
            temp[i, :] = test_func(lon, lat)

        test_lon, test_lat = 4 * (np.random.rand(2) - 0.5)
        self.assertAlmostEqual(
            model.evaluate(test_lon, test_lat, test_r, "t"),
            test_func(test_lon, test_lat),
            delta=1,
        )

    def test_evaluate_nearest(self):
        lon = [0, 1, 2]
        lat = [0, 1, 0]
        r = [1]
        model = TerraModel(lon, lat, r)
        u_xyz = model.new_field("u_xyz")
        u_xyz[0, 0, :] = [1, 2, 3]
        u_xyz[0, 1, :] = [10, 20, 30]
        u_xyz[0, 2, :] = [-2, -4, 10]
        self.assertTrue(
            np.all(
                model.evaluate(1.9, 0.1, 1, "u_xyz", method="nearest") == [-2, -4, 10]
            )
        )


class TestNearestIndex(unittest.TestCase):
    def test_below(self):
        self.assertEqual(terra_model._nearest_index(0.9, [1, 2, 3]), 0)

    def test_above(self):
        self.assertEqual(terra_model._nearest_index(3.1, [1, 2, 3]), 2)

    def test_between(self):
        self.assertEqual(terra_model._nearest_index(2.1, [1, 2, 3]), 1)
        self.assertEqual(terra_model._nearest_index(2.6, [1, 2, 3]), 2)


class TestBoundingIndices(unittest.TestCase):
    def test_below(self):
        self.assertCountEqual(
            terra_model._bounding_indices(-5, [-4, -3, -2]), (0, 0), 2
        )

    def test_above(self):
        self.assertCountEqual(terra_model._bounding_indices(0, [-4, -3, -2]), (2, 2), 2)

    def test_between(self):
        self.assertCountEqual(
            terra_model._bounding_indices(-2.1, [-4, -3, -2]), (1, 2), 2
        )

    def test_equal(self):
        self.assertCountEqual(
            terra_model._bounding_indices(-3, [-4, -3, -2]), (1, 1), 2
        )


class TestPlotLayer(unittest.TestCase):
    def test_errors(self):
        model = dummy_model(with_fields=True)

        with self.assertRaises(ValueError):
            model.plot_layer("t")

        with self.assertRaises(ValueError):
            model.plot_layer("t", index=-1)
        with self.assertRaises(ValueError):
            model.plot_layer("t", index=len(model.get_radii()) + 1)

    def test_plot_layer_radius(self):
        model = dummy_model(with_fields=True)
        fig, ax = model.plot_layer("t", 4000, show=False)
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, GeoAxesSubplot)

    def test_plot_layer_depth(self):
        model = dummy_model(with_fields=True)
        fig, ax = model.plot_layer("t", 100, depth=True, show=False)
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, GeoAxesSubplot)

    def test_plot_layer_index(self):
        model = dummy_model(with_fields=True)
        fig, ax = model.plot_layer("t", index=2, depth=True, show=False)
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, GeoAxesSubplot)


if __name__ == "__main__":
    unittest.main()
