import numpy as np
import unittest

from terratools.geographic import (
    angular_distance,
    angular_step,
    azimuth,
    cart2geog,
    geog2cart,
    spherical_triangle_area,
)


def assert_tuples_close(testcase, tup1, tup2, atol=1e-7):
    """Assert that all elements of the two tuples tup1 and tup2
    are within atol of each other."""
    testcase.assertEqual(len(tup1), len(tup2))
    testcase.assertEqual(type(tup1), type(tup2))
    for v1, v2 in zip(tup1, tup2):
        testcase.assertAlmostEqual(v1, v2, delta=atol)


def assert_tuples_equal(testcase, tup1, tup2):
    """Assert that two tuples are exactly equal"""
    assert_tuples_close(testcase, tup1, tup2, atol=0)


def assert_array_equals_scalar_vec3(testcase, func, args1, args2, args3, atol=1e-7):
    """Assert that the same results are returned either when calling
    func on vector arguments, or when calling on scalar arguments
    many times over, when func takes three arguments and returns
    three values.

    Note that there are sometimes differences between the answers
    around the last significant digit.  Perhaps this is due to
    FMA or fastmath, but whatever the case, we have to deal with it."""
    vouts1, vouts2, vouts3 = func(args1, args2, args3)
    for arg1, vout1, arg2, vout2, arg3, vout3 in zip(
        args1, vouts1, args2, vouts2, args3, vouts3
    ):
        out1, out2, out3 = func(arg1, arg2, arg3)
        assert_tuples_close(
            testcase, (vout1, vout2, vout3), (out1, out2, out3), atol=1e-7
        )


def assert_array_equals_scalar_nargs(testcase, func, args, atol=1e-7):
    """
    Assert that the same results are returned either when calling func
    on vector arguments, or when calling on scalar arguments many times
    over, when func returns a single value.
    """
    vouts = func(*args)
    outs = [func(*args_) for args_ in zip(*args)]

    for vout, out in zip(vouts, outs):
        testcase.assertAlmostEqual(vout, out, delta=atol)


class TestGeog2Cart(unittest.TestCase):
    def test_geog2cart_negative_radius(self):
        with self.assertRaises(ValueError):
            geog2cart(1, 2, -3)

    def test_geog2cart_degrees(self):
        assert_tuples_close(self, geog2cart(0, 0, 1), (1.0, 0.0, 0.0))
        assert_tuples_close(self, geog2cart(90, 0, 10), (0.0, 10.0, 0.0))
        assert_tuples_close(self, geog2cart(0, 90, 100), (0.0, 0.0, 100.0))
        assert_tuples_close(self, geog2cart(0, 0, 1), (1.0, 0.0, 0.0))
        assert_tuples_close(self, geog2cart(-90, 0, 10), (0.0, -10.0, 0.0))
        assert_tuples_close(self, geog2cart(0, -90, 100), (0.0, 0.0, -100.0))

    def test_geog2cart_radians(self):
        assert_tuples_close(self, geog2cart(0, 0, 1, radians=True), (1.0, 0.0, 0.0))
        assert_tuples_close(
            self, geog2cart(np.pi / 2, 0, 10, radians=True), (0.0, 10.0, 0.0)
        )
        assert_tuples_close(
            self, geog2cart(0, np.pi / 2, 100, radians=True), (0.0, 0.0, 100.0)
        )
        assert_tuples_close(self, geog2cart(0, 0, 1, radians=True), (1.0, 0.0, 0.0))
        assert_tuples_close(
            self, geog2cart(-np.pi / 2, 0, 10, radians=True), (0.0, -10.0, 0.0)
        )
        assert_tuples_close(
            self, geog2cart(0, -np.pi / 2, 100, radians=True), (0.0, 0.0, -100.0)
        )

    def test_geog2cart_r0(self):
        assert_tuples_equal(self, geog2cart(30, -6, 0), (0, 0, 0))

    def test_geog2cart_arrays(self):
        n = 100
        lon = 360 * np.random.rand(n)
        lat = 180 * (np.random.rand(n) - 0.5)
        r = 10000 * np.random.rand(n)
        assert_array_equals_scalar_vec3(self, geog2cart, lon, lat, r)


class TestCart2Geog(unittest.TestCase):
    def test_cart2geog_degrees(self):
        assert_tuples_close(self, cart2geog(1, 0, 0), (0, 0, 1))
        assert_tuples_close(self, cart2geog(0, 10, 0), (90, 0, 10))
        assert_tuples_close(self, cart2geog(0, 0, 100), (0, 90, 100))
        assert_tuples_close(self, cart2geog(-1, 0, 0), (180, 0, 1))
        assert_tuples_close(self, cart2geog(0, -10, 0), (-90, 0, 10))
        assert_tuples_close(self, cart2geog(0, 0, -100), (0, -90, 100))

    def test_cart2geog_radians(self):
        assert_tuples_close(self, cart2geog(1, 0, 0, radians=True), (0, 0, 1))
        assert_tuples_close(self, cart2geog(0, 10, 0, radians=True), (np.pi / 2, 0, 10))
        assert_tuples_close(
            self, cart2geog(0, 0, 100, radians=True), (0, np.pi / 2, 100)
        )
        assert_tuples_close(self, cart2geog(-1, 0, 0, radians=True), (np.pi, 0, 1))
        assert_tuples_close(
            self, cart2geog(0, -10, 0, radians=True), (-np.pi / 2, 0, 10)
        )
        assert_tuples_close(
            self, cart2geog(0, 0, -100, radians=True), (0, -np.pi / 2, 100)
        )

    def test_cart2geog_r0(self):
        assert_tuples_equal(self, cart2geog(0, 0, 0), (0, 0, 0))

    def test_cart2geog_arrays(self):
        n = 100
        max_r = 10000
        x = max_r * np.random.rand(n)
        y = max_r * np.random.rand(n)
        z = max_r * np.random.rand(n)
        assert_array_equals_scalar_vec3(self, cart2geog, x, y, z)


class TestAngularDistance(unittest.TestCase):
    def test_angular_distance_degrees(self):
        self.assertAlmostEqual(
            angular_distance(0, 0, 0, 10), np.radians(10), delta=1e-7
        )
        self.assertAlmostEqual(
            angular_distance(0, 10, 0, 0), np.radians(10), delta=1e-7
        )
        self.assertAlmostEqual(angular_distance(0, 90, 0, -90), np.pi, delta=1e-7)

    def test_angular_distance_radians(self):
        self.assertAlmostEqual(
            angular_distance(0, 0, 0, 0.1, radians=True), 0.1, delta=1e-7
        )
        self.assertAlmostEqual(
            angular_distance(0, 0.2, 0, 0, radians=True), 0.2, delta=1e-7
        )
        self.assertAlmostEqual(
            angular_distance(0, np.pi / 2, 0, -np.pi / 2, radians=True),
            np.pi,
            delta=1e-7,
        )

    def test_angular_distance_arrays(self):
        n = 100
        lon1 = 360 * np.random.rand(n)
        lat1 = 180 * (np.random.rand(n) - 0.5)
        lon2 = 360 * np.random.rand(n)
        lat2 = 180 * (np.random.rand(n) - 0.5)
        assert_array_equals_scalar_nargs(
            self, angular_distance, (lon1, lat1, lon2, lat2)
        )


class TestAzimuth(unittest.TestCase):
    def test_azimuth_degrees(self):
        self.assertAlmostEqual(azimuth(0, 0, 0, 1), 0)
        self.assertAlmostEqual(azimuth(0, 0, 1, 0), 90)
        self.assertAlmostEqual(azimuth(0, 0, 0, -1), 180)
        self.assertAlmostEqual(azimuth(0, 0, -1, 0), -90)

    def test_azimuth_radians(self):
        self.assertAlmostEqual(azimuth(0, 0, 0, 0.1, radians=True), 0, delta=1e-7)
        self.assertAlmostEqual(
            azimuth(0, 0, 0.1, 0, radians=True), np.pi / 2, delta=1e-7
        )
        self.assertAlmostEqual(azimuth(0, 0, 0, -0.1, radians=True), np.pi, delta=1e-7)
        self.assertAlmostEqual(
            azimuth(0, 0, -0.1, 0, radians=True), -np.pi / 2, delta=1e-7
        )


class TestSphericalTriangleArea(unittest.TestCase):
    def test_vectorised(self):
        lon1 = np.array([0.0, 1.0, 0.0])
        lat1 = np.array([0.0, 0.0, 1.0])
        lon2 = np.array([1.0, 2.0, 1.0])
        lat2 = np.array([1.0, 1.0, 2.0])
        lon3 = np.array([2.0, 3.0, 2.0])
        lat3 = np.array([1.0, 1.0, 2.0])

        areas = spherical_triangle_area(lon1, lat1, lon2, lat2, lon3, lat3)
        self.assertAlmostEqual(areas[0], areas[1], delta=1.0e-7)
        self.assertTrue(areas[1] != areas[2])


class TestAngularSte(unittest.TestCase):
    def test_angular_step_degrees(self):
        lon, lat = angular_step(0, 0, 0, 20)
        self.assertAlmostEqual(lon, 0)
        self.assertAlmostEqual(lat, 20)

        lon, lat = angular_step(0, 0, -90, 40)
        self.assertAlmostEqual(lon, -40)
        self.assertAlmostEqual(lat, 0)

        lon, lat = angular_step(45, 45, -60, 180)
        self.assertAlmostEqual(lon, -135)
        self.assertAlmostEqual(lat, -45)

    def test_angular_step_radians(self):
        lon, lat = angular_step(0, 0, np.pi / 2, np.pi / 3, radians=True)
        self.assertAlmostEqual(lon, np.pi / 3)
        self.assertAlmostEqual(lat, 0)

        lon, lat = angular_step(-1, 0.5, 0.4, np.pi, radians=True)
        self.assertAlmostEqual(lon, np.pi - 1)
        self.assertAlmostEqual(lat, -0.5)
