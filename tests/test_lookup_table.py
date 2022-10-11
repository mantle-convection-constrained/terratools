from cgitb import lookup
import unittest
import numpy as np
from terratools.lookup_tables import SeismicLookupTable, _harmonic_mean, _check_bounds
import pathlib

TESTDATA_PATH = pathlib.Path(__file__).parent.joinpath("data", "test_lookup_table.txt")


class TestLookup(unittest.TestCase):
    def setUp(self):
        self.lookup_tab_path = "./tests/data/test_lookup_table.txt"
        self.tab = SeismicLookupTable(TESTDATA_PATH)

    def test_read_file(self):

        # test table has been read in correctly by comparing the
        # size of the array read in.
        self.assertEqual(self.tab.table.shape, (121, 10), "file not read in correctly")

        self.assertEqual(self.tab.table[0, 0], -50.0, "file not read in correctly")

    def test_interpolate_point(self):

        p_test = 15
        t_test = 15

        self.assertAlmostEqual(
            self.tab.interp_points(p_test, t_test, "Vp"),
            np.array([15]),
            msg="interpolation for single point failed",
        )

    def test_interpolate_grid(self):

        t_test = [4, 5, 6]
        p_test = 10
        outgrid = self.tab.interp_grid(p_test, t_test, "Vp")

        self.assertEqual(
            int(outgrid[0]), 4, msg="interpolation for grid of points failed"
        )
        self.assertEqual(
            int(outgrid[1]), 5, msg="interpolation for grid of points failed"
        )
        self.assertEqual(
            int(outgrid[2]), 6, msg="interpolation for grid of points failed"
        )


    def test_harmonic_mean_1D(self):
        test_data = np.array([1,5])
        test_weights = np.array([1,2])

        hmean = _harmonic_mean(test_data, test_weights)

        self.assertEqual(hmean, 15/7,
                         msg='harmonic mean with 1D arrays failed')

    def test_harmonic_mean_2D(self):
        test_data = np.ones((2,3,3))
        test_data[1] *= 5
        test_weights = np.array([1,2])

        hmean = _harmonic_mean(test_data, test_weights)

        np.testing.assert_array_equal(hmean, np.ones((3,3)) * 15/7,
                         err_msg='harmonic mean with 2D arrays failed')

    def test_check_bounds_scalar(self):
        Ps = np.arange(-50,51,1)

        # for when less than range
        corrected_value = _check_bounds(input=-100, check=Ps)
        self.assertEqual(corrected_value, -50,
                         msg = 'check bounds failed when input is less than range')

        # for when greater than range
        corrected_value = _check_bounds(input=100, check=Ps)
        self.assertEqual(corrected_value, 50, msg = 'check bounds failed when input is greater than range')

        # for when within range
        corrected_value = _check_bounds(input=-25.5, check=Ps)
        self.assertEqual(corrected_value, -25.5, msg = 'check bounds failed when input is within range')

    def test_check_bounds_array(self):
        Ps = np.arange(-50,51,1)
        values = np.array([-100,100,25.5])

        corrected_values = _check_bounds(input=values, check=Ps)
        np.testing.assert_array_equal(corrected_values, np.array([-50,50,25.5]),
                                      err_msg='check bounds failed for array of inputs')


if __name__ == '__main__':
    unittest.main()
