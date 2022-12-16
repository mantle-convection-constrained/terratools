import unittest
import numpy as np
from terratools import lookup_tables
from terratools.lookup_tables import (
    SeismicLookupTable,
    _harmonic_mean,
    _check_bounds,
    MultiTables,
)
import pathlib

TESTDATA_DIR = pathlib.Path(__file__).parent


class TestLookup(unittest.TestCase):
    def setUp(self):
        self.lookup_tab_path = "./tests/data/test_lookup_table.txt"
        self.tab = SeismicLookupTable(
            TESTDATA_DIR.joinpath("data", "test_lookup_table.txt")
        )
        self.tabs = {
            "tab1": TESTDATA_DIR.joinpath("data", "multi_table_test1.txt"),
            "tab2": TESTDATA_DIR.joinpath("data", "multi_table_test2.txt"),
        }
        self.multitable = MultiTables(self.tabs)

    def test_read_file(self):

        # test table has been read in correctly by comparing the
        # size of the arrays read in.
        self.assertEqual(
            self.tab.fields["vp"][1].shape, (11, 11), "file not read in correctly"
        )

        self.assertEqual(
            self.tab.fields["vp"][1][0, 0], -50.0, "file not read in correctly"
        )

    def test_missing_args(self):
        with self.assertRaises(ValueError):
            SeismicLookupTable(
                pressure=self.tab.pres,
                temperature=self.tab.temp,
                vp=self.tab.fields["vp"][1],
            )

    def test_args_wrong_shape(self):
        nT = len(self.tab.temp)
        nP = len(self.tab.pres)
        vals = np.zeros((nT, nP + 1))
        with self.assertRaises(ValueError):
            SeismicLookupTable(
                pressure=self.tab.pres,
                temperature=self.tab.temp,
                vp=vals,
                vs=vals,
                vp_an=vals,
                vs_an=vals,
                vphi=vals,
                density=vals,
                qs=vals,
                t_sol=vals,
            )

    def test_construct_from_arrays(self):
        nT = len(self.tab.temp)
        nP = len(self.tab.pres)
        vp = 1 * np.ones((nT, nP))
        vs = 2 * np.ones((nT, nP))
        vp_an = 3 * np.ones((nT, nP))
        vs_an = 4 * np.ones((nT, nP))
        vphi = 5 * np.ones((nT, nP))
        density = 6 * np.ones((nT, nP))
        qs = 7 * np.ones((nT, nP))
        t_sol = 8 * np.ones((nT, nP))
        table = SeismicLookupTable(
            pressure=self.tab.pres,
            temperature=self.tab.temp,
            vp=vp,
            vs=vs,
            vp_an=vp_an,
            vs_an=vs_an,
            vphi=vphi,
            density=density,
            qs=qs,
            t_sol=t_sol,
        )

        p = self.tab.pres[nP // 2]
        t = self.tab.temp[nT // 2]

        for i, field in enumerate(lookup_tables.TABLE_FIELDS):
            expected_val = i + 1
            self.assertAlmostEqual(table.interp_points(p, t, field), [expected_val])

    def test_interpolate_point(self):

        p_test = 15
        t_test = 15

        self.assertAlmostEqual(
            self.tab.interp_points(p_test, t_test, "vp"),
            np.array([15]),
            msg="interpolation for single point failed",
        )

    def test_interpolate_point_shape(self):
        p = np.reshape([10], (1, 1))
        t = np.reshape([5, 5], (2, 1))

        self.assertEqual(self.tab.interp_points(p, t, "density").shape, (2, 1))

    def test_interpolate_grid(self):

        t_test = [4, 5, 6]
        p_test = 10
        outgrid = self.tab.interp_grid(p_test, t_test, "vp")

        self.assertEqual(
            int(outgrid[0, 0]), 4, msg="interpolation for grid of points failed"
        )
        self.assertEqual(
            int(outgrid[0, 1]), 5, msg="interpolation for grid of points failed"
        )
        self.assertEqual(
            int(outgrid[0, 2]), 6, msg="interpolation for grid of points failed"
        )

    def test_harmonic_mean_1D(self):
        test_data = np.array([1, 5])
        test_weights = np.array([1, 2])

        hmean = _harmonic_mean(test_data, test_weights)

        self.assertEqual(hmean, 15 / 7, msg="harmonic mean with 1D arrays failed")

    def test_harmonic_mean_2D(self):
        test_data = np.ones((2, 3, 3))
        test_data[1] *= 5
        test_weights = np.array([1, 2])

        hmean = _harmonic_mean(test_data, test_weights)

        np.testing.assert_array_equal(
            hmean,
            np.ones((3, 3)) * 15 / 7,
            err_msg="harmonic mean with 2D arrays failed",
        )

    def test_harmonic_mean_list_of_arrays(self):
        data = []
        data.append(np.ones((2, 3)))
        data.append(2 * np.ones((2, 3)))

        fractions = []
        fractions.append(0.5 * np.ones((2, 3)))
        fractions.append(0.5 * np.ones((2, 3)))

        hmean = _harmonic_mean(data, fractions)

        self.assertTrue(np.allclose(hmean, 4 / 3 * np.ones((2, 3))))

    def test_check_bounds_scalar(self):
        Ps = np.arange(-50, 51, 1)

        # for when less than range
        corrected_value = _check_bounds(input=-100, check=Ps)
        self.assertEqual(
            corrected_value,
            -50,
            msg="check bounds failed when input is less than range",
        )

        # for when greater than range
        corrected_value = _check_bounds(input=100, check=Ps)
        self.assertEqual(
            corrected_value,
            50,
            msg="check bounds failed when input is greater than range",
        )

        # for when within range
        corrected_value = _check_bounds(input=-25.5, check=Ps)
        self.assertEqual(
            corrected_value, -25.5, msg="check bounds failed when input is within range"
        )

    def test_check_bounds_array(self):
        Ps = np.arange(-50, 51, 1)
        values = np.array([-100, 100, 25.5])

        corrected_values = _check_bounds(input=values, check=Ps)
        np.testing.assert_array_equal(
            corrected_values,
            np.array([-50, 50, 25.5]),
            err_msg="check bounds failed for array of inputs",
        )

    def test_multi_table(self):

        fracs = {"tab1": 1, "tab2": 2}
        pres = 25
        temp = 25

        value = self.multitable.evaluate(P=pres, T=temp, fractions=fracs, field="vp")

        self.assertEqual(value, 15 / 7, msg="Multitable evaluate failed.")


class TestMultiTableConstruction(unittest.TestCase):
    def test_construct_from_SeismicLookupTables(self):
        test_lookup = TestLookup()
        test_lookup.setUp()

        table_paths = test_lookup.tabs
        tables = {key: SeismicLookupTable(path) for key, path in table_paths.items()}

        multitable = MultiTables(tables)

        p = 25
        t = 25
        fracs = {"tab1": 0.2, "tab2": 0.8}

        self.assertEqual(
            multitable.evaluate(p, t, fracs, "density"),
            test_lookup.multitable.evaluate(p, t, fracs, "density"),
        )


if __name__ == "__main__":
    unittest.main()
