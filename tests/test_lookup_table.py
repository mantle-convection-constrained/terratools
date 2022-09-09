from cgitb import lookup
import unittest
import numpy as np
from terratools.lookup_tables import SeismicLookupTable
import pathlib

TESTDATA_PATH = pathlib.Path(__file__).parent.joinpath('data','test_lookup_table.txt')
class TestLookup(unittest.TestCase):

    def setUp(self):
        self.lookup_tab_path = './tests/data/test_lookup_table.txt'
        self.tab = SeismicLookupTable(TESTDATA_PATH)

    def test_read_file(self):

        # test table has been read in correctly by comparing the
        # size of the array read in.
        self.assertEqual(self.tab.table.shape, (121, 10),
                              'file not read in correctly')

        self.assertEqual(self.tab.table[0,0], -50.0,
                              'file not read in correctly')

    def test_interpolate_point(self):

        p_test=15
        t_test=15

        self.assertAlmostEqual(self.tab.interp_points([p_test, t_test], 'Vp'), np.array([15]),
                               msg='interpolation for single point failed')

    def test_interpolate_grid(self):

        p_test = [4,5,6]
        t_test = 10
        outgrid = self.tab.interp_grid(p_test, t_test, 'Vp')

        self.assertAlmostEqual(outgrid[0], 4,
                               msg='interpolation for grid of points failed')
        self.assertAlmostEqual(outgrid[1], 5,
                               msg='interpolation for grid of points failed')
        self.assertAlmostEqual(outgrid[2], 6,
                               msg='interpolation for grid of points failed')


if __name__ == '__main__':
    unittest.main()
