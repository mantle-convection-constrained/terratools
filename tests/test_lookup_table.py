from cgitb import lookup
import unittest
import numpy as np
from terratools.lookup_tables import SeismicLookupTable

lookup_tab_path = './data/test_lookup_table.txt'



class TestLookup(unittest.TestCase):

    def setUp(self):
        self.tab = SeismicLookupTable(lookup_tab_path)

    def test_read_file(self):
        
        # test table has been read in correctly by comparing the
        # size of the array read in.
        self.self.assertEqual(self.tab.table.shape, (121, 10), 
                              'file not read in correctly')

        self.self.assertEqual(self.tab.table[0,0], -50.0, 
                              'file not read in correctly')

    def test_interpolate_point(self):

        p_test=15
        t_test=15
        expected_vp = p_test**2 + t_test**2

        self.assertEqual(self.tab.interp_points(p_test, t_test), expected_vp, 
                         'interpolation for single point failed')

    def test_interpolate_grid(self):

        p_test = np.linspace(-20,20,5)
        t_test = np.linspace(-20,20,5)

        expected_vp = np.power(p_test, 2) + np.power(t_test, 2)

        self.assertEqual(self.tab.interp_points(p_test, t_test), expected_vp,
                         'interpolation for grid of points failed')
