import unittest
import numpy as np
from terratools import terra_model as tm


class TestRadialProfile(unittest.TestCase):
    def setUp(self):
        self.t_values = np.array([[0, 10], [0, 10], [0, 10]])
        self.m = tm.TerraModel(
            lon=[0, 10], lat=[0, 10], r=[4000, 6000, 6370], fields={"t": self.t_values}
        )

    def test_incorrect_method(self):
        with self.assertRaises(ValueError):
            self.m.radial_profile(0, 0, "t", method="incorrect method")

    def test_recover_radial_profile(self):
        t_profile = np.array([0, 0, 0])

        out_profile = self.m.radial_profile(lat=0, lon=0, field="t")
        self.assertEqual(
            t_profile[0], out_profile[0], "recovering radial profile not working."
        )
        self.assertEqual(
            t_profile[1], out_profile[1], "recovering radial profile not working."
        )
        self.assertEqual(
            t_profile[2], out_profile[2], "recovering radial profile not working."
        )

    def test_mean_radial_profile(self):
        t_profile = np.array([5, 5, 5])
        out_profile = self.m.mean_radial_profile(field="t")

        self.assertEqual(t_profile[0], out_profile[0], "mean radial profile failed.")
        self.assertEqual(t_profile[1], out_profile[1], "mean radial profile failed.")
        self.assertEqual(t_profile[2], out_profile[2], "mean radial profile failed.")
