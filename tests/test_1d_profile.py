import unittest
import numpy as np
from terratools import terra_model as tm


class TestAdiabat(unittest.TestCase):
    def setUp(self):
        self.t_values = np.array([[0, 10], [0, 10], [0, 10]])
        self.m = tm.TerraModel(
            lon=[0, 10], lat=[0, 10], r=[4000, 6000, 6370], fields={"t": self.t_values}
        )

    def test_recover_1d_profile(self):
        t_profile = np.array([0, 0, 0])

        out_profile = self.m.get_1d_profile(lat=0, lon=0, field="t")
        self.assertEqual(
            t_profile[0], out_profile[0], "recovering 1d profile not working."
        )
        self.assertEqual(
            t_profile[1], out_profile[1], "recovering 1d profile not working."
        )
        self.assertEqual(
            t_profile[2], out_profile[2], "recovering 1d profile not working."
        )

    def test_mean_1d_profile(self):
        t_profile = np.array([5, 5, 5])
        out_profile = self.m.mean_1d_profilen(field="t")

        self.assertEqual(t_profile[0], out_profile[0], "mean 1d profile failed.")
        self.assertEqual(t_profile[1], out_profile[1], "mean 1d profile failed.")
        self.assertEqual(t_profile[2], out_profile[2], "mean 1d profile failed.")
