import unittest
import numpy as np
from terratools import terra_model as tm

m = tm.TerraModel(
    lon=[0], lat=[0], r=[4000, 6000, 6370], fields={"t": np.zeros((3, 1))}
)


class TestAdiabat(unittest.TestCase):
    def test_calculate_adiabat(self):
        depth = 2000

        t1 = tm._calculate_adiabat(depth)

        # upper mantle
        t1_compare = ((depth * 0.5) + 1600) * (
            1 - (1 / (1 + (np.exp((-1 * depth - 660) / 60))))
        )

        # lower mantle
        t2_compare = ((-0.00002 * depth**2) + (0.4 * depth) + 1700) * (
            1 / (1 + (np.exp((-1 * depth - 660) / 60)))
        )

        # combined
        adiabat = t1_compare + t2_compare - 1600

        self.assertEqual(t1, adiabat, "relative adiabat calculation is not working")

    def test_add_adiabat(self):

        m.add_adiabat()
        radii = m.get_radii()
        temps = m.get_field("t")

        depth = radii.max() - radii[0]

        t1 = ((-0.00002 * depth**2) + (0.4 * depth) + 1700) * (
            1 / (1 + (np.exp((-1 * depth - 660) / 60)))
        )
        t2 = ((depth * 0.5) + 1600) * (
            1 - (1 / (1 + (np.exp((-1 * depth - 660) / 60))))
        )

        adiabat = t1 + t2 - 1600

        self.assertAlmostEqual(
            temps[0],
            np.array(adiabat),
            msg="did not add adiabat correctly",
        )
