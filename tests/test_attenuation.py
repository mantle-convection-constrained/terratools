import unittest
import numpy as np
from terratools.properties.attenuation import Q7g


class TestAttenuation(unittest.TestCase):
    def test_Goes_attenuation_single(self):
        p0 = Q7g.anelastic_properties(
            elastic_Vp=1.0,
            elastic_Vs=1.0,
            pressure=10.0e9,
            temperature=1500.0,
            frequency=1.0,
        )
        self.assertTrue(p0.Q_K == 1.0e3)

    def test_Goes_attenuation_array(self):
        p0 = Q7g.anelastic_properties(
            elastic_Vp=1.0,
            elastic_Vs=1.0,
            pressure=np.array([10.0e9, 20.0e9]),
            temperature=np.array([1500.0, 1600.0]),
            frequency=1.0,
        )
        self.assertTrue(p0.Q_K[0] == 1.0e3)
        self.assertTrue(p0.Q_K[1] == 1.0e3)
        self.assertTrue(p0.Q_S[0] != p0.Q_S[1])


if __name__ == "__main__":
    unittest.main()
