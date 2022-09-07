import unittest
import numpy as np
from terratools.properties.attenuation import Q7g


class TestAttenuation(unittest.TestCase):
    def test_Goes_attenuation_single(self):
        p0 = Q7g.anelastic_properties(elastic_Vp=1.,
                                      elastic_Vs=1.,
                                      pressure=10.e9,
                                      temperature=1500.,
                                      frequency=1.)
        self.assertTrue(p0.Q_K == 1.e3)

    def test_Goes_attenuation_array(self):
        p0 = Q7g.anelastic_properties(elastic_Vp=1.,
                                      elastic_Vs=1.,
                                      pressure=np.array([10.e9, 20.e9]),
                                      temperature=np.array([1500., 1600.]),
                                      frequency=1.)
        self.assertTrue(p0.Q_K[0] == 1.e3)
        self.assertTrue(p0.Q_K[1] == 1.e3)
        self.assertTrue(p0.Q_S[0] != p0.Q_S[1])


if __name__ == '__main__':
    unittest.main()
