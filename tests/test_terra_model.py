import unittest

from terratools.terra_model import TerraModel


class TestTerraModel(unittest.TestCase):
    def test_model_layers(self):
        model = TerraModel(12)
        self.assertTrue(len(model.radii) == model.nlayers)


if __name__ == '__main__':
    unittest.main()
