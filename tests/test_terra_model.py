import unittest

import terratools.terra_model
from terratools.terra_model import TerraModel


class TestTerraModel(unittest.TestCase):
    def test_is_valid_field_name(self):
        """Test for validity of a field name"""
        self.assertFalse(terra_model._is_valid_field_name("incorrect field name"))
        self.assertEqual(terra_model._is_valid_field_name("t"))

    def test_variable_name_from_field(self):
        """Translation of field name to NetCDF variable name(s)"""
        self.assertEqual(terra_model._variable_name_from_field("t"),
            ["Temperature"])
        self.assertEqual(terra_model._variable_name_from_field("u"),
            ["Velocity_x", "Velocity_y", "Velocity_z"])
        self.assertEqual(terra_model._variable_name_from_field("c_hist"),
            ["BasaltFrac", "LherzFrac"])
        with self.assertRaises(KeyError):
            terra_model._variable_name_from_field("incorrect field name")

    def test_invalid_field_name(self):
        """Check that an error is thrown when asking for an invalid field"""
        model = TerraModel()
        with self.assertRaises(KeyError):
            model.get_value(4000, 0, 0, "incorrect field name")


if __name__ == '__main__':
    unittest.main()
