from cgitb import lookup
import unittest
import numpy as np
from terratools.flow_conversion import rotate_vector
import terratools.terra_model as tm



class TestRotateVector(unittest.TestCase):
    def setUp(self):
        self.x_vector = [1, 0, 0]
        self.y_vector = [0, 1, 0]
        self.z_vector = [0, 0, 1]
        self.minus_x_vector = [-1, 0, 0]
        self.minus_y_vector = [0, -1, 0]
        self.minus_z_vector = [0, 0, -1]

    def test_north_at_equator(self):
        lat = 0
        lon = 0
        rotated_vector = rotate_vector(self.z_vector, lat, lon)
        expected_vector = np.array([1, 0, 0])

        self.assertAlmostEqual(rotated_vector[0], expected_vector[0])
        self.assertAlmostEqual(rotated_vector[1], expected_vector[1])
        self.assertAlmostEqual(rotated_vector[2], expected_vector[2])

    def test_south_at_equator(self):
        lat = 0
        lon = 0
        rotated_vector = rotate_vector(self.minus_z_vector, lat, lon)
        expected_vector = np.array([-1, 0, 0])
        self.assertAlmostEqual(rotated_vector[0], expected_vector[0])
        self.assertAlmostEqual(rotated_vector[1], expected_vector[1])
        self.assertAlmostEqual(rotated_vector[2], expected_vector[2])

    def test_radial_at_equator(self):
        lat = 0
        lon = 0
        rotated_vector = rotate_vector(self.x_vector, lat, lon)
        expected_vector = np.array([0, 0, 1])
        self.assertAlmostEqual(rotated_vector[0], expected_vector[0])
        self.assertAlmostEqual(rotated_vector[1], expected_vector[1])
        self.assertAlmostEqual(rotated_vector[2], expected_vector[2])

    def test_inward_at_equator(self):
        lat = 0
        lon = 0
        rotated_vector = rotate_vector(self.minus_x_vector, lat, lon)
        expected_vector = np.array([0, 0, -1])
        self.assertAlmostEqual(rotated_vector[0], expected_vector[0])
        self.assertAlmostEqual(rotated_vector[1], expected_vector[1])
        self.assertAlmostEqual(rotated_vector[2], expected_vector[2])

    def test_west_at_equator(self):
        lat = 0
        lon = 0
        rotated_vector = rotate_vector(self.y_vector, lat, lon)
        expected_vector = np.array([0, -1, 0])
        self.assertAlmostEqual(rotated_vector[0], expected_vector[0])
        self.assertAlmostEqual(rotated_vector[1], expected_vector[1])
        self.assertAlmostEqual(rotated_vector[2], expected_vector[2])

    def test_east_at_equator(self):
        lat = 0
        lon = 0
        rotated_vector = rotate_vector(self.minus_y_vector, lat, lon)
        expected_vector = np.array([0, 1, 0])
        self.assertAlmostEqual(rotated_vector[0], expected_vector[0])
        self.assertAlmostEqual(rotated_vector[1], expected_vector[1])
        self.assertAlmostEqual(rotated_vector[2], expected_vector[2])

    def test_radial_at_pole(self):
        lat = 90
        lon = 0
        rotated_vector = rotate_vector(self.z_vector, lat, lon)
        expected_vector = np.array([0, 0, 1])
        self.assertAlmostEqual(rotated_vector[0], expected_vector[0])
        self.assertAlmostEqual(rotated_vector[1], expected_vector[1])
        self.assertAlmostEqual(rotated_vector[2], expected_vector[2])

    def test_south_at_pole(self):
        lat = 90
        lon = 0
        rotated_vector = rotate_vector(self.x_vector, lat, lon)
        expected_vector = np.array([-1, 0, 0])
        self.assertAlmostEqual(rotated_vector[0], expected_vector[0])
        self.assertAlmostEqual(rotated_vector[1], expected_vector[1])
        self.assertAlmostEqual(rotated_vector[2], expected_vector[2])

    def test_inward_at_pole(self):
        lat = 90
        lon = 0
        rotated_vector = rotate_vector(self.minus_z_vector, lat, lon)
        expected_vector = np.array([0, 0, -1])
        self.assertAlmostEqual(rotated_vector[0], expected_vector[0])
        self.assertAlmostEqual(rotated_vector[1], expected_vector[1])
        self.assertAlmostEqual(rotated_vector[2], expected_vector[2])

    def test_multiple_input_vectors(self):
        lat = 90
        lon = 0
        rotated_vector = rotate_vector([self.minus_z_vector, self.x_vector], lat, lon)
        expected_vector = np.array([[0, 0, -1], [-1, 0, 0]])
        vec1 = rotated_vector[0]
        vec2 = rotated_vector[1]
        expected_vector_1 = expected_vector[0]
        expected_vector_2 = expected_vector[1]

        self.assertAlmostEqual(vec1[0], expected_vector_1[0])
        self.assertAlmostEqual(vec1[1], expected_vector_1[1])
        self.assertAlmostEqual(vec1[2], expected_vector_1[2])

        self.assertAlmostEqual(vec2[0], expected_vector_2[0])
        self.assertAlmostEqual(vec2[1], expected_vector_2[1])
        self.assertAlmostEqual(vec2[2], expected_vector_2[2])


class TestRotateVectorErrors(unittest.TestCase):
    def test_wrong_input_type(self):
        lon = "lon"
        lat = "lat"
        vec = ["vec1", "vec2", "vec3"]
        vec_scalar = "2"

        with self.assertRaises(AssertionError):
            rotate_vector([1, 0, 0], lat, 0)

        with self.assertRaises(AssertionError):
            rotate_vector([1, 0, 0], 0, lon)

        with self.assertRaises(AssertionError):
            rotate_vector(vec, 0, 0)

        with self.assertRaises(AssertionError):
            rotate_vector(vec_scalar, 0, 0)

class TestAddGeogFlow(unittest.TestCase):
    def setUp(self):
        self.x_vector = [1, 0, 0]
        self.y_vector = [0, 1, 0]
        self.z_vector = [0, 0, 1]
        self.minus_x_vector = [-1, 0, 0]
        self.minus_y_vector = [0, -1, 0]
        self.minus_z_vector = [0, 0, -1]
        self.m = tm.TerraModel(
            lon=[0,0], lat=[0,90], r=[6370], fields={"u_xyz": np.array([[self.z_vector, self.minus_z_vector]])}
                                )
    def test_add_geog_flow(self):

        self.m.add_geog_flow()

        u_geog = self.m.get_field('u_geog')
        print(u_geog)
        expected_vector_1 = np.array([1, 0, 0])
        expected_vector_2 = np.array([0, 0, -1])

        self.assertAlmostEqual(u_geog[0,0,0], expected_vector_1[0])
        self.assertAlmostEqual(u_geog[0,0,1], expected_vector_2[1])

if __name__ == "__main__":
    unittest.main()
