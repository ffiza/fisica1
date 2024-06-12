import unittest
import numpy as np
from nbody.nbody import _calculate_gravitational_potential


class TestPotentialCalculation(unittest.TestCase):
    """
    Some tests for the `_calculate_gravitational_potential` function defined
    in `nbody.py`.
    """

    def test_1d(self):
        masses = np.array([1, 1, 1])
        xposs = np.array([-1, 0, 1])
        yposs = np.array([0, 0, 0])
        grav_const = 1.0
        potential = _calculate_gravitational_potential(
            masses, xposs, yposs, grav_const)
        np.testing.assert_allclose(potential, -2.5)

    def test_2d(self):
        masses = np.array([1, 1])
        xposs = np.array([0, 1])
        yposs = np.array([0, 1])
        grav_const = 1.0
        potential = _calculate_gravitational_potential(
            masses, xposs, yposs, grav_const)
        np.testing.assert_allclose(potential, - 1 / np.sqrt(2))


if __name__ == '__main__':
    unittest.main()
