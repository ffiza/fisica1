import unittest
import numpy as np
from nbody.nbody import _calculate_forces


class TestForceCalculation(unittest.TestCase):
    """
    Some tests for the `_calculate_forces` function defined in `nbody.py`.
    """

    def test_1d(self):
        masses = np.array([1, 1, 1])
        xposs = np.array([-1, 0, 1])
        yposs = np.array([0, 0, 0])
        grav_const = 1.0
        softening = 0.0
        forces = _calculate_forces(masses, xposs, yposs, grav_const, softening)
        np.testing.assert_allclose(forces[0, 0], 1.25)
        np.testing.assert_allclose(forces[1, 0], 0.0)
        np.testing.assert_allclose(forces[2, 0], -1.25)
        np.testing.assert_allclose(forces[0, 1], 0.0)
        np.testing.assert_allclose(forces[1, 1], 0.0)
        np.testing.assert_allclose(forces[2, 1], 0.0)

    def test_2d(self):
        masses = np.array([1, 1])
        xposs = np.array([0, 1])
        yposs = np.array([0, 1])
        grav_const = 1.0
        softening = 0.0
        forces = _calculate_forces(masses, xposs, yposs, grav_const, softening)
        np.testing.assert_allclose(forces[0, 0], np.cos(np.pi / 4) / 2)
        np.testing.assert_allclose(forces[0, 1], np.sin(np.pi / 4) / 2)
        np.testing.assert_allclose(forces[1, 0], - np.cos(np.pi / 4) / 2)
        np.testing.assert_allclose(forces[1, 1], - np.sin(np.pi / 4) / 2)


if __name__ == '__main__':
    unittest.main()
