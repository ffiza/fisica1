import unittest
import numpy as np
from nbody.nbody import _calculate_kinetic_energy


class TestKineticEnergyCalculation(unittest.TestCase):
    """
    Some tests for the `_calculate_kinetic_energy` function defined in
    `nbody.py`.
    """

    def test_moving_2steps(self):
        masses = np.array([1.0, 1.0])
        xvels = np.array([[0.0, 0.0], [1.0, 0.0]])
        yvels = np.array([[0.0, 0.0], [0.0, 0.0]])
        kinetic_energy = _calculate_kinetic_energy(
            masses=masses, xvels=xvels, yvels=yvels)
        np.testing.assert_allclose(kinetic_energy[0], 0.0)
        np.testing.assert_allclose(kinetic_energy[1], 0.5)

    def test_rest_1steps(self):
        masses = np.array([1.0, 1.0])
        xvels = np.array([[0.0, 0.0], [0.0, 0.0]])
        yvels = np.array([[0.0, 0.0], [0.0, 0.0]])
        kinetic_energy = _calculate_kinetic_energy(
            masses=masses, xvels=xvels, yvels=yvels)
        np.testing.assert_allclose(kinetic_energy[0], 0.0)
        np.testing.assert_allclose(kinetic_energy[1], 0.0)

    def test_moving_1part_3steps(self):
        masses = np.array([2.0])
        xvels = np.array([[1.0], [2.0], [3.0]])
        yvels = np.array([[0.0], [0.0], [0.0]])
        kinetic_energy = _calculate_kinetic_energy(
            masses=masses, xvels=xvels, yvels=yvels)
        np.testing.assert_allclose(kinetic_energy[0], 1.0)
        np.testing.assert_allclose(kinetic_energy[1], 4.0)
        np.testing.assert_allclose(kinetic_energy[2], 9.0)


if __name__ == '__main__':
    unittest.main()
