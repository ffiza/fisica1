import numpy as np
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

CONFIG = yaml.safe_load(open("configs/global.yml"))


def _calculate_forces(masses: np.ndarray,
                      xposs: np.ndarray,
                      yposs: np.ndarray,
                      grav_const: float,
                      softening: float,
                      ) -> np.ndarray:
    """
    Calculate a 2D array with the forces of a given particle. Element
    `[i, 0]` of the array is the force acting on particle `i` in the x-axis;
    element `[i, 1]` of the array is the force acting on particle `i`
    in the y-axis.

    Parameters
    ----------
    masses : np.ndarray
        The masses of the particles.
    xposs : np.ndarray
        The x-positions of the particles.
    yposs : np.ndarray
        The y-positions of the particles.
    grav_const : float
        The gravitational constant in m^3 / kg / s^2.
    softening : float
        The softening length to use in the force calculation in m.

    Returns
    -------
    forces : np.ndarray
        A 2D array with the forces of the particles.
    """
    pos = np.vstack((xposs, yposs)).T  # Particle positions as an array
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)  # Avoid division by zero

    forces_mag = grav_const * (masses[:, np.newaxis] * masses[np.newaxis, :]) \
        / (dist**2 + softening**2)
    forces = - forces_mag[:, :, np.newaxis] * (diff / dist[:, :, np.newaxis])

    return np.sum(forces, axis=1)


def _calculate_gravitational_potential(masses: np.ndarray,
                                       xposs: np.ndarray,
                                       yposs: np.ndarray,
                                       grav_const: float
                                       ) -> float:
    """
    Calculate a the total gravitational potential of the system.

    Parameters
    ----------
    masses : np.ndarray
        The masses of the particles.
    xposs : np.ndarray
        The x-positions of the particles.
    yposs : np.ndarray
        The y-positions of the particles.
    grav_const : float
        The gravitational constant in m^3 / kg / s^2.

    Returns
    -------
    potential : float
        The total gravitational potential energy.
    """

    n_steps, n_particles = xposs.shape
    potential = np.zeros(n_steps)

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = xposs[:, j] - xposs[:, i]
            dy = yposs[:, j] - yposs[:, i]
            r = np.sqrt(dx**2 + dy**2)
            potential -= grav_const * masses[i] * masses[j] / r

    return potential


def _calculate_kinetic_energy(masses: np.ndarray,
                              xvels: np.ndarray,
                              yvels: np.ndarray,
                              ) -> float:
    """
    Calculate a the total kinetic energy of the system.

    Parameters
    ----------
    masses : np.ndarray
        The masses of the particles. This array is of shape (n_particles,)
    xvels : np.ndarray
        The x-velocities of the particles. This array is of shape (n_steps,
        n_particles).
    yvels : np.ndarray
        The y-velocities of the particles. This array is of shape (n_steps,
        n_particles).

    Returns
    -------
    float
        The total kinetic energy.
    """
    n_particles = masses.shape[0]
    kinetic_energies = 0.5 * masses.reshape(1, n_particles) \
        * (xvels**2 + yvels**2)
    return np.sum(kinetic_energies, axis=1)


def _calculate_leapfrog_step(masses: np.ndarray,
                             xposs: np.ndarray,
                             yposs: np.ndarray,
                             xvels: np.ndarray,
                             yvels: np.ndarray,
                             particle_types: np.ndarray,
                             grav_const: float,
                             softening: float,
                             timestep: float) -> tuple:
    """
    Calculates one step in the simulation using the leapfrog method.

    Parameters
    ----------
    masses : np.ndarray
        A NumPy array with the masses of the particles to be simulated.
    xposs : np.ndarray
        A NumPy array with the initial positions of the particles
        in the x-axis.
    yposs : np.ndarray
        A NumPy array with the initial positions of the particles
        in the y-axis.
    xvels : np.ndarray
        A NumPy array with the initial velocities of the particles
        in the x-axis.
    yvels : np.ndarray
        A NumPy array with the initial velocities of the particles
        in the y-axis.
    particle_types : np.ndarray
        A NumPy array with `0` for the static particles and `1` for
        the dynamic ones.
    grav_const : float
        The gravitational constant in m^3 / kg / s^2.
    softening : float
        The softening length to use in the force calculation in m.
    timestep : float
        The timestep of the simulation.
    n_steps : int
        The number of steps to simulate.

    Returns
    -------
    tuple
        A tuple with the new positions in the x-axis, the new positions in
        the y-axis, the new velocities in the x-axis and the new velocities
        in the y-axis.
    """
    # Calculate the forces acting on the particles on previous step
    forces_then = _calculate_forces(
        masses=masses, xposs=xposs, yposs=yposs,
        grav_const=grav_const, softening=softening)
    # Null forces for static particles
    forces_then[particle_types == 0] = np.zeros(2)

    # Update the positions
    acc_then = forces_then / masses[:, np.newaxis]
    new_xposs = xposs + xvels * timestep + 0.5 * acc_then[:, 0] * timestep**2
    new_yposs = yposs + yvels * timestep + 0.5 * acc_then[:, 1] * timestep**2

    # Recalculate the force in the current step
    forces_now = _calculate_forces(
        masses=masses, xposs=new_xposs, yposs=new_yposs,
        grav_const=grav_const, softening=softening)
    # Null forces for static particles
    forces_now[particle_types == 0] = np.zeros(2)

    # Update the velocities
    acc_now = forces_now / masses[:, np.newaxis]
    new_xvels = xvels + 0.5 * (acc_then[:, 0] + acc_now[:, 0]) * timestep
    new_yvels = yvels + 0.5 * (acc_then[:, 1] + acc_now[:, 1]) * timestep

    return new_xposs, new_yposs, new_xvels, new_yvels


def simulate(masses: list,
             initial_xposs: list,
             initial_yposs: list,
             initial_xvels: list,
             initial_yvels: list,
             particle_types: list,
             grav_const: float,
             softening: float,
             timestep: float,
             n_steps: int,
             filename: str,
             ) -> pd.DataFrame:
    """
    Perform an N-body simulation.

    Parameters
    ----------
    masses : list
        A list with the masses of the particles to be simulated.
    initial_xposs : list
        A list with the initial positions of the particles in the x-axis.
    initial_yposs : list
        A list with the initial positions of the particles in the y-axis.
    initial_xvels : list
        A list with the initial velocities of the particles in the x-axis.
    initial_yvels : list
        A list with the initial velocities of the particles in the y-axis.
    particle_types : list
        A list with `0` for the static particles and `1` for the dynamic ones.
    grav_const : float
        The gravitational constant in m^3 / kg / s^2.
    softening : float
        The softening length to use in the force calculation in m.
    timestep : float
        The timestep of the simulation.
    n_steps : int
        The number of steps to simulate.
    filename : str
        The name of the simulation file.

    Returns
    -------
    pd.DataFrame
        A Pandas data frame with the results of the simulation.
    """
    # Check if all inputs have the same size
    if len(masses) != len(initial_xposs) != len(initial_yposs) != \
            len(initial_xvels) != len(initial_yvels):
        raise ValueError("All inputs must have the same size.")

    n_bodies = len(masses)

    # Make any velocity of a static particle
    initial_xvels[particle_types == 0] = 0.0
    initial_yvels[particle_types == 0] = 0.0

    time = np.zeros(n_steps)

    xposs = np.zeros((n_steps, n_bodies))
    yposs = np.zeros((n_steps, n_bodies))
    xvels = np.zeros((n_steps, n_bodies))
    yvels = np.zeros((n_steps, n_bodies))

    xposs[0] = initial_xposs
    yposs[0] = initial_yposs
    xvels[0] = initial_xvels
    yvels[0] = initial_yvels

    # Integrate using the leapfrog method
    for step in tqdm(range(1, n_steps),
                     desc="Integrating...",
                     colour="#A241B2",
                     ncols=100):

        new_xposs, new_yposs, new_xvels, new_yvels = \
            _calculate_leapfrog_step(
                masses,
                xposs[step - 1], yposs[step - 1],
                xvels[step - 1], yvels[step - 1],
                particle_types, grav_const, softening, timestep)

        # Update quantities
        xposs[step] = new_xposs
        yposs[step] = new_yposs
        xvels[step] = new_xvels
        yvels[step] = new_yvels
        time[step] = time[step - 1] + timestep

    # Create data frame with the results
    df = pd.DataFrame()
    df['Time'] = time
    for k in range(n_bodies):
        df[f"xPosition{k}"] = xposs[:, k]
        df[f"yPosition{k}"] = yposs[:, k]
        df[f"xVelocity{k}"] = xvels[:, k]
        df[f"yVelocity{k}"] = yvels[:, k]
    df["KineticEnergy"] = _calculate_kinetic_energy(
            masses=masses, xvels=xvels, yvels=yvels)
    df["Potential"] = _calculate_gravitational_potential(
        masses=masses, xposs=xposs, yposs=yposs, grav_const=grav_const)
    df["Energy"] = df["KineticEnergy"] + df["Potential"]

    df = df.round(decimals=CONFIG["DATAFRAME_DECIMALS"])

    # Sample data frame to a given FPS
    n_rows = int(CONFIG["FPS"] * df["Time"].to_numpy()[-1])
    idx = (np.linspace(
        0, 1, n_rows, endpoint=False) * len(df)).astype(np.int64)
    df = df.iloc[idx]
    df.reset_index(inplace=True, drop=True)

    df.to_csv(f"results/{filename}.csv")

    return df


def main():
    # Get IC file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics", type=str, required=True,
                        help="The physics configuration file.")
    parser.add_argument("--ic", type=str, required=True,
                        help="The initial condition file.")
    args = parser.parse_args()

    # Read configuration file
    PHYSICS = yaml.safe_load(open(f"configs/physics_{args.physics}.yml"))
    IC = pd.read_csv(f"ics/ic_{args.ic}.csv")

    # Run simulation
    simulate(masses=np.array(IC["Mass_kg"].to_list()),
             initial_xposs=np.array(IC["xPosition_m"].to_list()),
             initial_yposs=np.array(IC["yPosition_m"].to_list()),
             initial_xvels=np.array(IC["xVelocity_m/s"].to_list()),
             initial_yvels=np.array(IC["yVelocity_m/s"].to_list()),
             particle_types=np.array(IC["DynamicParticle"].to_list()),
             grav_const=PHYSICS["GRAV_CONST"],
             softening=PHYSICS["SOFTENING_LENGTH"],
             timestep=PHYSICS["TIMESETP"],
             n_steps=PHYSICS["N_STEPS"],
             filename=f"simulation_p{args.physics}_ic{args.ic}")


if __name__ == "__main__":
    main()
