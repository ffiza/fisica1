import numpy as np
from typing import Tuple
import argparse
import yaml
import pandas as pd


def simulate(mass: float,
             position: Tuple[float, float],
             velocity: Tuple[float, float],
             elastic_const: float,
             natural_length: float,
             grav_const: float,
             timestep: float,
             n_steps: int,
             filename: str,
             ) -> pd.DataFrame:
    """
    Perform the spring and mass simulation.

    Parameters
    ----------
    mass : float
        The mass of particle in kg.
    position : Tuple[float, float]
        The initial position in the x- and y- axis of the particle in m.
    velocity : Tuple[float, float]
        The initial velocity in the x- and y- axis of the particle in m/s.
    elastic_const : float
        The elastic constant of the spring in N/m.
    natural_length : float
        The natural length of the spring in m.
    grav_const : float
        The gravitational constant g in m/s^2.
    timestep : float
        The time step of the simulation in s.
    n_steps : int
        The number of steps to simulate.
    filename : str
        The name of the simulation file.

    Returns
    -------
    df : pd.DataFrame
        A data frame with the relevant information resulting from the
        simulation. Each row represents a given step.
    """

    pos = np.zeros((n_steps, 2))
    vel = np.zeros((n_steps, 2))
    time = np.zeros(n_steps)

    pos[0] = np.array(position)
    vel[0] = np.array(velocity)

    for step in range(1, n_steps):
        versor = pos[step - 1] / np.linalg.norm(pos[step - 1])
        force = - elastic_const * (pos[step - 1] - natural_length * versor) \
            - mass * grav_const * np.array([0, 1])

        acc = force / mass

        # TODO: Maybe test the semi-implicit Euler method for better results?
        # Integrate using the standard Euler method
        vel[step] = vel[step - 1] + acc * timestep
        pos[step] = pos[step - 1] + vel[step - 1] * timestep

        # Update the time
        time[step] = time[step - 1] + timestep

    norms = np.linalg.norm(pos, axis=1)
    versors = np.divide(pos, norms.reshape((n_steps, 1)))
    elastic_potential = 0.5 * elastic_const \
        * np.linalg.norm(pos - natural_length * versors, axis=1)**2
    gravitational_potential = mass * grav_const * pos[:, 1]

    df = pd.DataFrame({
        "Time": time,
        "xPosition": pos[:, 0],
        "yPosition": pos[:, 1],
        "xVelocity": vel[:, 0],
        "yVelocity": vel[:, 1],
        "ElasticPotential": elastic_potential,
        "GravitationalPotential": gravitational_potential,
        "KineticEnergy": 0.5 * mass * np.linalg.norm(vel, axis=1)**2})
    df["Energy"] = df["KineticEnergy"] + df["ElasticPotential"] \
        + df["GravitationalPotential"]

    df.to_csv(f"animations/mass_and_spring/data/{filename}.csv")

    return df


def main():
    # Get IC file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="The configuration file.")
    args = parser.parse_args()

    # Read configuration file
    config = yaml.safe_load(
        open(f"animations/mass_and_spring/configs/{args.config}.yml"))

    # Run simulation
    simulate(mass=config["mass"],
             position=(config["xposition"], config["yposition"]),
             velocity=(config["xvelocity"], config["yvelocity"]),
             elastic_const=config["elastic_const"],
             natural_length=config["natural_length"],
             grav_const=config["grav_const"],
             timestep=config["timestep"],
             n_steps=config["n_steps"],
             filename=config["filename"],
             )


if __name__ == "__main__":
    main()
