"""
Title: An N-Body Simulation
Author: Federico Iza
"""

import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import argparse
import pandas as pd
from tqdm import tqdm
import shutil

GRAV_CONST: float = 10.0  # m^3 / kg / s^2

VIOLET: str = "#A241B2"
BLUE: str = "#2C73D6"
GREEN: str = "#00D75B"
YELLOW: str = "#FEF058"
ORANGE: str = "#FFAA4C"
RED: str = "#FA4656"
COLORS: list = [RED, BLUE, GREEN, YELLOW, ORANGE, VIOLET]
BACKGROUND: str = "#010001"


def calculate_forces(masses: np.ndarray,
                     xposs: np.ndarray,
                     yposs: np.ndarray,
                     ) -> np.ndarray:
    """
    Calculate a 3D array with the forces of a given particle. Element
    `[i, j, 0]` of the array is the force acting on particle `i` due to
    particle `j` in the x-axis. Element `[i, j, 1]` of the array is the force
    acting on particle `i` due to particle `j` in the y-axis.

    Parameters
    ----------
    masses : np.ndarray
        The masses of the particles.
    xposs : np.ndarray
        The x-positions of the particles.
    yposs : np.ndarray
        The y-positions of the particles.

    Returns
    -------
    forces : np.ndarray
        A 3D array with the forces of the particles.
    """
    n_bodies = len(masses)

    # Calculate the difference of position vectors
    dr = np.zeros((n_bodies, n_bodies, 2), dtype=np.float64)
    for i, j in itertools.product(range(n_bodies), range(n_bodies)):
        dr[i, j] = np.array([xposs[i] - xposs[j], yposs[i] - yposs[j]])

    # Calculate the force matrix
    forces = np.zeros((n_bodies, n_bodies, 2), dtype=np.float64)
    for i, j in itertools.product(range(n_bodies), range(n_bodies)):
        if j > i:
            forces[i, j] = - GRAV_CONST * masses[i] \
                * masses[j] * dr[i, j] / np.linalg.norm(dr[i, j])**3
            forces[j, i] = - forces[i, j]

    return forces


def calculate_gravitational_potential(masses: np.ndarray,
                                      xposs: np.ndarray,
                                      yposs: np.ndarray,
                                      ) -> np.ndarray:
    """
    Calculate a the total gravitational potential of the system at the
    given state.

    Parameters
    ----------
    masses : np.ndarray
        The masses of the particles.
    xposs : np.ndarray
        The x-positions of the particles.
    yposs : np.ndarray
        The y-positions of the particles.

    Returns
    -------
    potential : float
        The total gravitational potential energy.
    """
    n_bodies = len(masses)

    potential = 0.0
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            dr = np.array([xposs[i] - xposs[j], yposs[i] - yposs[j]])
            potential += - GRAV_CONST * masses[i] * masses[j] \
                / np.linalg.norm(dr)
    return potential


def calculate_kinetic_energy(masses: np.ndarray,
                             xvels: np.ndarray,
                             yvels: np.ndarray,
                             ) -> np.ndarray:
    """
    Calculate a the total kinetic energy of the system.

    Parameters
    ----------
    masses : np.ndarray
        The masses of the particles.
    xvels : np.ndarray
        The x-velocities of the particles.
    yvels : np.ndarray
        The y-velocities of the particles.

    Returns
    -------
    energy : float
        The total kinetic energy.
    """
    n_bodies = len(masses)

    energy = 0.0
    for i in range(n_bodies):
        energy += 0.5 * masses[i] * (xvels[i]**2 + yvels[i]**2)
    return energy


def simulate(masses: list,
             initial_xposs: list,
             initial_yposs: list,
             initial_xvels: list,
             initial_yvels: list,
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

    if len(masses) > len(COLORS):
        raise ValueError("There aren't that many colors, keep the number "
                         f"of particles below {len(COLORS) + 1}, ok?")

    n_bodies = len(masses)

    time = np.zeros(n_steps)

    xposs = np.zeros((n_steps, n_bodies))
    yposs = np.zeros((n_steps, n_bodies))
    xvels = np.zeros((n_steps, n_bodies))
    yvels = np.zeros((n_steps, n_bodies))

    for j in range(n_bodies):
        xposs[0, j] = initial_xposs[j]
        yposs[0, j] = initial_yposs[j]
        xvels[0, j] = initial_xvels[j]
        yvels[0, j] = initial_yvels[j]

    # Integrate using the leapfrog method
    for step in range(1, n_steps):
        # Calculate the forces acting on the particles on previous step
        forces_then = calculate_forces(masses=masses,
                                       xposs=xposs[step - 1],
                                       yposs=yposs[step - 1])
        # Update the positions
        for k in range(n_bodies):
            acc_then = forces_then[k].sum(axis=0) / masses[k]
            xposs[step, k] = xposs[step - 1, k] \
                + xvels[step - 1, k] * timestep \
                + 0.5 * acc_then[0] * timestep**2
            yposs[step, k] = yposs[step - 1, k] \
                + yvels[step - 1, k] * timestep \
                + 0.5 * acc_then[1] * timestep**2
        # Recalculate the force in the current step
        forces_now = calculate_forces(masses=masses,
                                      xposs=xposs[step],
                                      yposs=yposs[step])
        # Update the velocities
        for k in range(n_bodies):
            acc_then = forces_then[k].sum(axis=0) / masses[k]
            acc_now = forces_now[k].sum(axis=0) / masses[k]
            xvels[step, k] = xvels[step - 1, k] \
                + 0.5 * (acc_then[0] + acc_now[0]) * timestep
            yvels[step, k] = yvels[step - 1, k] \
                + 0.5 * (acc_then[1] + acc_now[1]) * timestep
        # Update the time
        time[step] = time[step - 1] + timestep

    df = pd.DataFrame()
    df['Time'] = time
    for k in range(n_bodies):
        df[f"xPosition{k}"] = xposs[:, k]
        df[f"yPosition{k}"] = yposs[:, k]
        df[f"xVelocity{k}"] = xvels[:, k]
        df[f"yVelocity{k}"] = yvels[:, k]

    # Calculate energies
    kinetic_energies = np.zeros(len(df))
    potentials = np.zeros(len(df))
    for i in range(len(df)):
        kinetic_energies[i] = calculate_kinetic_energy(masses=masses,
                                                       xvels=xvels[i],
                                                       yvels=yvels[i])
        potentials[i] = calculate_gravitational_potential(masses=masses,
                                                          xposs=xposs[i],
                                                          yposs=yposs[i])
    df["KineticEnergy"] = kinetic_energies
    df["Potential"] = potentials
    df["Energy"] = df["KineticEnergy"] + df["Potential"]

    df.to_csv(f"data/{filename}.csv")

    return df


def multi_width_line(x: np.ndarray,
                     y: np.ndarray,
                     color: str = "black",
                     max_width: float = 2.5,
                     min_width: float = 0.0) -> LineCollection:
    """
    Create a LineCollection instance that can be used as a line with
    different widths.

    Parameters
    ----------
    x : np.ndarray
        The values in the x-axis.
    y : np.ndarray
        The values in the y-axis.
    color : str, optional
        The color of the line, by default "black".
    max_width : float, optional
        The maximum width of the line, by default 3.0.
    min_width : float, optional
        The minimum width of the line, by default 0.0.

    Returns
    -------
    lc : LineCollection
        The LineCollection instance.
    """
    lws = np.linspace(min_width, max_width, len(x))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return LineCollection(segments, linewidths=lws, color=color)


def scene(df: pd.DataFrame,
          idx: int,
          frame_path: str,
          ) -> None:
    """
    Create the scene (frame) for row `idx` of `df` and store the image
    at `frame_path`.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with the results of each time step of the simulation.
    idx : int
        The row index to represent in the scene.
    frame_path : str
        A path (including file name) to store the image.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.set_facecolor(BACKGROUND)
    ax.set_xlim((-16, 16))
    ax.set_ylim((-9, 9))
    ax.set_xticks([])
    ax.set_yticks([])

    plt.text(x=-15.5,
             y=-8.5,
             s=f"Time: {df['Time'].loc[idx]:.1f} s",
             va="bottom",
             ha="left",
             color="lightgray")
    plt.text(x=-15.5,
             y=-7.5,
             s=f"Energy: {df['Energy'].loc[idx]:.2f} J",
             va="bottom",
             ha="left",
             color="lightgray")

    # Plot reference lines
    ax.plot(ax.get_xlim(), [0, 0], lw=0.5, ls="solid", c="dimgray", zorder=-10)
    ax.plot([0, 0], ax.get_ylim(), lw=0.5, ls="solid", c="dimgray", zorder=-10)

    # Plot energy bars
    energies = df[["Potential", "KineticEnergy", "Energy"]].to_numpy()
    max_energy = np.max(np.abs(energies))
    factor = 5 / max_energy

    rectangles = [
        plt.Rectangle(xy=(-15, 0),
                      width=0.5,
                      height=df["Energy"].loc[idx] * factor,
                      color="lightgray"),
        plt.Rectangle(xy=(-14, 0),
                      width=0.5,
                      height=df["Potential"].loc[idx] * factor,
                      color="lightgray"),
        plt.Rectangle(xy=(-13, 0),
                      width=0.5,
                      height=df["KineticEnergy"].loc[idx] * factor,
                      color="lightgray")]
    for rectangle in rectangles:
        ax.add_patch(rectangle)

    # Fix the energy label to avoid changes due to numerical noise
    label_props = (0.5, "bottom") if np.mean(df["Energy"]) <= 0 else (-0.5,
                                                                      "top")
    ax.annotate(text=r"$\mathbf{E}$",
                xy=(-14.75, label_props[0]),
                ha="center",
                va=label_props[1],
                fontsize=12,
                weight="bold",
                color="lightgray")
    ax.annotate(text=r"$\mathbf{U}$",
                xy=(-13.75, 0.5),
                ha="center",
                va="bottom",
                fontsize=12,
                weight="bold",
                color="lightgray")
    ax.annotate(text=r"$\mathbf{K}$",
                xy=(-12.75, -0.5),
                ha="center",
                va="top",
                fontsize=12,
                weight="bold",
                color="lightgray")

    # Find the number of particles stored in the data frame
    n_bodies = df.filter(regex="xPosition\\d+").shape[1]

    for i in range(n_bodies):
        # Plot the trail of each particle
        lc = multi_width_line(x=df[f"xPosition{i}"].to_numpy()[:idx + 1],
                              y=df[f"yPosition{i}"].to_numpy()[:idx + 1],
                              color=COLORS[i])
        ax.add_collection(lc)

        # Plot the particles as circles
        ax.scatter(x=df[f"xPosition{i}"].loc[idx],
                   y=df[f"yPosition{i}"].loc[idx],
                   color=COLORS[i],
                   s=40.0)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(frame_path, dpi=600, pad_inches=0, bbox_inches="tight")
    plt.close()


def generate_frames(df: pd.DataFrame, one_every: int = 1):
    """
    Create all the frames of the simulation, for for each row in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas data frame with the results of each step of the simulation
        as rows.
    one_every : int, optional
        Output one frame every `one_every` frames. 1 by default.
    """
    n_frames = len(df)
    if os.path.exists("movies/frames/"):
        shutil.rmtree("movies/frames/")
    os.makedirs(os.path.dirname("movies/frames/"), exist_ok=False)
    j = 0
    for i in tqdm(range(n_frames)):
        if i == 0 or i % one_every == 0:
            scene(df=df,
                  idx=i,
                  frame_path=f"movies/frames/frame{j}.png")
            j += 1


def movie(fps: int,
          frames_dir: str = "movies/frames/",
          output_dir: str = "movies/",
          filename: str = "movie") -> None:
    """
    Create a movie from the images in `frames_dir`.

    Parameters
    ----------
    fps : int
        The frames per second of the movie.
    frames_dir : str, optional
        The directory of the images, by default "movies/frames/".
    output_dir : str, optional
        The directory to save the output file, by default "movies/".
    filename : str, optional
        The name of the output file, by default "movie".
    """
    os.system(
        f"ffmpeg -r {fps} -i {frames_dir}frame%01d.png "
        f"-vcodec mpeg4 -y {output_dir}{filename}.mp4")
    shutil.rmtree(frames_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masses",
                        type=float,
                        nargs="+",
                        required=True,
                        help="The masses of the particles.")
    parser.add_argument("--xpositions",
                        type=float,
                        nargs="+",
                        required=True,
                        help="The initial positions of the particles in the "
                             "x-axis.")
    parser.add_argument("--ypositions",
                        type=float,
                        nargs="+",
                        required=True,
                        help="The initial positions of the particles in the "
                             "y-axis.")
    parser.add_argument("--xvelocities",
                        type=float,
                        nargs="+",
                        required=True,
                        help="The initial velocities of the particles in the "
                             "x-axis.")
    parser.add_argument("--yvelocities",
                        type=float,
                        nargs="+",
                        required=True,
                        help="The initial velocities of the particles in the "
                             "y-axis.")
    parser.add_argument("--timestep",
                        type=float,
                        required=True,
                        help="The time step of the simulation in seconds.")
    parser.add_argument("--steps",
                        type=int,
                        required=True,
                        help="The number of steps.")
    parser.add_argument("--filename",
                        type=str,
                        required=True,
                        help="The name of the simulation file.")
    parser.add_argument("--fps",
                        type=int,
                        required=True,
                        help="The frames per second of the animation.")
    parser.add_argument("--one_every",
                        type=int,
                        required=False,
                        default=1,
                        help="Output one frame every `one_every` frames.")
    args = parser.parse_args()

    # Run simulation
    df = simulate(masses=args.masses,
                  initial_xposs=args.xpositions,
                  initial_yposs=args.ypositions,
                  initial_xvels=args.xvelocities,
                  initial_yvels=args.yvelocities,
                  timestep=args.timestep,
                  n_steps=args.steps,
                  filename=args.filename)
    generate_frames(df=df, one_every=args.one_every)
    movie(fps=args.fps, filename=args.filename)


if __name__ == "__main__":
    main()
