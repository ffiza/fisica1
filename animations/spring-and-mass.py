"""
Title: A Spring and Mass Simulation

Examples
--------
spring-and-mass-example1
    --xposition=5.0
    --yposition=0.0
    --xvelocity=0.0
    --yvelocity=1.0
    --mass=1.0
    --elastic_constant=1.0
    --natural_length=4.0
    --timestep=0.01
    --steps=5000
    --filename="spring-and-mass-example1"
    --fps=100
    --gravity=0.0
spring-and-mass-example2
    --xposition=5.0
    --yposition=0.0
    --xvelocity=0.0
    --yvelocity=1.0
    --mass=1.0
    --elastic_constant=1.0
    --natural_length=4.0
    --timestep=0.01
    --steps=5000
    --filename="spring-and-mass-example2"
    --fps=100
    --gravity=0.5
spring-and-mass-example3
    --xposition=5.0
    --yposition=0.0
    --xvelocity=0.0
    --yvelocity=0.0
    --mass=1.0
    --elastic_constant=1.0
    --natural_length=4.0
    --timestep=0.01
    --steps=5000
    --filename="spring-and-mass-example3"
    --fps=100
    --gravity=0.0
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Tuple
import argparse
import pandas as pd
from tqdm import tqdm
import shutil


def create_spring(xy1: Tuple[float, float],
                  xy2: Tuple[float, float],
                  lw: float,
                  color: str,
                  n_loops: int,
                  loop_width: float,
                  base_fraction: float,
                  ) -> LineCollection:
    """
    Create a line collection that represents a spring.

    Parameters
    ----------
    xy1 : Tuple[float, float]
        The position of the tail of the spring.
    xy2 : Tuple[float, float]
        The position of the head of the spring.
    lw : float
        The line width.
    color : str
        The color.
    n_loops : int
        The number of loops in the spring.
    loop_width : float
        The width of each loop.
    base_fraction : float
        The fraction of the total length corresponding to the base of the
        spring (the region with no loops).

    Returns
    -------
    spring : LineCollection
        The line collection that represents the spring.

    Raises
    ------
    ValueError
        If the number of loops is zero or negative.
    """
    if n_loops <= 0:
        raise ValueError("Loop count must be a positive integer.")

    xy1 = np.array(xy1)
    xy2 = np.array(xy2)

    n_points = 3 + 4 * n_loops

    # Properties
    length = np.linalg.norm(xy2 - xy1)
    base_length = base_fraction * length
    loops_length = length - 2 * base_length
    loop_length = loops_length / n_loops
    dxy = (xy2 - xy1) / np.linalg.norm(xy2 - xy1)
    dxy_orth = np.array((-dxy[1], dxy[0]))

    points = np.nan * np.ones((n_points, 2))
    points[0] = xy1
    points[1] = xy1 + base_fraction * (xy2 - xy1)
    multipliers = [1, -1, -1, 1]
    j = 0
    for i in range(2, n_loops * 4 + 2):
        points[i] = points[i - 1] + dxy * loop_length / 4 \
            + multipliers[j] * dxy_orth * loop_width / 2
        j += 1
        if j > 3:
            j = 0
    points[-1] = xy2

    x_points = points[:, 0]
    y_points = points[:, 1]

    points = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    spring = LineCollection(segments, linewidths=lw, color=color)

    return spring


def simulate(mass: float,
             pos0: Tuple[float, float],
             vel0: Tuple[float, float],
             elastic_constant: float,
             natural_length: float,
             gravity: float,
             timestep: float,
             n_steps: int,
             ) -> pd.DataFrame:
    """
    Perform the spring and mass simulation.

    Parameters
    ----------
    mass : float
        The mass of particle in kg.
    r1_initial : Tuple[float, float]
        The initial position in the x- and y- axis of the particle in m.
    v1_initial : Tuple[float, float]
        The initial velocity in the x- and y- axis of the particle in m/s.
    elastic_constant : float
        The elastic constant of the spring in N/m.
    natural_length : float
        The natural length of the spring in m.
    gravity : float
        The gravitational constant g in m/s^2.
    timestep : float
        The time step of the simulation in s.
    n_steps : int
        The number of steps to simulate.

    Returns
    -------
    df : pd.DataFrame
        A data frame with the relevant information resulting from the
        simulation. Each row represents a given step.
    """

    pos = np.zeros((n_steps, 2))
    vel = np.zeros((n_steps, 2))

    pos[0] = np.array(pos0)
    vel[0] = np.array(vel0)

    for i in range(1, n_steps):
        versor = pos[i - 1] / np.linalg.norm(pos[i - 1])
        force = - elastic_constant * (pos[i - 1] - natural_length * versor) \
            - mass * gravity * np.array([0, 1])

        acc = force / mass

        # Integrate using the standard Euler method
        vel[i] = vel[i - 1] + acc * timestep
        pos[i] = pos[i - 1] + vel[i - 1] * timestep

    norms = np.linalg.norm(pos, axis=1)
    versors = np.divide(pos, norms.reshape((n_steps, 1)))
    elastic_potential = 0.5 * elastic_constant \
        * np.linalg.norm(pos - natural_length * versors, axis=1)**2
    gravitational_potential = mass * gravity * pos[:, 1]

    df = pd.DataFrame({
        "xPosition": pos[:, 0],
        "yPosition": pos[:, 1],
        "xVelocity": vel[:, 0],
        "yVelocity": vel[:, 1],
        "ElasticPotential": elastic_potential,
        "GravitationalPotential": gravitational_potential,
        "KineticEnergy": 0.5 * mass * np.linalg.norm(vel, axis=1)**2})
    df["Energy"] = df["KineticEnergy"] + df["ElasticPotential"] \
        + df["GravitationalPotential"]

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
    lc = LineCollection(segments, linewidths=lws, color=color)
    return lc


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
    fig.set_facecolor("#010001")
    ax.set_xlim((-22, 18))
    ax.set_ylim((-12, 12))
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot energy bars
    energies = df[["ElasticPotential",
                   "KineticEnergy",
                   "Energy"]].to_numpy()
    max_energy = np.max(np.abs(energies))
    factor = 10 / max_energy

    rectangles = []
    rectangles.append(plt.Rectangle(
        xy=(-21, 0),
        width=1,
        height=df["Energy"].loc[idx] * factor,
        color="#fa4656"))
    rectangles.append(plt.Rectangle(
        xy=(-19, 0),
        width=1,
        height=df["ElasticPotential"].loc[idx] * factor,
        color="#a241b2"))
    rectangles.append(plt.Rectangle(
        xy=(-17, 0),
        width=1,
        height=df["GravitationalPotential"].loc[idx] * factor,
        color="#fef058"))
    rectangles.append(plt.Rectangle(
        xy=(-15, 0),
        width=1,
        height=df["KineticEnergy"].loc[idx] * factor,
        color="#2c73d6"))
    for rectangle in rectangles:
        ax.add_patch(rectangle)

    ax.annotate(text=r"$\mathbf{E}$",
                xy=(-20.5, -0.5),
                ha="center",
                va="top",
                fontsize=12,
                weight="bold",
                color="#fa4656")
    ax.annotate(text=r"$\mathbf{U}_k$",
                xy=(-18.5, -0.5),
                ha="center",
                va="top",
                fontsize=12,
                weight="bold",
                color="#a241b2")
    if df["GravitationalPotential"].loc[idx] >= 0:
        label_props = (-0.5, "top")
    else:
        label_props = (0.5, "bottom")
    ax.annotate(text=r"$\mathbf{U}_g$",
                xy=(-16.5, label_props[0]),
                ha="center",
                va=label_props[1],
                fontsize=12,
                weight="bold",
                color="#fef058")
    ax.annotate(text=r"$\mathbf{K}$",
                xy=(-14.5, -0.5),
                ha="center",
                va="top",
                fontsize=12,
                weight="bold",
                color="#2c73d6")

    spring = create_spring(xy1=(0, 0),
                           xy2=(df["xPosition"].loc[idx],
                                df["yPosition"].loc[idx]),
                           lw=1.0,
                           color="gainsboro",
                           n_loops=5,
                           loop_width=0.5,
                           base_fraction=0.3)
    ax.add_collection(spring)

    # Plot the trail of each particle
    lc = multi_width_line(x=df["xPosition"].to_numpy()[:idx + 1],
                          y=df["yPosition"].to_numpy()[:idx + 1],
                          color="#2c73d6")
    ax.add_collection(lc)

    # Plot the particles as circles
    ax.scatter(x=df["xPosition"].loc[idx],
               y=df["yPosition"].loc[idx],
               color="#2c73d6",
               s=40.0,
               zorder=10)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(frame_path, dpi=600, pad_inches=0, bbox_inches="tight")
    plt.close()


def generate_frames(df: pd.DataFrame):
    """
    Create all the frames of the simulation, for for each row in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas data frame with the results of each step of the simulation
        as rows.
    """
    n_frames = len(df)
    if os.path.exists("movies/frames/"):
        shutil.rmtree("movies/frames/")
    os.makedirs(os.path.dirname("movies/frames/"), exist_ok=False)
    for i in tqdm(range(n_frames)):
        scene(df=df,
              idx=i,
              frame_path=f"movies/frames/frame{i}.png",
              )


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
    """
    Main method to get arguments from user, run the simulation and create
    the movie.
    """

    parser = argparse.ArgumentParser(
        description="A spring and mass simulator.")
    parser.add_argument("--xposition",
                        type=float,
                        required=True,
                        help="The initial position of the particle in "
                             "the x-axis in m.",
                        )
    parser.add_argument("--yposition",
                        type=float,
                        required=True,
                        help="The initial position of the particle in "
                             "the y-axis in m.",
                        )
    parser.add_argument("--xvelocity",
                        type=float,
                        required=True,
                        help="The initial velocity of the particle in "
                             "the x-axis in m/s.",
                        )
    parser.add_argument("--yvelocity",
                        type=float,
                        required=True,
                        help="The initial velocity of the particle in "
                             "the y-axis in m/s.",
                        )
    parser.add_argument("--mass",
                        type=float,
                        required=True,
                        help="The mass of the particle in kg.",
                        )
    parser.add_argument("--elastic_constant",
                        type=float,
                        required=True,
                        help="The elastic constant of the spring in N/m.",
                        )
    parser.add_argument("--natural_length",
                        type=float,
                        required=True,
                        help="The natural length of the spring in m.",
                        )
    parser.add_argument("--timestep",
                        type=float,
                        required=True,
                        help="The timestep of the simulation.")
    parser.add_argument("--steps",
                        type=int,
                        required=True,
                        help="The number of steps to simulate.")
    parser.add_argument("--filename",
                        type=str,
                        required=False,
                        default="movie",
                        help="The name of the output movie.")
    parser.add_argument("--fps",
                        type=float,
                        required=True,
                        help="The FPS of the movie.")
    parser.add_argument("--gravity",
                        type=int,
                        required=False,
                        default=0.0,
                        help="The gravitational constant g in m/s^2.")
    args = parser.parse_args()

    pos0 = (args.xposition, args.yposition)
    vel0 = (args.xvelocity, args.yvelocity)

    df = simulate(mass=args.mass,
                  pos0=pos0,
                  vel0=vel0,
                  elastic_constant=args.elastic_constant,
                  natural_length=args.natural_length,
                  gravity=args.gravity,
                  timestep=args.timestep,
                  n_steps=args.steps,
                  )
    generate_frames(df=df)
    movie(fps=args.fps, filename=args.filename)


if __name__ == "__main__":
    main()
