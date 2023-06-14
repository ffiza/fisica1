"""
Title: A Two-Body Simulation


Examples of ICs
---------------
two-body-example1
    --velocity=3.162278
    --angle=110.0
    --mass=6.0
    --distance=6.0
    --timestep=0.01
    --steps=1500
    --show_reference="no"
    --fps=100
    --filename="two-body-example1"
two-body-example2
    --velocity=3.162278
    --angle=225.0
    --mass=6.0
    --distance=6.0
    --timestep=0.01
    --steps=1500
    --show_reference="yes"
    --fps=100
    --filename="two-body-example2"
two-body-example3
    --velocity=2.0
    --angle=110.0
    --mass=6.0
    --distance=6.0
    --timestep=0.01
    --steps=1500
    --show_reference="no"
    --fps=100
    --filename="two-body-example3"
two-body-example4
    --velocity=1.0
    --angle=45.0
    --mass=6.0
    --distance=6.0
    --timestep=0.01
    --steps=1500
    --show_reference="no"
    --fps=100
    --filename="two-body-example4"
two-body-example5 (output one every 10 frames)
    --velocity=1.0
    --angle=45.0
    --mass=6.0
    --distance=6.0
    --timestep=0.001
    --steps=15000
    --show_reference="no"
    --fps=100
    --filename="two-body-example5"
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

GRAV_CONST: float = 10.0  # m^3 / kg / s^2


def simulate(m1: float,
             m2: float,
             r1_initial: Tuple[float, float],
             r2_initial: Tuple[float, float],
             v1_initial: Tuple[float, float],
             v2_initial: Tuple[float, float],
             timestep: float,
             n_steps: int,
             ) -> pd.DataFrame:
    """
    Perform the two-body simulation.

    Parameters
    ----------
    m1 : float
        The mass of particle 1 in kg.
    m2 : float
        The mass of particle 2 in kg.
    r1_initial : Tuple[float, float]
        The initial position in the x- and y- axis of particle 1 in m.
    r2_initial : Tuple[float, float]
        The initial position in the x- and y- axis of particle 2 in m.
    v1_initial : Tuple[float, float]
        The initial velocity in the x- and y- axis of particle 1 in m/s.
    v2_initial : Tuple[float, float]
        The initial velocity in the x- and y- axis of particle 2 in m/s.
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

    r1_initial = np.array(r1_initial)
    r2_initial = np.array(r2_initial)
    v1_initial = np.array(v1_initial)
    v2_initial = np.array(v2_initial)

    r1 = np.zeros((n_steps, 2))
    r2 = np.zeros((n_steps, 2))
    v1 = np.zeros((n_steps, 2))
    v2 = np.zeros((n_steps, 2))

    r1[0] = r1_initial
    r2[0] = r2_initial

    v1[0] = v1_initial
    v2[0] = v2_initial

    for i in range(1, n_steps):
        dr = r2[i - 1] - r1[i - 1]
        distance = np.linalg.norm(dr)
        force = - GRAV_CONST * m1 * m2 / distance**3 * dr

        a1 = - force / m1
        a2 = + force / m2

        # Use the semi-implicit Euler method
        v1[i] = v1[i - 1] + a1 * timestep
        v2[i] = v2[i - 1] + a2 * timestep
        r1[i] = r1[i - 1] + v1[i] * timestep
        r2[i] = r2[i - 1] + v2[i] * timestep

    distance = np.linalg.norm(r2 - r1, axis=1)

    df = pd.DataFrame({"xpos1": r1[:, 0], "ypos1": r1[:, 1],
                       "xpos2": r2[:, 0], "ypos2": r2[:, 1],
                       "xvel1": v1[:, 0], "yvel1": v1[:, 1],
                       "xvel2": v2[:, 0], "yvel2": v2[:, 1],
                       "potential": - GRAV_CONST * m1 * m2 / distance,
                       "kinetic1": 0.5 * m1 * np.linalg.norm(v1, axis=1)**2,
                       "kinetic2": 0.5 * m2 * np.linalg.norm(v2, axis=1)**2,
                       })
    df["energy"] = df["potential"] + df["kinetic1"] + df["kinetic2"]

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
          show_reference: bool
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
    show_reference : bool
        Whether to draw reference cirlces or not.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.set_facecolor("#010001")
    ax.set_xlim((-22, 18))
    ax.set_ylim((-12, 12))
    ax.set_xticks([])
    ax.set_yticks([])

    if show_reference:  # Plot reference circles
        d0 = np.abs(df["xpos1"].loc[0] - df["xpos2"].loc[0])
        for diameter in [0.5 * d0, d0, 1.5 * d0]:
            circle = plt.Circle((0, 0), diameter / 2,
                                color='white',
                                alpha=0.3,
                                lw=0.5,
                                fill=False)
            ax.add_patch(circle)

    # Plot energy bars
    energies = df[["potential", "kinetic1", "kinetic2", "energy"]].to_numpy()
    max_energy = np.max(np.abs(energies))
    factor = 10 / max_energy

    rectangles = []
    rectangles.append(plt.Rectangle(xy=(-21, 0),
                                    width=1,
                                    height=df["energy"].loc[idx] * factor,
                                    color="#fa4656"))
    rectangles.append(plt.Rectangle(xy=(-19, 0),
                                    width=1,
                                    height=df["potential"].loc[idx] * factor,
                                    color="#a241b2"))
    rectangles.append(plt.Rectangle(xy=(-17, 0),
                                    width=1,
                                    height=df["kinetic1"].loc[idx] * factor,
                                    color="#2c73d6"))
    rectangles.append(plt.Rectangle(xy=(-15, 0),
                                    width=1,
                                    height=df["kinetic2"].loc[idx] * factor,
                                    color="#00d75b"))
    for rectangle in rectangles:
        ax.add_patch(rectangle)

    # Fix the energy label to avoid changes due to numerical noise
    if np.mean(df["energy"]) <= 0:
        label_props = (0.5, "bottom")
    else:
        label_props = (-0.5, "top")
    ax.annotate(text=r"$\mathbf{E}$",
                xy=(-20.5, label_props[0]),
                ha="center",
                va=label_props[1],
                fontsize=12,
                weight="bold",
                color="#fa4656")
    ax.annotate(text=r"$\mathbf{U}$",
                xy=(-18.5, 0.5),
                ha="center",
                va="bottom",
                fontsize=12,
                weight="bold",
                color="#a241b2")
    ax.annotate(text=r"$\mathbf{K}_1$",
                xy=(-16.5, -0.5),
                ha="center",
                va="top",
                fontsize=12,
                weight="bold",
                color="#2c73d6")
    ax.annotate(text=r"$\mathbf{K}_2$",
                xy=(-14.5, -0.5),
                ha="center",
                va="top",
                fontsize=12,
                weight="bold",
                color="#00d75b")

    # Plot the trail of each particle
    lc = multi_width_line(x=df["xpos1"].to_numpy()[:idx + 1],
                          y=df["ypos1"].to_numpy()[:idx + 1],
                          color="#2c73d6")
    ax.add_collection(lc)

    lc = multi_width_line(x=df["xpos2"].to_numpy()[:idx + 1],
                          y=df["ypos2"].to_numpy()[:idx + 1],
                          color="#00d75b")
    ax.add_collection(lc)

    # Plot the particles as circles
    ax.scatter(x=df["xpos1"].loc[idx],
               y=df["ypos1"].loc[idx],
               color="#2c73d6",
               s=40.0)
    ax.scatter(x=df["xpos2"].loc[idx],
               y=df["ypos2"].loc[idx],
               color="#00d75b",
               s=40.0)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(frame_path, dpi=600, pad_inches=0, bbox_inches="tight")
    plt.close()


def generate_frames(df: pd.DataFrame, show_reference: bool):
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
              show_reference=show_reference)


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
        description="A two-body gravity simulator")
    parser.add_argument("--velocity",
                        type=float,
                        required=True,
                        help="The absolute value of the initial velocity of "
                             "both particles in m/s.",
                        )
    parser.add_argument("--angle",
                        type=float,
                        required=True,
                        help="The angle of the initial velocity of both "
                             "particles in degrees.",
                        )
    parser.add_argument("--mass",
                        type=float,
                        required=True,
                        help="The mass of both particles in kg.",
                        )
    parser.add_argument("--distance",
                        type=float,
                        required=True,
                        help="The initial distance between particles in m.",
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
    parser.add_argument("--show_reference",
                        type=str,
                        required=True,
                        default="no",
                        choices=["yes", "no"],
                        help="Whether to show or not reference circles.")
    parser.add_argument("--fps",
                        type=int,
                        required=True,
                        help="The FPS of the movie.")
    args = parser.parse_args()

    df = simulate(m1=args.mass,
                  m2=args.mass,
                  r1_initial=(args.distance / 2, 0),
                  r2_initial=(-args.distance / 2, 0),
                  v1_initial=(args.velocity * np.cos(np.radians(args.angle)),
                              args.velocity * np.sin(np.radians(args.angle))),
                  v2_initial=(-args.velocity * np.cos(np.radians(args.angle)),
                              -args.velocity * np.sin(np.radians(args.angle))),
                  timestep=args.timestep,
                  n_steps=args.steps,
                  )
    show_reference = True if args.show_reference == "yes" else False
    generate_frames(df=df,
                    show_reference=show_reference)
    movie(fps=args.fps, filename=args.filename)


if __name__ == "__main__":
    main()
