import numpy as np
import os
import shutil
import pandas as pd
import numba


def create_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(os.path.dirname(path), exist_ok=False)


def get_particle_count_from_df(df: pd.DataFrame) -> int:
    return df.filter(regex="xPosition\\d+").shape[1]


@numba.njit
def calculate_mean_interparticle_distance(pos: np.ndarray) -> float:
    """
    This method calculates the mean inteparticle distance for a given list
    of particle positions. This method becomes very slow for large particle
    numbers.

    Parameters
    ----------
    pos : np.ndarray
        An array with the x-positions in the first column and the y-positions
        in the second column. This array should be of shape (N, 2), where
        N is the number of particles.

    Returns
    -------
    mid : float
        The mean interparticle distance.
    """
    n_part = pos.shape[0]
    n_dist = n_part * (n_part - 1) / 2

    mid = 0.0
    for i in numba.prange(n_part):
        for j in numba.prange(i + 1, n_part):
            mid += np.linalg.norm(pos[i] - pos[j]) / n_dist

    return mid
