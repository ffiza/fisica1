import os
import shutil
import pandas as pd


def create_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(os.path.dirname(path), exist_ok=False)


def get_particle_count_from_df(df: pd.DataFrame) -> int:
    return df.filter(regex="xPosition\\d+").shape[1]
