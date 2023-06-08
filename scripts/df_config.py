import numpy as np


def set_types(df):
    """
    This method configures the data types of the data frame.

    Parameters
    ----------
    df
        The data frame with student grades.
    """
    df["anio"] = df["anio"].astype(np.uint16)
    df["cuatrimestre"] = df["cuatrimestre"].astype(np.uint8)
    df["hash"] = df["hash"].astype("str")
    for grade in ["nota_p1", "nota_p2", "nota_r1", "nota_r1", "nota_final",
                  "p1_rozamiento", "p1_oscilaciones", "p1_sni"]:
        df[grade] = df[grade].astype(np.float16)
    for concept in ["concepto_p1"]:
        df[concept] = df[concept].astype("category")
