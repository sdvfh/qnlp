from pathlib import Path

import numpy as np


def get_path() -> dict:
    path = {
        "root": Path(__file__).parent.parent.parent,
    }
    path["code"] = path["root"] / "code"
    path["data"] = path["root"] / "data"
    return path


def transform_labels(y: np.array):
    return np.where(y == 1, 1, 0)


WORD_COUNT_MIN = 30
N_REPETITIONS = 30
N_FEATURES = 8
