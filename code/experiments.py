from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def read_dataset(data_path, level):
    """
    Read dataset files for a given level and return a dictionary containing the data and targets.

    Args:
        data_path (Path): The base path to the dataset directory.
        level (str): The difficulty level of the dataset (e.g., "easy", "medium", "hard").

    Returns:
        dict: A dictionary with keys "train" and "test", each containing another dictionary with keys "data" and "targets".
    """
    datasets = {}
    for dataset in ["train", "test"]:
        data, targets = read_files(data_path / "chatgpt" / level / (dataset + ".txt"))
        datasets[dataset] = {"data": data, "targets": targets}
    return datasets


def read_files(filename: Path) -> Tuple[List[str], np.ndarray]:
    """
    Read data from a file and return sentences and targets.

    Args:
        filename (str): The path to the file to read.

    Returns:
        Tuple[List[str], np.ndarray]: A tuple containing a list of sentences and a numpy array of targets.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line in the file does not start with an integer label.
        IOError: If there is an error reading the file.
    """
    if not filename.exists():
        raise FileNotFoundError(f"The file {filename} does not exist.")

    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            data, targets = zip(
                *[
                    (line[1:].strip(), int(line[0]))
                    for line in lines
                    if line[0].isdigit()
                ],
                strict=True,
            )
    except IOError as e:
        raise IOError(
            f"An error occurred while reading the file {filename}: {e}"
        ) from None
    except ValueError as e:
        raise ValueError(f"Line does not start with an integer label: {e}") from None

    return list(data), np.array(targets)


levels = ["easy", "medium", "hard"]
types_datasets = ["train", "test"]
paths = {"data": Path(__file__).parent.parent / "data"}
dfs = {level: read_dataset(paths["data"], level) for level in levels}
model = SentenceTransformer("all-mpnet-base-v2")
for level in levels:
    for dataset in types_datasets:
        dfs[level][dataset]["embeddings"] = model.encode(dfs[level][dataset]["data"])
