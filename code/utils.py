from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from pennylane import numpy as np
from sentence_transformers import SentenceTransformer

# Define type aliases for clarity
DatasetType = Dict[str, Union[List[str], np.ndarray, Any]]
DataDict = Dict[str, DatasetType]


def read_dataset(data_path: Path, level: str) -> Dict[str, DatasetType]:
    """
    Reads dataset files for a specified difficulty level and returns a dictionary containing the data and labels.

    Args:
        data_path (Path): The base path to the dataset directory.
        level (str): The difficulty level of the dataset (e.g., "easy", "medium", "hard").

    Returns:
        Dict[str, DatasetType]: A dictionary with keys "train" and "test", each containing another dictionary with keys
        "data" and "targets".
    """
    datasets: Dict[str, DatasetType] = {}
    for dataset in ["train", "test"]:
        file_path = data_path / "chatgpt" / level / f"{dataset}.txt"
        data, targets = read_files(file_path)
        datasets[dataset] = {"data": data, "targets": targets}
    return datasets


def read_files(filename: Path) -> Tuple[List[str], np.ndarray]:
    """
    Reads data from a file and returns sentences along with their corresponding labels.

    Args:
        filename (Path): The path to the file to be read.

    Returns:
        Tuple[List[str], np.ndarray]: A tuple containing a list of sentences and a NumPy array of labels.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line in the file does not begin with an integer label.
        IOError: If an error occurs during file reading.
    """
    if not filename.exists():
        raise FileNotFoundError(f"The file {filename} does not exist.")

    try:
        with open(filename, "r") as file:
            lines = file.readlines()

        data_targets = []
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:  # Ignore empty lines
                continue
            if not stripped_line[0].isdigit():
                raise ValueError(
                    f"The line does not begin with an integer label: {line}"
                )
            label = int(stripped_line[0])
            text = stripped_line[1:].strip()
            data_targets.append((text, label))

        if not data_targets:
            raise ValueError(f"No valid data found in file {filename}.")

        data, targets = zip(*data_targets, strict=True)
    except IOError as e:
        raise IOError(
            f"An error occurred while reading the file {filename}: {e}"
        ) from None
    except ValueError as e:
        raise ValueError(f"Error processing file {filename}: {e}") from None

    return list(data), np.array(targets)


def get_embeddings(
    dfs: Dict[str, DataDict], levels: List[str], types_datasets: List[str]
) -> Dict[str, DataDict]:
    """
    Generates sentence embeddings for the dataset.

    Args:
        dfs (Dict[str, DataDict]): A dictionary containing the dataset with keys corresponding to difficulty levels and
        values as dictionaries with keys "train" and "test".
        levels (List[str]): A list of difficulty levels (e.g., ["easy", "medium", "hard"]).
        types_datasets (List[str]): A list of dataset types (e.g., ["train", "test"]).

    Returns:
        Dict[str, DataDict]: The updated dictionary with embeddings added to the dataset.
    """
    model = SentenceTransformer("all-mpnet-base-v2")
    for level in levels:
        for dataset in types_datasets:
            dfs[level][dataset]["embeddings"] = model.encode(
                dfs[level][dataset]["data"]
            )
    return dfs
