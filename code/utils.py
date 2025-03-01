from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from pennylane import numpy as np
from sentence_transformers import SentenceTransformer

# Define type aliases for clarity
DatasetType = Dict[str, Union[List[str], np.ndarray, Any]]
DataDict = Dict[str, DatasetType]


def read_dataset(data_path: Path, level: str) -> Dict[str, DatasetType]:
    """
    Reads dataset files for a specified difficulty level and returns a dictionary
    containing the data and corresponding labels.

    Args:
        data_path (Path): The base path to the dataset directory.
        level (str): The difficulty level of the dataset (e.g., "easy", "medium", "hard").

    Returns:
        Dict[str, DatasetType]: A dictionary with keys "train" and "test". Each value is a dictionary
        containing keys "data" (list of sentences) and "targets" (NumPy array of labels).
    """
    datasets: Dict[str, DatasetType] = {}
    for dataset in ["train", "test"]:
        file_path: Path = data_path / "chatgpt" / level / f"{dataset}.txt"
        data, targets = read_files(file_path)
        datasets[dataset] = {"data": data, "targets": targets}
    return datasets


def read_files(filename: Path) -> Tuple[List[str], np.ndarray]:
    """
    Reads data from a file and extracts sentences along with their corresponding labels.

    Args:
        filename (Path): The path to the file to be read.

    Returns:
        Tuple[List[str], np.ndarray]: A tuple where the first element is a list of sentences
        and the second element is a NumPy array of integer labels.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If a line in the file does not begin with an integer label.
        IOError: If an error occurs during file reading.
    """
    if not filename.exists():
        raise FileNotFoundError(f"The file {filename} does not exist.")

    try:
        with open(filename, "r") as file:
            lines: List[str] = file.readlines()

        data_targets: List[Tuple[str, int]] = []
        for line in lines:
            stripped_line: str = line.strip()
            if not stripped_line:  # Skip empty lines
                continue
            if not stripped_line[0].isdigit():
                raise ValueError(
                    f"The line does not start with an integer label: {line}"
                )
            label: int = int(stripped_line[0])
            text: str = stripped_line[1:].strip()
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
    dfs: Dict[str, DataDict],
    levels: List[str],
    types_datasets: List[str],
    truncate_dim: int,
    model_template_name: str,
) -> Dict[str, DataDict]:
    """
    Generates sentence embeddings for the dataset using a pre-trained Sentence Transformer model.

    This function iterates over each specified difficulty level and dataset type (e.g., "train" and "test"),
    computes sentence embeddings for the text data using the "all-mpnet-base-v2" model, and adds the resulting
    embeddings to the dataset dictionary.

    Args:
        dfs (Dict[str, DataDict]): A dictionary containing datasets keyed by difficulty levels. Each value is a
            dictionary with keys "train" and "test", each holding a dataset with textual data under the key "data".
        levels (List[str]): A list of difficulty levels (e.g., ["easy", "medium", "hard"]).
        types_datasets (List[str]): A list of dataset types (e.g., ["train", "test"]).

    Returns:
        Dict[str, DataDict]: The updated dictionary where each dataset has an additional key "embeddings"
        containing the computed sentence embeddings.
    """
    model = SentenceTransformer(
        model_template_name, truncate_dim=truncate_dim, trust_remote_code=True
    )
    for level in levels:
        for dataset in types_datasets:
            dfs[level][dataset]["embeddings"] = model.encode(
                dfs[level][dataset]["data"]
            )
    return dfs
