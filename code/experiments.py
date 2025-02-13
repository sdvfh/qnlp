import operator
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
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


def quantum_circuit_fn(weights: np.ndarray, x: np.ndarray) -> float:
    """
    Implements the quantum circuit for variational classification.

    Args:
        weights (np.ndarray): The weights for the quantum gates.
        x (np.ndarray): The input features to be embedded into the quantum circuit.

    Returns:
        float: The expectation value of the PauliZ operator applied to the circuit.
    """
    wires = range(N_QUBITS)
    qml.AmplitudeEmbedding(features=x, wires=wires, normalize=True, pad_with=0.0)
    qml.StronglyEntanglingLayers(weights, wires=wires)
    # Combine the PauliZ observables for all wires
    result = reduce(operator.matmul, [qml.PauliZ(i) for i in wires])
    return qml.expval(result)


def variational_classifier(
    weights: np.ndarray, bias: float, x: np.ndarray
) -> np.ndarray:
    """
    Computes the output of the variational quantum classifier in a vectorized manner.

    Args:
        weights (np.ndarray): The weights for the quantum gates.
        bias (float): The bias term to be added to the circuit's output.
        x (np.ndarray): The input features (or batch of features) to be embedded into the quantum circuit.

    Returns:
        np.ndarray: The outputs of the variational quantum classifier.
    """
    return quantum_circuit(weights, x) + bias


def square_loss(
    labels: Union[np.ndarray, List[float]], predictions: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculates the mean squared error loss between the labels and predictions.

    Args:
        labels (np.ndarray or List[float]): The true labels.
        predictions (np.ndarray or List[float]): The predicted values.

    Returns:
        float: The mean squared error loss.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    return np.mean((labels - predictions) ** 2)


def cost(
    weights: np.ndarray, bias: float, x_batch: np.ndarray, y_batch: np.ndarray
) -> float:
    """
    Calculates the cost (mean squared error loss) for a batch of data using the variational quantum classifier.

    Args:
        weights (np.ndarray): The weights for the quantum gates.
        bias (float): The bias term to be added to the circuit's output.
        x_batch (np.ndarray): A batch of input features for the quantum circuit.
        y_batch (np.ndarray): The true labels for the batch.

    Returns:
        float: The mean squared error loss for the batch.
    """
    predictions = variational_classifier(weights, bias, x_batch)
    return square_loss(y_batch, predictions)


def accuracy(
    labels: Union[np.ndarray, List[float]], predictions: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculates the accuracy of the predictions compared to the true labels.

    Args:
        labels (np.ndarray or List[float]): The true labels.
        predictions (np.ndarray or List[float]): The predicted values.

    Returns:
        float: The accuracy expressed as the fraction of correct predictions.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    return float(np.mean(np.abs(labels - predictions) < 1e-5))


# Constants and Hyperparameters
LEVELS = ["easy", "medium", "hard"]
TYPES_DATASETS = ["train", "test"]
EPOCHS = 100
BATCH_SIZE = 5
N_QUBITS = 10
n_layers = 1

# Initialize the device (reassign the device in qnode if necessary)
device = qml.device("default.qubit", wires=N_QUBITS)
quantum_circuit = qml.QNode(quantum_circuit_fn, device=device, interface="autograd")

# Paths and random number generator
paths = {"data": Path(__file__).parent.parent / "data"}
rng = np.random.default_rng(seed=0)

# Load and embed dataset for each level
dfs = {level: read_dataset(paths["data"], level) for level in LEVELS}
dfs = get_embeddings(dfs, LEVELS, TYPES_DATASETS)

# Use the "easy" level for training and testing in this example
x_train = np.array(dfs["easy"]["train"]["embeddings"], requires_grad=False)
y_train = np.array(dfs["easy"]["train"]["targets"], requires_grad=False)
x_test = np.array(dfs["easy"]["test"]["embeddings"], requires_grad=False)
y_test = np.array(dfs["easy"]["test"]["targets"], requires_grad=False)
len_train = len(y_train)

# Initialize weights and bias
weights_init = 0.01 * rng.normal(size=(n_layers, N_QUBITS, 3), requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

opt = NesterovMomentumOptimizer(0.01)

weights = weights_init
bias = bias_init

# Training loop
for epoch in range(1, EPOCHS + 1):
    indices = np.arange(len_train)
    rng.shuffle(indices)

    for start_idx in range(0, len_train, BATCH_SIZE):
        end_idx = start_idx + BATCH_SIZE
        batch_indices = indices[start_idx:end_idx]

        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]

        weights, bias, _, _ = opt.step(cost, weights, bias, x_batch, y_batch)

    # Compute test predictions in a vectorized manner
    y_pred = np.sign(variational_classifier(weights, bias, x_test))
    acc = accuracy(y_test, y_pred)

    print(
        "Epoch: {:5d} | Training Cost: {:0.7f} | Test Accuracy: {:0.7f}".format(
            epoch, cost(weights, bias, x_train, y_train), acc
        )
    )
