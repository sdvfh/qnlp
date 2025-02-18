"""
This module executes parallel model evaluations over multiple datasets and random seeds.
It reads datasets, obtains embeddings, trains each model, computes evaluation metrics,
and saves the results to disk.
"""

import pickle
from itertools import product
from pathlib import Path
from typing import Any, Dict, Type

from custom_classifier import models
from joblib import Parallel, delayed
from pennylane import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from utils import get_embeddings, read_dataset


def run_model_for_seed(
    model: Type[Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    level: str,
    n_layers: int,
    seed: int,
    results_path: Path,
    epochs: int,
    batch_size: int,
) -> None:
    """
    Execute the given model with a specific random seed and save its performance metrics.

    This function instantiates the model with the provided parameters, fits it on the training data,
    predicts probabilities on the test data, computes various evaluation metrics, and saves these metrics
    to a file in the results' directory.

    Args:
        model (Type[Any]): The classifier model class.
        x_train (np.ndarray): The training feature matrix.
        y_train (np.ndarray): The training target vector.
        x_test (np.ndarray): The testing feature matrix.
        y_test (np.ndarray): The testing target vector.
        level (str): The dataset difficulty level (e.g., "easy", "medium", "hard").
        n_layers (int): The number of layers used in the model.
        seed (int): The random seed for reproducibility.
        results_path (Path): The directory where the results will be saved.
        epochs (int): The number of training epochs.
        batch_size (int): The size of the training batches.

    Returns:
        None
    """
    model_name: str = model.__name__
    filename: str = f"{model_name}_{level}_{n_layers}_{seed}.pkl"
    file_path: Path = results_path / filename

    if file_path.exists():
        return

    print(
        f"Running {model_name} for {level} level with {n_layers} layers and seed {seed}."
    )

    # Initialize and train the model
    model_instance = model(
        n_layers=n_layers,
        max_iter=epochs,
        batch_size=batch_size,
        random_state=seed,
    )
    model_instance.fit(x_train, y_train)
    y_pred = model_instance.predict_proba(x_test)[:, 1]

    # Compute ROC curve and evaluation metrics
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    y_pred_round = y_pred.round()
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_test, y_pred_round),
        "f1": f1_score(y_test, y_pred_round),
        "precision": precision_score(y_test, y_pred_round),
        "recall": recall_score(y_test, y_pred_round),
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "thresholds": thresholds,
        "loss_curve": getattr(model_instance, "loss_curve_", None),
        "weights": getattr(model_instance, "weights_", None),
        "biases": getattr(model_instance, "bias_", None),
        "n_layers": getattr(model_instance, "n_layers", n_layers),
        "epochs": getattr(model_instance, "max_iter", epochs),
        "batch_size": getattr(model_instance, "batch_size", batch_size),
        "level": level,
        "seed": seed,
        "y_pred": y_pred,
    }

    # Save the computed metrics to a file
    with open(file_path, "wb") as f:
        pickle.dump(metrics, f)


def main() -> None:
    """
    Main function to execute parallel experiments on different datasets and models.

    This function loads datasets for multiple difficulty levels, computes embeddings,
    and runs several model evaluations in parallel using different random seeds.
    The evaluation metrics for each model and seed are saved to disk.
    """
    # Constants and hyperparameters
    levels = ["easy", "medium", "hard"]
    type_datasets = ["train", "test"]
    epochs = 100
    batch_size = 5
    n_repetitions = 30
    n_layers_list = [1, 10]

    # Define paths for data and results
    root_path: Path = Path(__file__).parent.parent.resolve()
    data_path: Path = root_path / "data"
    results_path: Path = root_path / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    # Load datasets and compute embeddings for each level
    datasets = {level: read_dataset(data_path, level) for level in levels}
    datasets = get_embeddings(datasets, levels, type_datasets)

    # Iterate over each number of layers and difficulty level to run model evaluations in parallel
    for n_layer in n_layers_list:
        for level in levels:
            x_train = np.array(
                datasets[level]["train"]["embeddings"], requires_grad=False
            )
            y_train = np.array(datasets[level]["train"]["targets"], requires_grad=False)
            x_test = np.array(
                datasets[level]["test"]["embeddings"], requires_grad=False
            )
            y_test = np.array(datasets[level]["test"]["targets"], requires_grad=False)

            Parallel(n_jobs=-1)(
                [
                    delayed(run_model_for_seed)(
                        model,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        level,
                        n_layer,
                        seed,
                        results_path,
                        epochs,
                        batch_size,
                    )
                    for model, seed in product(models, range(n_repetitions))
                ]
            )


if __name__ == "__main__":
    main()
