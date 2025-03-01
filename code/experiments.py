"""
This module executes parallel model evaluations over multiple datasets and random seeds.
It leverages joblib for parallel processing, computes embeddings, trains models, evaluates metrics,
and persists results efficiently.
"""

import pickle
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Type

from joblib import Parallel, delayed
from pennylane import numpy as np
from qvc import models
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
    truncate_dim: int,
    n_qubits: int,
    exp_name: str,
) -> None:
    """Train and evaluate a model instance with specific parameters and random seed.

    Args:
        model: Model class to instantiate
        x_train: Training features with shape (n_samples, n_features)
        y_train: Training labels with shape (n_samples,)
        x_test: Test features with shape (n_samples, n_features)
        y_test: Test labels with shape (n_samples,)
        level: Dataset difficulty level ('easy', 'medium', 'hard')
        n_layers: Number of model layers
        seed: Random seed for reproducibility
        results_path: Base directory for saving results
        epochs: Number of training epochs
        batch_size: Training batch size
        truncate_dim: Embedding truncation dimension
        n_qubits: Number of qubits for quantum models
        exp_name: Experiment name for directory organization
    """
    model_name = model.__name__
    filename = f"{model_name}_{level}_{n_layers}_{seed}.pkl"
    dir_path = results_path / exp_name / str(n_qubits) / str(truncate_dim)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / filename

    if file_path.exists():
        return

    print(
        f"Running {model_name} for {level} level with {n_layers} layers and seed {seed}"
    )

    model_instance = model(
        n_layers=n_layers,
        max_iter=epochs,
        batch_size=batch_size,
        random_state=seed,
        n_qubits=n_qubits,
    )
    model_instance.fit(x_train, y_train)
    y_pred = model_instance.predict_proba(x_test)[:, 1]

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    y_pred_round = y_pred.round()

    metrics = {
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
        "n_layers": n_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "level": level,
        "seed": seed,
        "y_pred": y_pred,
    }

    with open(file_path, "wb") as f:
        pickle.dump(metrics, f)


def main() -> None:
    """Main execution flow for parallel model evaluation."""
    # Configuration
    levels: List[str] = ["easy", "medium", "hard"]
    model_mappings = {
        "mpnet": "all-mpnet-base-v2",
        "matryoshka": "tomaarsen/mpnet-base-nli-matryoshka",
        "nomic": "nomic-ai/nomic-embed-text-v1.5",
    }
    training_config = {
        "epochs": 100,
        "batch_size": 5,
        "n_repetitions": 30,
        "truncate_dim": 32,
        "n_layers_list": [1, 10],
        "n_qubits": 10,
        "exp_name": "mpnet",
    }

    # Path setup
    root_path = Path(__file__).parent.parent.resolve()
    data_path = root_path / "data"
    results_path = root_path / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    # Dataset processing
    datasets = {level: read_dataset(data_path, level) for level in levels}
    datasets = get_embeddings(
        datasets,
        levels,
        ["train", "test"],
        training_config["truncate_dim"],
        model_mappings[training_config["exp_name"]],
    )

    # Parallel execution
    for n_layers in training_config["n_layers_list"]:
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
                delayed(run_model_for_seed)(
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    level,
                    n_layers,
                    seed,
                    results_path,
                    training_config["epochs"],
                    training_config["batch_size"],
                    training_config["truncate_dim"],
                    training_config["n_qubits"],
                    training_config["exp_name"],
                )
                for model, seed in product(
                    models, range(training_config["n_repetitions"])
                )
            )


if __name__ == "__main__":
    main()
