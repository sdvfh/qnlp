import pickle
from pathlib import Path

from custom_classifier import models
from pennylane import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from utils import get_embeddings, read_dataset

# Constants and Hyperparameters
LEVELS = ["easy", "medium", "hard"]
TYPES_DATASETS = ["train", "test"]
EPOCHS = 1
BATCH_SIZE = 5
N_REPETITIONS = 1

n_layers = 1
seed = 0

# Paths and random number generator
paths = {"data": Path(__file__).parent.parent / "data"}

# Load and embed dataset for each level
dfs = {level: read_dataset(paths["data"], level) for level in LEVELS}
dfs = get_embeddings(dfs, LEVELS, TYPES_DATASETS)
for level in LEVELS:
    x_train = np.array(dfs[level]["train"]["embeddings"], requires_grad=False)
    y_train = np.array(dfs[level]["train"]["targets"], requires_grad=False)
    x_test = np.array(dfs[level]["test"]["embeddings"], requires_grad=False)
    y_test = np.array(dfs[level]["test"]["targets"], requires_grad=False)

    for model in models:
        model_name = model.__name__
        for seed in range(N_REPETITIONS):
            filename = f"{model_name}_{level}_{n_layers}_{seed}.pkl"
            print(
                f"Running {model_name} for {level} level with {n_layers} layers and seed {seed}."
            )
            model = model(
                n_layers=n_layers,
                max_iter=EPOCHS,
                batch_size=BATCH_SIZE,
                random_state=seed,
            )
            model.fit(x_train, y_train)
            y_pred = model.predict_proba(x_test)[:, 1]

            # Metrics
            false_positive_rate, true_positive_rate, threshold = roc_curve(
                y_test, y_pred
            )
            y_prend_round = y_pred.round()
            metrics = {
                "accuracy": accuracy_score(y_test, y_prend_round),
                "f1": f1_score(y_test, y_prend_round),
                "precision": precision_score(y_test, y_prend_round),
                "recall": recall_score(y_test, y_prend_round),
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "threshold": threshold,
                "loss_curve": model.loss_curve_,
                "weights": model.weights_,
                "biases": model.bias_,
                "n_layers": model.n_layers,
                "epochs": model.max_iter,
                "batch_size": model.batch_size,
                "level": level,
                "seed": seed,
            }

# qvc2 = Ansatz1(n_layers=10, max_iter=1, batch_size=20, random_state=0)
# ensemble = AdaBoostClassifier(estimator=qvc1, n_estimators=10, random_state=0)
# ensemble = BaggingClassifier(
#     estimator=qvc1, n_estimators=2, random_state=0, max_features=0.8, max_samples=0.8
# )
# estimators = [("qvc1", qvc1), ("qvc2", qvc2)]
# ensemble = VotingClassifier(estimators=estimators, voting="soft")
# ensemble = VotingClassifier(estimators=estimators, voting="hard")
