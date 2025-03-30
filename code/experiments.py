import argparse

from pathlib import Path

from datasets import read_dataset
from utils import get_args_hash
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

import wandb



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments execution.")
    parser.add_argument("-dataset", type=str, required=True, help="Dataset name.")

    parser.add_argument(
        "-model_transformer", type=str, required=True, help="Transformer name."
    )

    parser.add_argument(
        "-n_features",
        type=int,
        required=True,
        help="Number of output features from model embedding.",
    )

    parser.add_argument(
        "-model_classifier", type=str, required=True, help="Model name."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        help="Number of epochs of QVC.",
        default=100,
    )

    parser.add_argument(
        "--batch_size", type=int, required=False, help="Batch size of QVC.", default=5
    )

    parser.add_argument(
        "--n_repetitions",
        type=int,
        required=False,
        help="Number of repetitions of QVC.",
        default=30,
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        required=False,
        help="Number of layers of QVC.",
        default=1,
    )

    parser.add_argument(
        "--n_qubits",
        type=int,
        required=False,
        help="Number of qubits of QVC.",
        default=1,
    )

    args = parser.parse_args()

    # TODO: make verification on all parameters

    paths = {"root": Path(__file__).parent.parent.resolve()}
    paths["data"] = paths["root"] / "data"

    x_train, y_train, x_test, y_test = read_dataset(
        args.dataset, args.model_transformer, args.n_features, paths
    )

    args_hash = get_args_hash(args)

    for seed in range(args.n_repetitions):
        wandb.init(entity="svf", project="qnlp", group=args_hash)
        print(
            f"Running {args.model_classifier} for "
            f'"{args.dataset}" dataset with '
            f"{args.n_layers} layers and "
            f"seed {seed}."
        )


        model = model_class(
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
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": seed,
        "y_pred": y_pred,
        "dataset": args.dataset,
        "model_transformer": args.model_transformer,
        "model_classifier": args.model_classifier,
        "n_features": args.n_features,
        "n_qubits": args.n_qubits,
        "n_repetitions": args.n_repetitions,
    }
