import argparse
import sys
from pathlib import Path

from datasets import read_dataset
from joblib import Parallel, delayed
from models import get_model_classifier
from utils import compute_metrics, get_args_hash

import wandb


def run(args, args_hash, config, seed, x_train, y_train, x_test, y_test, testing):
    config["seed"] = seed
    if not testing:
        wandb.init(entity="svf", project="qnlp2", group=args_hash, config=config)
    print(
        f"Running {args.model_classifier} for "
        f"{args.dataset!r} dataset with "
        f"{args.model_transformer} model transformer, "
        f"{args.n_layers} layers and "
        f"seed {seed}."
    )

    model = get_model_classifier(args, seed)
    model.fit(x_train, y_train)
    model_has_proba = True
    try:
        y_pred = model.predict_proba(x_test)
    except AttributeError:
        y_pred = model.predict(x_test)
        model_has_proba = False
    if not testing:
        model.save(y_pred, model_has_proba=model_has_proba)
    compute_metrics(y_test, y_pred, testing, model_has_proba=model_has_proba)
    if not testing:
        wandb.finish()


def run_already_logged(project, config_hash):
    api = wandb.Api(timeout=60)
    runs = api.runs(project, filters={"config.hash": config_hash})
    return len(runs) > 0


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
        default=10,
    )

    parser.add_argument("--testing", action="store_true", default=False)

    args = parser.parse_args()

    config = vars(args)
    testing = config.pop("testing")
    args_hash = get_args_hash(args)
    args.testing = testing
    config["hash"] = args_hash

    if not testing:
        if run_already_logged("svf/qnlp2", args_hash):
            print("Skipping registered run.")
            sys.exit(0)

    # TODO: make verification on all parameters

    paths = {"root": Path(__file__).parent.parent.resolve()}
    paths["data"] = paths["root"] / "data"

    x_train, y_train, x_test, y_test = read_dataset(
        args.dataset, args.model_transformer, args.n_features, paths
    )

    n_jobs = 1 if testing else int(args.n_repetitions)

    Parallel(n_jobs=n_jobs)(
        delayed(run)(
            args, args_hash, config, seed, x_train, y_train, x_test, y_test, testing
        )
        for seed in range(args.n_repetitions)
    )
