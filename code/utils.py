import hashlib
import json
import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb


def create_and_log_artifact(name, data, filename, artifact_type="data"):
    """Salva 'data' em um arquivo JSON, cria e loga um artifact no wandb, e remove o arquivo."""
    with open(filename, "w") as f:
        json.dump(data, f)
    artifact = wandb.Artifact(name, type=artifact_type)
    artifact.add_file(filename)
    wandb.log_artifact(artifact)
    os.remove(filename)


def get_args_hash(args):
    args_dict = vars(args)
    args_json = json.dumps(args_dict, sort_keys=True)
    return hashlib.md5(args_json.encode()).hexdigest()


def compute_metrics(model, y_test, y_pred):
    y_pred_round = y_pred[:, 1].round()

    weights = getattr(model, "weights_", None)
    bias = getattr(model, "bias_", None)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_round),
        "f1": f1_score(y_test, y_pred_round),
        "precision": precision_score(y_test, y_pred_round),
        "recall": recall_score(y_test, y_pred_round),
    }

    wandb.log({"roc": wandb.plot.roc_curve(y_test, y_pred)})

    wandb.log(metrics)

    if weights is not None:
        create_and_log_artifact("weights", weights.tolist(), "weights.json")

    if bias is not None:
        create_and_log_artifact("biases", {"bias": float(bias)}, "biases.json")

    create_and_log_artifact("y_pred", y_pred[:, 1].tolist(), "y_pred.json")
