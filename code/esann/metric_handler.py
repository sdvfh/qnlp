import json

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .utils import get_path


class MetricHandler:
    def __init__(self):
        self.path = get_path()
        self.path["folder_root_metrics"] = self.path["data"] / "metrics"
        self.path["folder_root_metrics"].mkdir(parents=True, exist_ok=True)
        self.configs = {}

    def load(self, dataset_name, model_name, seed):
        self.configs = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "seed": seed,
        }
        self.path["file"] = (
            self.path["folder_root_metrics"]
            / f"{dataset_name}_{model_name}_{seed}.json"
        )

    def get_metrics(self, y_pred, y_true):
        return {
            "accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

    def save_metrics(self, metrics):
        json_to_save = {
            "dataset": self.configs["dataset_name"],
            "model": self.configs["model_name"],
            "seed": self.configs["seed"],
            "metrics": metrics,
        }
        with open(self.path["file"], "w") as file:
            json.dump(json_to_save, file)

    def metric_exists(self):
        return self.path["file"].exists()
