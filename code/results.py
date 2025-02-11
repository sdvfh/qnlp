import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment import Experiment
from sklearn.metrics import roc_curve


def read_results() -> pd.DataFrame:
    """Read all results from the results folder and return a DataFrame."""
    result_files = Path("./results").rglob("*.pkl.gz")
    results_list = []
    for result_file in result_files:
        with gzip.open(result_file, "rb") as f:
            data = pickle.load(f)
        data["symbols"] = {key.name: value for key, value in data["symbols"].items()}
        results_list.append(data)
    return pd.DataFrame(results_list)


def aggregate_metrics(experiments: list) -> pd.Series:
    """Aggregates the metrics of a DataFrame row."""
    return pd.DataFrame(experiments).agg(["mean", "std"]).stack()


def get_loss_info(losses_info: list) -> list:
    """Aggregate loss and processing time information for each experiment."""
    aggregated_info = []
    for experiment_info in losses_info:
        epoch_data = {
            epoch: {
                "Processing time": sum(
                    batch["Processing time"] for batch in batches.values()
                ),
                "Loss": sum(batch["Loss"] for batch in batches.values()),
            }
            for epoch, batches in experiment_info.items()
        }
        # losses = [data["Loss"] for data in epoch_data.values()]
        processing_time = sum(data["Processing time"] for data in epoch_data.values())
        # aggregated_info.append({"losses": losses, "processing_time": processing_time})
        # return aggregated_info
        aggregated_info.append(processing_time)
    return {
        "mean": np.mean(aggregated_info) / 3600,
        "std": np.std(aggregated_info) / 3600,
    }


# def compute_roc_curve(experiments: list) -> dict:
#     """Compute the ROC curve for each experiment."""
#     for experiment in experiments:
#         data, targets = self.read_files(
#             f"../data/chatgpt/{self.level}/{dataset}.txt"
#         )
#         Experiment.read_files()
#         roc_curve()
#     return {
#         "false_rate": [roc["false_rate"] for roc in y_pred],
#         "true_rate": [roc["true_rate"] for roc in y_pred],
#         "thresholds": [roc["thresholds"] for roc in y_pred],
#     }

results = read_results()
results = results.drop(
    columns=[
        "batch_size",
        "epochs",
        "n_repetitions",
        "device",
        "experiment_id",
        "n_qubits_circuits",  # TODO study circuit's depth and width?
    ]
)

results = results.groupby(
    [
        "level",
        "ansatz",
        "dim_prepositional_phrase",
        "dim_noun",
        "dim_sentence",
        "n_layer",
    ]
).agg(list)
results = results[results["seed"].apply(len) == 30]
results = results.drop(columns="seed")
results = pd.concat((results, results["metrics"].apply(aggregate_metrics)), axis=1)
results = results.drop(columns="metrics")
results["epoch_infos"] = results["epoch_infos"].apply(get_loss_info)
symbols = results["symbols"].apply(aggregate_metrics).T
results = results.drop(columns="symbols")
# results["roc_curve"] = results.apply(compute_roc_curve)
