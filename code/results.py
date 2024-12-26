import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_results() -> pd.DataFrame:
    """Read all results from the results folder and return a DataFrame."""
    result_files = Path("../results").rglob("*.pkl.gz")
    results_list = []
    for result_file in result_files:
        with gzip.open(result_file, "rb") as f:
            data = pickle.load(f)
        data["symbols"] = {symbol.name: symbol for symbol in data["symbols"]}
        results_list.append(data)
    return pd.DataFrame(results_list)


def aggregate_metrics(experiment: list) -> pd.Series:
    """Aggregates the metrics of a DataFrame row."""
    return pd.DataFrame(experiment).agg(["mean", "std"]).stack()


results = read_results()
agg_results = results.groupby(
    [
        "level",
        "ansatz",
        "dim_prepositional_phrase",
        "dim_noun",
        "dim_sentence",
        "n_layer",
        "batch_size",
        "epochs",
        "n_repetitions",
        "device",
    ]
).agg(list)
agg_results = agg_results[agg_results["seed"].apply(len) == 30]
metrics = agg_results["metrics"].apply(aggregate_metrics)
agg_results = pd.concat((agg_results, metrics), axis=1)
