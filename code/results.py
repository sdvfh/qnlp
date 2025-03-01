import csv
import pickle
from pathlib import Path

import pandas as pd

root_path: Path = Path(__file__).parent.parent.resolve()
data_path: Path = root_path / "data"
results_path: Path = root_path / "results"
exp_name = "mpnet"
n_qubits = 10
truncate_dim = 32
dir_path = results_path / exp_name / str(n_qubits) / str(truncate_dim)

file_results_path = results_path.rglob("*.pkl")

results = []

for file in file_results_path:
    with open(file, "rb") as f:
        metrics = pickle.load(f)
        metrics["file"] = file.stem
        results.append(metrics)

results = pd.DataFrame(results)
results["model"] = results["file"].str.split("_").str[0]

metrics = results.groupby(["n_layers", "model", "level"]).agg(
    {
        "accuracy": ["mean", "std"],
        "f1": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
    }
)
metrics.to_csv(
    dir_path / "metrics.csv",
    index=True,
    encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL,
)
