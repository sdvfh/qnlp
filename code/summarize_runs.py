import pickle
from pathlib import Path

import pandas as pd


def read_summary(results_path):
    with open(str(results_path / "summary.pkl"), "rb") as f:
        summary = pickle.load(f)

    new_summary = []
    for run in summary:
        current_run = {
            "project": run["project"],
            "entity": run["entity"],
            "id": run["id"],
            **run["config"],
            **run["summary"],
        }

        new_summary.append(current_run)

    df = pd.DataFrame(new_summary)
    return df


results_path = Path(__file__).parent.parent / "results"

datasets = ["chatgpt_easy", "chatgpt_medium", "chatgpt_hard"]
models = {
    "quantum": ["singlerotx", "singleroty", "singlerotz", "rot", "rotcnot"],
    "classic": [
        "svmrbf",
        "svmlinear",
        "svmpoly",
        "logistic",
        "randomforest",
        "knn",
        "mlp",
    ],
}
n_layers = [1, 10]
df = read_summary(results_path)

df = df.drop(
    columns=[
        "project",
        "entity",
        "epochs",
        "n_repetitions",
        "_step",
        "_timestamp",
        "loss",
        "id",
    ]
)

df = df.sort_values(by=df.columns.tolist())
df_grouped = df.groupby(
    [
        # "hash",
        "dataset",
        "n_layers",
        "n_qubits",
        "batch_size",
        "n_features",
        "model_classifier",
        "model_transformer",
    ]
).agg(
    {
        "_runtime": ["mean", "std"],
        "accuracy": ["mean", "std"],
        "f1": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "seed": "count",
    }
)
print(len(df_grouped[df_grouped["seed"]["count"] != 30]))
df.to_csv("all_runs.csv", index=False)
df_grouped.to_csv("grouped_runs.csv")
