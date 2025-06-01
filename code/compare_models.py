import pickle
from pathlib import Path

import pandas as pd
from aeon.visualisation import plot_critical_difference


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
        if (
            (current_run["model_transformer"] == "all-mpnet-base-v2")
            and (current_run["model_classifier"] == "knn")
            and (current_run["n_features"] == 32)
            and (current_run["n_qubits"] == 5)
        ):
            continue
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
df = df[df["model_classifier"] != "rotcnot"]

datasets = [("chatgpt_easy", 5), ("chatgpt_medium", 5), ("chatgpt_hard", 5), ("sst", 4)]

for dataset_name, n_qubits in datasets:

    dataset = df[
        (df["dataset"] == dataset_name)
        & (df["model_transformer"] == "tomaarsen/mpnet-base-nli-matryoshka")
        & (df["n_qubits"] == n_qubits)
    ]

    dataset["model_name"] = (
        dataset["model_classifier"] + "_" + dataset["n_layers"].astype(str)
    )

    dataset = dataset.pivot(index="seed", columns="model_name", values="f1")

    fig, ax, pvalues = plot_critical_difference(
        dataset.values, dataset.columns, reverse=True, return_p_values=True, alpha=0.05
    )

    fig.set_size_inches(30, 30)
    fig.tight_layout()
    fig.show()
