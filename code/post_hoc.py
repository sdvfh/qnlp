import pickle
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


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


# datasets = ["chatgpt_easy", "chatgpt_medium", "chatgpt_hard"]
datasets = ["chatgpt_easy"]
models = {
    "quantum": [
        "singlerotx",
        "singleroty",
        "singlerotz",
        "rot",
        "rotcnot",
        "maouaki1",
        "maouaki6",
        "maouaki9",
        "maouaki15",
        "ent1",
        "ent2",
        "ent3",
        "ent4",
    ],
    "classical": [
        "svmrbf",
        "svmlinear",
        "svmpoly",
        "logistic",
        "randomforest",
        "knn",
        "mlp",
    ],
}
models_transformer = [
    "all-mpnet-base-v2",
    "tomaarsen/mpnet-base-nli-matryoshka",
    "all-mpnet-base-v2",
]
results_path = Path(__file__).parent.parent / "results"

df = read_summary(results_path)

dataset_name = datasets[0]

dataset = df[
    (df["dataset"] == dataset_name) & (df["model_transformer"] == "all-mpnet-base-v2")
]
dataset["model_name"] = (
    dataset["model_classifier"]
    + "_"
    + dataset["n_features"].astype(str)
    + "_"
    + dataset["n_layers"].astype(str)
)
dataset = dataset.pivot(index="seed", columns="model_name", values="f1")

n_features_comp = {}
for model in models["quantum"]:
    for n_layer in [1, 10]:
        values = {}
        for n_features in [16, 32, 768]:
            values[n_features] = dataset[
                model + "_" + str(n_features) + "_" + str(n_layer)
            ].tolist()
        results_multipletests = multipletests(
            [
                wilcoxon(values[768], values[32]).pvalue,
                wilcoxon(values[32], values[16]).pvalue,
                wilcoxon(values[768], values[16]).pvalue,
            ],
            method="holm",
        )
        n_features_comp[model + "_" + str(n_features) + "_" + str(n_layer)] = (
            results_multipletests[0]
        )
n_features_comp = pd.DataFrame(n_features_comp).T
n_features_comp.columns = ["768_32", "32_16", "768_16"]
print(n_features_comp)
