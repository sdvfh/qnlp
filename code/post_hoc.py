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

# 1. Dimensionality reduction
results = []
for dataset in datasets:
    for model_type in models:
        for model in models[model_type]:
            if model_type == "quantum":
                for n_layer in n_layers:
                    dims = []
                    for n_features in [32, 768]:
                        dim = df.loc[
                            (df["dataset"] == dataset)
                            & (df["model_transformer"] == "all-mpnet-base-v2")
                            & (df["model_classifier"] == model)
                            & (df["n_features"] == n_features)
                            & (df["n_qubits"] == 10)
                            & (df["n_layers"] == n_layer),
                            ["seed", "f1"],
                        ]
                        dim = dim.set_index("seed")
                        dim.columns = [f"f1_{n_features}"]
                        dims.append(dim)
                    dims = pd.concat(dims, axis=1).values
                    stats = wilcoxon(dims[:, 0], dims[:, 1])

                    results.append(
                        {
                            "dataset": dataset,
                            "model_transformer": "all-mpnet-base-v2",
                            "model_classifier": model,
                            "n_qubits": 10,
                            "n_layers": n_layer,
                            "wilcoxon_statistics": stats.statistic,
                            "wilcoxon_p_value": stats.pvalue,
                            "f1_768": dims[:, 0].mean(),
                            "f1_32": dims[:, 1].mean(),
                        }
                    )
            else:
                dims = []
                for n_features in [32, 768]:
                    dim = df.loc[
                        (df["dataset"] == dataset)
                        & (df["model_transformer"] == "all-mpnet-base-v2")
                        & (df["model_classifier"] == model)
                        & (df["n_features"] == n_features),
                        ["seed", "f1"],
                    ]
                    dim = dim.set_index("seed")
                    dim.columns = [f"f1_{n_features}"]
                    dims.append(dim)
                dims = pd.concat(dims, axis=1).values
                stats = wilcoxon(dims[:, 0], dims[:, 1])

                results.append(
                    {
                        "dataset": dataset,
                        "model_transformer": "all-mpnet-base-v2",
                        "model_classifier": model,
                        "wilcoxon_statistics": stats.statistic,
                        "wilcoxon_p_value": stats.pvalue,
                        "f1_768": dims[:, 0].mean(),
                        "f1_32": dims[:, 1].mean(),
                    }
                )

results = pd.DataFrame(results)
_, p_values_fixed, _, _ = multipletests(results["wilcoxon_p_value"], method="holm")
results["wilcoxon_p_value_fixed"] = p_values_fixed
results["wilcoxon_shows_diff"] = results["wilcoxon_p_value_fixed"] < 0.05

results.to_csv(str(results_path / "features_wilcoxon_p_value_fixed.csv"), index=False)


# 2. Qubits reduction
results = []
for dataset in datasets:
    for model_type in models:
        for model in models[model_type]:
            if model_type == "quantum":
                for n_layer in n_layers:
                    dims = []
                    for n_qubits in [5, 10]:
                        dim = df.loc[
                            (df["dataset"] == dataset)
                            & (df["model_transformer"] == "all-mpnet-base-v2")
                            & (df["model_classifier"] == model)
                            & (df["n_features"] == 32)
                            & (df["n_qubits"] == n_qubits)
                            & (df["n_layers"] == n_layer),
                            ["seed", "f1"],
                        ]
                        dim = dim.set_index("seed")
                        dim.columns = [f"f1_{n_qubits}"]
                        dims.append(dim)
                    dims = pd.concat(dims, axis=1).values
                    stats = wilcoxon(dims[:, 0], dims[:, 1])

                    results.append(
                        {
                            "dataset": dataset,
                            "model_transformer": "all-mpnet-base-v2",
                            "model_classifier": model,
                            "n_layers": n_layer,
                            "wilcoxon_statistics": stats.statistic,
                            "wilcoxon_p_value": stats.pvalue,
                            "f1_5": dims[:, 0].mean(),
                            "f1_10": dims[:, 1].mean(),
                        }
                    )

results = pd.DataFrame(results)
_, p_values_fixed, _, _ = multipletests(results["wilcoxon_p_value"], method="holm")
results["wilcoxon_p_value_fixed"] = p_values_fixed
results["wilcoxon_shows_diff"] = results["wilcoxon_p_value_fixed"] < 0.05

results.to_csv(str(results_path / "qubits_wilcoxon_p_value_fixed.csv"), index=False)
