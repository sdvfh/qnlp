import pickle
from pathlib import Path

import pandas as pd
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import friedmanchisquare, wilcoxon
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

# 3. Preprocessing methods
results = []
for dataset in datasets:
    for model_type in models:
        for model in models[model_type]:
            if model_type == "quantum":
                for n_layer in n_layers:
                    dims = []
                    for model_transformer in [
                        "all-mpnet-base-v2",
                        "tomaarsen/mpnet-base-nli-matryoshka",
                        "nomic-ai/nomic-embed-text-v1.5",
                    ]:
                        dim = df.loc[
                            (df["dataset"] == dataset)
                            & (df["model_transformer"] == model_transformer)
                            & (df["model_classifier"] == model)
                            & (df["n_features"] == 32)
                            & (df["n_qubits"] == 5)
                            & (df["n_layers"] == n_layer),
                            ["seed", "f1"],
                        ]
                        dim = dim.set_index("seed")
                        dim.columns = [f"f1_{model_transformer}"]
                        dims.append(dim)
                    dims = pd.concat(dims, axis=1).values
                    stats = friedmanchisquare(dims[:, 0], dims[:, 1], dims[:, 2])

                    results.append(
                        {
                            "dataset": dataset,
                            "model_classifier": model,
                            "n_layers": n_layer,
                            "friedman_statistics": stats.statistic,
                            "friedman_p_value": stats.pvalue,
                            "f1_mpnet": dims[:, 0].mean(),
                            "f1_matryoshka": dims[:, 1].mean(),
                            "f1_nomic": dims[:, 2].mean(),
                            "dims": dims,
                        }
                    )
            else:
                dims = []
                for model_transformer in [
                    "all-mpnet-base-v2",
                    "tomaarsen/mpnet-base-nli-matryoshka",
                    "nomic-ai/nomic-embed-text-v1.5",
                ]:
                    dim = df.loc[
                        (df["dataset"] == dataset)
                        & (df["model_transformer"] == model_transformer)
                        & (df["model_classifier"] == model)
                        & (df["n_features"] == 32)
                        & (df["n_qubits"] == 5),
                        ["seed", "f1"],
                    ]
                    dim = dim.set_index("seed")
                    dim.columns = [f"f1_{model_transformer}"]
                    dims.append(dim)
                dims = pd.concat(dims, axis=1).values
                stats = friedmanchisquare(dims[:, 0], dims[:, 1], dims[:, 2])

                results.append(
                    {
                        "dataset": dataset,
                        "model_classifier": model,
                        "friedman_statistics": stats.statistic,
                        "friedman_p_value": stats.pvalue,
                        "f1_mpnet": dims[:, 0].mean(),
                        "f1_matryoshka": dims[:, 1].mean(),
                        "f1_nomic": dims[:, 2].mean(),
                        "dims": dims,
                    }
                )

results = pd.DataFrame(results)
_, p_values_fixed, _, _ = multipletests(results["friedman_p_value"], method="holm")
results["friedman_p_value_fixed"] = p_values_fixed
results["friedman_shows_diff"] = results["friedman_p_value_fixed"] < 0.05

for i, row in results.iterrows():
    if row["friedman_shows_diff"]:
        nemenyi = posthoc_nemenyi_friedman(row["dims"])
        results.loc[i, "nemenyi_mpnet_matryoska_p_value"] = nemenyi.loc[0, 1]
        results.loc[i, "nemenyi_mpnet_nomic_p_value"] = nemenyi.loc[0, 2]
        results.loc[i, "nemenyi_matryoska_nomic_p_value"] = nemenyi.loc[1, 2]

results["nemenyi_mpnet_matryoska_shows_diff"] = (
    results["nemenyi_mpnet_matryoska_p_value"] < 0.05
)
results["nemenyi_mpnet_nomic_shows_diff"] = (
    results["nemenyi_mpnet_nomic_p_value"] < 0.05
)
results["nemenyi_matryoska_nomic_shows_diff"] = (
    results["nemenyi_matryoska_nomic_p_value"] < 0.05
)
results = results.drop(columns=["dims"])
results.to_csv(str(results_path / "preprocessing_nemenyi_p_value.csv"), index=False)
