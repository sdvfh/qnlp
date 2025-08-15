from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np

import wandb

API_PATH = "svf/qnlp2"
PER_PAGE = 1_000
SLEEP_BETWEEN = 60
ROOT_DIR = Path(__file__).with_suffix("").parent.parent / "downloaded_models"
ROOT_DIR.mkdir(exist_ok=True)


def count_parameters(model):
    # LogisticRegression: coef_ + intercept_
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        return int(np.prod(model.coef_.shape) + np.prod(model.intercept_.shape))
    # SVC: number of support vectors
    if hasattr(model, "support_vectors_"):
        return int(model.support_vectors_.shape[0])
    # RandomForestClassifier: sum of node counts over all trees
    if hasattr(model, "estimators_"):
        return sum(est.tree_.node_count for est in model.estimators_)
    # KNeighborsClassifier: number of stored samples
    if hasattr(model, "n_samples_fit_"):
        return int(model.n_samples_fit_)
    # MLPClassifier: sum of all weights + biases
    if hasattr(model, "coefs_") and hasattr(model, "intercepts_"):
        total = 0
        for w in model.coefs_:
            total += np.prod(w.shape)
        for b in model.intercepts_:
            total += np.prod(b.shape)
        return int(total)
    return None


def download_and_summarize():
    api = wandb.Api(timeout=1_200)
    runs = api.runs(API_PATH, per_page=PER_PAGE)
    summary = {}

    for i, run in enumerate(runs):
        print(f"[{i + 1}/{len(runs)}] Processing run {run.id}...")
        # fetch all model artifacts logged by this run
        for art in run.logged_artifacts():
            if art.type != "model":
                continue
            local_path = art.download(root=ROOT_DIR)
            # look for .pkl files in the artifact folder
            for fname in os.listdir(local_path):
                if not fname.endswith(".pkl"):
                    continue
                model_file = os.path.join(local_path, fname)
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                n_params = count_parameters(model)
                summary[f"{run.id}:{art.name}"] = n_params

        # rate-limit to avoid API throttling
        if i and i % PER_PAGE == 0:
            time.sleep(SLEEP_BETWEEN)

    # print out the parameter counts
    for key, val in summary.items():
        print(f"{key} → {val} parameters")


if __name__ == "__main__":
    download_and_summarize()
