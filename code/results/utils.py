import json
from pathlib import Path

import pandas as pd


def read_summary():
    results_path = Path(__file__).parent.parent.parent / "results"
    with open(str(results_path / "runs.json"), "r") as f:
        summary = json.load(f)

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
    df["model_classifier"] = df["model_classifier"].replace(models)

    return df


models = {
    "singlerotx": 1,
    "singleroty": 2,
    "singlerotz": 3,
    "rot": 5,
    "rotcnot": 6,
    "maouaki1": 4,
    "maouaki6": 15,
    "maouakiquasi7": 12,
    "maouaki7": 13,
    "maouaki9": 8,
    "maouaki15": 14,
    "ent1": 7,
    "ent2": 10,
    "ent3": 11,
    "ent4": 9,
    "svmrbf": 38,
    "svmlinear": 36,
    "svmpoly": 37,
    "logistic": 32,
    "randomforest": 35,
    "knn": 34,
    "mlp": 33,
    "adaboost_rotcnot": 16,
    "bagging_rotcnot": 17,
    "adaboost_ent4": 18,
    "bagging_ent4": 19,
    "adaboost_maouaki15": 20,
    "bagging_maouaki15": 21,
    "soft_voting_1_2_3": 22,
    "hard_voting_1_2_3": 23,
    "soft_voting_1_2_3_5": 24,
    "hard_voting_1_2_3_5": 25,
    "soft_voting_1_2_3_5_6": 26,
    "hard_voting_1_2_3_5_6": 27,
    "soft_voting_7_8_9_10_11": 28,
    "hard_voting_7_8_9_10_11": 29,
    "soft_voting_12_14_15": 30,
    "hard_voting_12_14_15": 31,
    "adaboost_logistic": 39,
    "bagging_logistic": 40,
    "soft_voting_svm": 41,
    "hard_voting_svm": 42,
    "soft_voting_logistic_mlp_knn": 43,
    "hard_voting_logistic_mlp_knn": 44,
}

quantum_models = list(range(16))
quantum_ensemble_models = list(range(16, 32))
classical_models = list(range(32, 39))
classical_ensemble_models = list(range(39, 45))
