import pickle
from pathlib import Path

import pandas as pd


def read_summary():
    with open(str(Path(__file__).parent.parent / "results" / "summary.pkl"), "rb") as f:
        summary = pickle.load(f)

    new_summary = []
    for run in summary:
        new_summary.append(
            {
                "project": run["project"],
                "entity": run["entity"],
                "id": run["id"],
                **run["config"],
                **run["summary"],
            }
        )

    df = pd.DataFrame(new_summary)
    return df


df = read_summary()
