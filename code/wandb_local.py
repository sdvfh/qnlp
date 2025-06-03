import pickle
from pathlib import Path

import wandb

api = wandb.Api(timeout=60)
runs = api.runs("svf/qnlp2")

history = []
for run in runs:
    summary = dict(run.summary)
    if "_wandb" in summary:
        summary.pop("_wandb")
    if "roc_table" in summary:
        summary.pop("roc_table")
    history.append(
        {
            "project": run.project,
            "entity": run.entity,
            "id": run.id,
            "config": run.config,
            "summary": summary,
        }
    )

root = Path(__file__).parent.parent
dataset_folder = root / "results"

with open(str(dataset_folder / "summary.pkl"), "wb") as f:
    pickle.dump(history, f)
