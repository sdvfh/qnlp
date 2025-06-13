from __future__ import annotations

import json
import time
from pathlib import Path

import wandb

API_PATH = "svf/qnlp2"
PER_PAGE = 1_000
SLEEP_BETWEEN = 60
RESULTS_DIR = Path(__file__).with_suffix("").parent.parent / "results"
OUT_DIR = RESULTS_DIR / "runs_json"
OUT_DIR.mkdir(exist_ok=True)

api = wandb.Api(timeout=1_200)
runs = api.runs(API_PATH, per_page=PER_PAGE)

for i, run in enumerate(runs):
    json_path = OUT_DIR / f"{run.id}.json"
    if json_path.exists():
        print(f"Skipping {i}")
        continue
    print(i)

    summary = dict(run.summary)
    for key in ("_wandb", "roc_table"):
        summary.pop(key, None)

    payload = {
        "id": run.id,
        "entity": run.entity,
        "project": run.project,
        "config": dict(run.config),
        "summary": summary,
    }
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    if i % (PER_PAGE + 1) == 0:
        time.sleep(SLEEP_BETWEEN)


jsons = OUT_DIR.glob("*.json")
final_json = [json.loads(file.read_text()) for file in jsons]
with open(RESULTS_DIR / "runs.json", "w", encoding="utf-8") as fp:
    json.dump(final_json, fp, ensure_ascii=False, indent=2)
