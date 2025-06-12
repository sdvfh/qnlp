from __future__ import annotations

import json
import time
from pathlib import Path

import wandb

API_PATH = "svf/qnlp2"
PER_PAGE = 25
SLEEP_BETWEEN = 0.4  # alivia rate-limit
OUT_DIR = Path(__file__).with_suffix("").parent / "runs_json"
OUT_DIR.mkdir(exist_ok=True)

api = wandb.Api(
    timeout=120
)  # timeout maior ajuda em links instáveis :contentReference[oaicite:4]{index=4}
runs = api.runs(API_PATH, per_page=PER_PAGE)

for run in runs:  # paginação natural
    json_path = OUT_DIR / f"{run.id}.json"
    if json_path.exists():  # já baixei? pula
        continue

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

    time.sleep(SLEEP_BETWEEN)
