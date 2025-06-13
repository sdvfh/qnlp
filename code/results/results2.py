"""
Scatter plot: entanglement × expressability × F1  –––  Y em ESCALA LOG

────────────────────────────────────────────────────────────────────────
• X-axis .......... entanglement             (média por circuito/n_layers)
• Y-axis (log) .... expressability           (média idem, clip 0 → ε)
• Colour (Viridis)  F1-score                 (média idem)
• Marker shape .... n_layers   (o = 1 layer │ ^ = 10 layers)
• Labels .......... top-K   e  bottom-K      F1 (evita poluição visual)
────────────────────────────────────────────────────────────────────────
Dependências extra:
    pip install adjustText
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
from utils import models, read_summary  # mapeamento & função existente

# ───────────────────────────── parâmetros ────────────────────────────
CSV_MEASURES = Path("../../results/measures_all.csv")  # métricas exp / ent
K_LABELS = 5  # rótulos extremos ±K
DATASET_FILTER = "chatgpt_easy"  # apenas esta base
EPS = 1e-3  # evita log(0)

# ──────────────────── 1. carregar & preparar dados ───────────────────
df_meas = pd.read_csv(CSV_MEASURES)
df_meas["model"] = df_meas["model"].replace(models)  # normaliza IDs

df_runs = read_summary()
df_runs = df_runs[df_runs["dataset"] == DATASET_FILTER]  # só chatgpt_easy

# ──────────────────── 2. combinar e agregar médias ───────────────────
df_full = df_meas.merge(
    df_runs[["model_classifier", "n_layers", "f1"]],
    left_on=["model", "n_layers"],
    right_on=["model_classifier", "n_layers"],
    how="inner",
).drop(columns="model_classifier")

df_avg = df_full.groupby(["model", "n_layers"], as_index=False, sort=False).agg(
    ent=("ent", "mean"), exp=("exp", "mean"), f1=("f1", "mean")
)

# ───────────── 3. escolher quais pontos receberão rótulo ─────────────
df_sorted = df_avg.sort_values("f1")
highlight_idx = pd.concat([df_sorted.head(K_LABELS), df_sorted.tail(K_LABELS)]).index

# ───────────────────── 4. configurar scatter ─────────────────────────
MARKERS = {1: "o", 10: "^"}
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
texts: list[plt.Text] = []

for n_layer, mk in MARKERS.items():
    sub = df_avg[df_avg["n_layers"] == n_layer]

    y_vals = sub["exp"].clip(lower=EPS)  # clip para log(0) não explodir
    sc = ax.scatter(
        sub["ent"],
        y_vals,
        c=sub["f1"],
        cmap="viridis",
        marker=mk,
        s=80,
        edgecolors="black",
        label=f"{n_layer} layer{'s' if n_layer > 1 else ''}",
        zorder=3,
    )

    # rótulos ±K
    for idx, row in sub.iterrows():
        if idx in highlight_idx:
            texts.append(
                ax.text(
                    row["ent"],
                    max(row["exp"], EPS),  # usa mesmo clip de y_vals
                    row["model"],
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    zorder=4,
                )
            )

# ───────────────────────── 5. estética fina ──────────────────────────
adjust_text(
    texts,
    expand_points=(1.2, 1.4),
    arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.5},
)

ax.set_yscale("log")  # ← escala log no eixo Y
ax.set_xlabel("Entanglement (média)")
ax.set_ylabel("Expressability (média, escala log)")
ax.set_title(
    f"Circuit landscape — média por ID / n_layers "
    f"(labels ±{K_LABELS} extremos de F1)"
)

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("F1-score (média)")

ax.legend(title="n_layers", frameon=True)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

plt.tight_layout()
plt.show()  # ou fig.savefig("scatter_logY.pdf", bbox_inches="tight")
