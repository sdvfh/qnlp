"""
Scatter plot: entanglement × expressability × F1

────────────────────────────────────────────────────────────────────────
• X-axis .......... entanglement             (média por circuito/n_layers)
• Y-axis .......... expressability           (média idem)
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
from utils import models, read_summary  # 𝐼mporta mapeamento e função existente

# ───────────────────────────── parâmetros ────────────────────────────
CSV_MEASURES = Path("../../results/measures_all.csv")  # métricas exp / ent
K_LABELS = 5  # rótulos extremos ±K
DATASET_FILTER = "chatgpt_easy"  # apenas esta base

# ──────────────────── 1. carregar & preparar dados ───────────────────
# Mede expressabilidade / emaranhamento
df_meas = pd.read_csv(CSV_MEASURES)
df_meas["model"] = df_meas["model"].replace(models)  # normaliza IDs

# Métricas de desempenho (F1) provenientes dos runs W&B
df_runs = read_summary()
df_runs = df_runs[df_runs["dataset"] == DATASET_FILTER]  # só chatgpt_easy

# ──────────────────── 2. combinar e agregar médias ───────────────────
# INNER JOIN → mantém apenas circuitos presentes em ambas as tabelas
df_full = df_meas.merge(
    df_runs[["model_classifier", "n_layers", "f1"]],
    left_on=["model", "n_layers"],
    right_on=["model_classifier", "n_layers"],
    how="inner",
).drop(columns="model_classifier")

# Agora calcula a média por (modelo, n_layers)
df_avg = df_full.groupby(["model", "n_layers"], as_index=False, sort=False).agg(
    ent=("ent", "mean"),
    exp=("exp", "mean"),
    f1=("f1", "mean"),
)

# ───────────── 3. escolher quais pontos receberão rótulo ─────────────
df_sorted = df_avg.sort_values("f1")
highlight_idx = pd.concat([df_sorted.head(K_LABELS), df_sorted.tail(K_LABELS)]).index

# ───────────────────── 4. configurar scatter ─────────────────────────
MARKERS = {1: "o", 10: "^"}  # shapes por layer
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
texts: list[plt.Text] = []

for n_layer, mk in MARKERS.items():
    sub = df_avg[df_avg["n_layers"] == n_layer]
    scatter = ax.scatter(
        sub["ent"],
        sub["exp"],
        c=sub["f1"],
        cmap="viridis",
        marker=mk,
        s=80,
        edgecolors="black",
        label=f"{n_layer} layer{'s' if n_layer > 1 else ''}",
        zorder=3,
    )

    # rótulos apenas para índices em highlight_idx
    for idx, row in sub.iterrows():
        if idx in highlight_idx:
            texts.append(
                ax.text(
                    row["ent"],
                    row["exp"],
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

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("F1-score (média)")

ax.set_xlabel("Entanglement (média)")
ax.set_ylabel("Expressability (média)")
ax.set_title(
    f"Circuit landscape — média por ID / n_layers "
    f"(labels ±{K_LABELS} extremos de F1)"
)
ax.legend(title="n_layers", frameon=True)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
plt.tight_layout()

plt.show()  # ou fig.savefig("scatter_selective_labels.pdf", bbox_inches="tight")
