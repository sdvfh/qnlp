"""
Scatter: entanglement × expressability × F1  (Y em escala log)

• Apenas o ponto (circuito 14, n_layers = 10) recebe rótulo “14”.
• Demais pontos aparecem sem texto.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy
from utils import models, read_summary


def place_label(
    ax,
    x,
    y,
    text,
    others_x,
    others_y,
    offsets=((10, -6), (8, 6), (-8, 6), (-8, -6), (8, 0), (-8, 0), (0, 8), (0, -8)),
    min_sep=10,
):
    """
    Coloca `text` próximo ao ponto (x,y) tentando evitar outros pontos.
    offsets  : lista de deslocamentos candidatos em *points* (dx, dy)
    min_sep  : separação mínima em *points* para considerar posição livre
    """
    # converte coords dos outros pontos para display coords
    xy_disp = ax.transData.transform(np.column_stack([others_x, others_y]))
    x_disp, y_disp = xy_disp[:, 0], xy_disp[:, 1]

    for dx, dy in offsets:
        txt_offset = offset_copy(
            ax.transData, fig=ax.figure, x=dx, y=dy, units="points"
        )
        # posição do texto em display coords
        x_t, y_t = txt_offset.transform((x, y))
        # distâncias para cada outro ponto em points
        d = np.hypot(x_disp - x_t, y_disp - y_t)
        if (d > min_sep).all():
            ax.annotate(
                text,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "gray",
                    "lw": 0.5,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
            )
            return
    # fallback: coloca no primeiro offset
    dx, dy = offsets[0]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=8,
        arrowprops={
            "arrowstyle": "-",
            "color": "gray",
            "lw": 0.5,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


# ─────────────────────────── parâmetros ──────────────────────────────
CSV_MEASURES = Path("../../results/measures_all.csv")
DATASET_FILTER = "chatgpt_easy"
EPS = 1e-3  # evita log(0)
ID_TO_LABEL = 14  # circuito a rotular
LAYER_TO_LABEL = 10  # layer correspondente

# ──────────────────────── carregar dados ─────────────────────────────
df_meas = pd.read_csv(CSV_MEASURES)
df_meas["model"] = df_meas["model"].replace(models)

df_runs = read_summary()
df_runs = df_runs[df_runs["dataset"] == DATASET_FILTER]

df_full = df_meas.merge(
    df_runs[["model_classifier", "n_layers", "f1"]],
    left_on=["model", "n_layers"],
    right_on=["model_classifier", "n_layers"],
    how="inner",
).drop(columns="model_classifier")

df_avg = df_full.groupby(["model", "n_layers"], as_index=False, sort=False).agg(
    ent=("ent", "mean"), exp=("exp", "mean"), f1=("f1", "mean")
)

# ─────────────────────────── scatter ─────────────────────────────────
MARKERS = {1: "o", 10: "^"}
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
texts = []

for n_layer, mk in MARKERS.items():
    sub = df_avg[df_avg["n_layers"] == n_layer]
    ax.scatter(
        sub["ent"],
        sub["exp"].clip(lower=EPS),
        c=sub["f1"],
        cmap="viridis",
        marker=mk,
        s=80,
        edgecolors="black",
        label=f"{n_layer} layer{'s' if n_layer > 1 else ''}",
        zorder=2,
    )

    mask_lbl = (df_avg["model"] == 14) & (df_avg["n_layers"] == 10)
    if mask_lbl.any():
        row = df_avg.loc[mask_lbl].iloc[0]
        x0, y0 = row["ent"], max(row["exp"], EPS)

        # arrays com as coordenadas dos OUTROS pontos
        others = df_avg[~mask_lbl]
        place_label(
            ax,
            x0,
            y0,
            "14",
            others["ent"].values,
            others["exp"].clip(lower=EPS).values,
            min_sep=20,
        )

# ───────────────────── estética geral ────────────────────────────────

ax.set_yscale("log")
ax.set_xlabel("Entanglement (média)")
ax.set_ylabel("Expressability (média, log)")
ax.set_title("Circuit landscape — chatgpt_easy")

sm = plt.cm.ScalarMappable(
    cmap="viridis", norm=plt.Normalize(df_avg["f1"].min(), df_avg["f1"].max())
)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("F1-score (média)")

ax.legend(title="n_layers", frameon=True)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

plt.tight_layout()
plt.show()  # ou fig.savefig("scatter_label14.pdf", bbox_inches="tight")
