from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from utils import models, read_summary


# ─────────────────────────  utilitário p/ rótulos ──────────────────────────
def place_label(
    ax,
    x,
    y,
    text,
    others_x,
    others_y,
    *,
    offsets,
    min_sep=1,
    fontsize=7,
):
    """
    Coloca 'text' perto de (x, y) tentando não colidir com outros pontos.
    Percorre a lista `offsets` (dx, dy em points) na ordem fornecida.
    """
    xy_disp = ax.transData.transform(np.column_stack([others_x, others_y]))
    xd, yd = xy_disp[:, 0], xy_disp[:, 1]

    for dx, dy in offsets:
        off = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
        xt, yt = off.transform((x, y))
        if (np.hypot(xd - xt, yd - yt) > min_sep).all():
            ax.annotate(
                text,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                ha="center",
                va="center",
            )
            return

    # fallback – desenha flecha fininha usando o 1.º deslocamento
    dx, dy = offsets[0]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": "gray",
            "lw": 0.5,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


# deslocamentos padrão (mantêm comportamento dos demais rótulos)
OFFSETS_DEFAULT = [(9, -5)]


# ───────────────────────────── dados ─────────────────────────────────
CSV_MEASURES = Path("../../results/measures_all.csv")
DATASET = "sst"
EPS = 1e-3

df_meas = pd.read_csv(CSV_MEASURES)
df_meas["model"] = df_meas["model"].replace(models)
df_meas = df_meas[df_meas["with_state_prep"]]

df_runs = read_summary()
df_runs = df_runs[df_runs["dataset"] == DATASET]

df = (
    df_meas.merge(
        df_runs[["model_classifier", "n_layers", "f1"]],
        left_on=["model", "n_layers"],
        right_on=["model_classifier", "n_layers"],
        how="inner",
    )
    .drop(columns="model_classifier")
    .groupby(["model", "n_layers"], as_index=False, sort=False)
    .agg(ent=("ent", "mean"), exp=("exp", "mean"), f1=("f1", "mean"))
)

# ─────────── intervalos e máscaras p/ insets ───────────
x1_min, x1_max = 0.784, 0.83  # zoom direita
y1_min, y1_max = 0.265e-2, 0.47e-2

x2_min, x2_max = 0.381, 0.3865  # zoom esquerda
y2_min, y2_max = 1.2e-2, 0.4e0

mask_inset1 = df["ent"].between(x1_min, x1_max) & df["exp"].between(y1_min, y1_max)
mask_inset2 = df["ent"].between(x2_min, x2_max) & df["exp"].between(y2_min, y2_max)

# rótulos no gráfico principal: tudo fora dos insets, exceto o triângulo 14
mask_main_labels = ~(mask_inset1 | mask_inset2)

# ─────────────────────── figura principal ───────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

MARKERS = {1: "o", 10: "^"}
for n_layer, mk in MARKERS.items():
    sub = df[df["n_layers"] == n_layer]
    ax.scatter(
        sub["ent"],
        sub["exp"].clip(lower=EPS),
        c=sub["f1"],
        cmap="viridis",
        marker=mk,
        s=80,
        edgecolors="black",
        label=f"{n_layer} camada{'s' if n_layer > 1 else ''}",
        zorder=2,
    )

# rótulos fora dos insets (padrão)
for idx, row in df[mask_main_labels].iterrows():
    others_x = df.loc[df.index != idx, "ent"].values
    others_y = df.loc[df.index != idx, "exp"].clip(lower=EPS).values
    place_label(
        ax,
        row["ent"],
        max(row["exp"], EPS),
        str(int(row["model"])),
        others_x,
        others_y,
        offsets=OFFSETS_DEFAULT,
        fontsize=7,
    )
# estética eixo principal
ax.set_yscale("log")
ax.set_xlabel("Emaranhamento (média)")
ax.set_ylabel("Expressabilidade (média, log)")
ax.set_title(f"Panorama das medidas — {DATASET}")
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
ax.legend(title="Legenda")

# colorbar
sm = plt.cm.ScalarMappable(
    cmap="viridis", norm=plt.Normalize(df["f1"].min(), df["f1"].max())
)
fig.colorbar(sm, ax=ax).set_label("F1-score (média)")

# ──────────────────────── inset 1 (zoom direita) ───────────────────────
axins = inset_axes(
    ax,
    width="120%",
    height="120%",
    bbox_to_anchor=(0.75, -0.65, 0.35, 0.35),
    bbox_transform=ax.transAxes,
    loc="lower left",
    borderpad=0,
)
axins.set_yscale("log")
axins.set_xlim(x1_min, x1_max)
axins.set_ylim(y1_min, y1_max)

for n_layer, mk in MARKERS.items():
    sub = df[df["n_layers"] == n_layer]
    axins.scatter(
        sub["ent"],
        sub["exp"].clip(lower=EPS),
        c=sub["f1"],
        cmap="viridis",
        marker=mk,
        s=80,
        edgecolors="black",
    )

# rótulos no zoom direita
sub1 = df[mask_inset1]
for idx, row in sub1.iterrows():
    others = sub1.drop(idx)
    place_label(
        axins,
        row["ent"],
        max(row["exp"], EPS),
        str(int(row["model"])),
        others["ent"].values,
        others["exp"].clip(lower=EPS).values,
        offsets=OFFSETS_DEFAULT,
        fontsize=7,
    )

mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="black", ls="--", lw=0.8)

# ──────────────────────── inset 2 (zoom esquerda) ───────────────────────
axins2 = inset_axes(
    ax,
    width="120%",
    height="120%",
    bbox_to_anchor=(0.05, -0.65, 0.35, 0.35),
    bbox_transform=ax.transAxes,
    loc="lower left",
    borderpad=0,
)
axins2.set_yscale("log")
axins2.set_xlim(x2_min, x2_max)
axins2.set_ylim(y2_min, y2_max)

for n_layer, mk in MARKERS.items():
    sub = df[df["n_layers"] == n_layer]
    axins2.scatter(
        sub["ent"],
        sub["exp"].clip(lower=EPS),
        c=sub["f1"],
        cmap="viridis",
        marker=mk,
        s=80,
        edgecolors="black",
    )

# rótulos no zoom esquerda
sub2 = df[mask_inset2]
for idx, row in sub2.iterrows():
    others = sub2.drop(idx)
    place_label(
        axins2,
        row["ent"],
        max(row["exp"], EPS),
        str(int(row["model"])),
        others["ent"].values,
        others["exp"].clip(lower=EPS).values,
        offsets=OFFSETS_DEFAULT,
        fontsize=7,
    )

mark_inset(ax, axins2, loc1=1, loc2=2, fc="none", ec="black", ls="--", lw=0.8)

# ───────────── layout – mantém espaço p/ insets ─────────────
fig.subplots_adjust(bottom=0.40, right=1)
plt.show()
