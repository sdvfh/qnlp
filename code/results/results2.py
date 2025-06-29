from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from utils import models, read_summary


# ─────────────────────────  utilitário rótulo “14” ──────────────────────────
def place_label(
    ax,
    x,
    y,
    text,
    others_x,
    others_y,
    offsets=((10, -6), (8, 6), (-8, 6), (-8, -6), (8, 0), (-8, 0), (0, 8), (0, -8)),
    min_sep=20,
):
    """Escolhe deslocamento que não colida com outros pontos."""
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
                fontsize=8,
                ha="center",
                va="center",
            )
            return

    # fallback
    dx, dy = offsets[0]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
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

# rótulo circuito 14, layer 10
mask14 = (df["model"] == 14) & (df["n_layers"] == 10)
if mask14.any():
    row = df.loc[mask14].iloc[0]
    place_label(
        ax,
        row["ent"],
        max(row["exp"], EPS),
        "14",
        df.loc[~mask14, "ent"].values,
        df.loc[~mask14, "exp"].clip(lower=EPS).values,
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

# ──────────────────────── inset externo (zoom 1) ───────────────────────
x_min, x_max = 0.77, 0.84
y_min, y_max = 0.20 * 1e-2, 0.6 * 1e-2  # [10⁻3, 10⁻2]

axins = inset_axes(
    ax,
    width="120%",
    height="120%",
    bbox_to_anchor=(0.75, -0.65, 0.35, 0.35),  # “fora” à direita-abaixo
    bbox_transform=ax.transAxes,
    loc="lower left",
    borderpad=0,
)
axins.set_yscale("log")
axins.set_xlim(x_min, x_max)
axins.set_ylim(y_min, y_max)

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

mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="black", ls="--", lw=0.8)

# ──────────── ALTERAÇÃO: rótulo da(s) bolinha(s) no inset ────────────
mask_inset_circle = (
    (df["n_layers"] == 1)  # bolinhas
    & (df["ent"].between(x_min, x_max))
    & (df["exp"].between(y_min, y_max))
)

if mask_inset_circle.any():
    sub_circles = df.loc[mask_inset_circle]
    for idx, row in sub_circles.iterrows():
        others = sub_circles.drop(idx)
        place_label(
            axins,
            row["ent"],
            max(row["exp"], EPS),
            str(int(row["model"])),  # ID do circuito
            others["ent"].values,
            others["exp"].clip(lower=EPS).values,
        )

# ──────────────────────── inset externo (zoom2) ───────────────────────
x_min, x_max = 0.37, 0.395
y_min, y_max = 1 * 1e-2, 0.5 * 1e-0

axins2 = inset_axes(
    ax,
    width="120%",
    height="120%",
    bbox_to_anchor=(0.05, -0.65, 0.35, 0.35),  # “fora” à direita-abaixo
    bbox_transform=ax.transAxes,
    loc="lower left",
    borderpad=0,
)
axins2.set_yscale("log")
axins2.set_ylim(y_min, y_max)
axins2.set_xlim(x_min, x_max)
mark_inset(ax, axins2, loc1=1, loc2=2, fc="none", ec="black", ls="--", lw=0.8)

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

# ─────────────────────────────────────────────────────────────────────

# ─────────────── layout: sem tight_layout para não cortar inset ─────
fig.subplots_adjust(bottom=0.40, right=1)  # margem inferior maior
plt.show()
