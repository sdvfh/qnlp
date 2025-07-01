"""
Unified scatter-plot figure generator for entanglement *and* expressability.

This script reproduces the visual language of the original F1-score box-and-whisker
plots, but now for the *measures* of quantum *expressability* (exp) and
*entanglement* (ent).  For every classifier ID we display the mean measure value
obtained **with** and **without** the state-preparation layer, and for each case we
show the results for **1** and **10** layers of the variational circuit.

Key design choices (mirroring the first script):
    • Two stacked panels – top for classifiers *3* & *33*, bottom for the rest.
    • Light grey separators between the two “state-prep” blocks.
    • Vertical dotted grid only; x-axis is the measure value.
    • Unified x-limits across panels.
    • Legend anchored in the lower-left of the bottom panel.

Visual encoding
───────────────
    ▸ *Colour*  → blue  = *sem* estado de preparação (`with_state_prep == False`)
                 orange = *com* estado de preparação (`True`).
    ▸ *Marker* → ● (filled circle) = 1 camada
                 ▲ (filled triangle) = 10 camadas

A dashed red vertical line indicates the baseline value of the chosen measure
calculated *only* on the state-preparation layer (when present in the CSV).

Author  : <your-name>
Date    : 2025-06-30
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ─────────────────────────────── imports do projecto ─────────────────────────
from utils import models  # mapping id → label (usado no CSV de medidas)
from utils import (
    classical_ensemble_models,
    classical_models,
    quantum_ensemble_models,
    quantum_models,
    read_summary,
)

# ───────────────────────── CONFIGURAÇÃO GERAL ───────────────────────────────
CSV_MEASURES = Path("../../results/measures_all.csv")
LAYERS = (1, 10)
COLOURS = {False: "#1f77b4", True: "#ff7f0e"}  # sem / com estado de prep.
MARKERS = {1: "o", 10: "^"}  # bolinha / triângulo

# ───────────────────── helpers (herdados e adaptados) ───────────────────────


def compute_positions_points(
    n_rows: int,
    primaries: Sequence[bool],
    layers: Sequence[int],
    gap: float = 0.25,
) -> tuple[Dict[tuple[bool, int], np.ndarray]]:
    mapping: dict[tuple[bool, int], np.ndarray] = {}
    base = np.arange(n_rows)
    for pi, prep in enumerate(primaries):
        block_off = (pi - (len(primaries) - 1) / 2) * gap
        for li, layer in enumerate(layers):
            layer_off = (li - (len(layers) - 1) / 2) * (gap / 2)
            mapping[(prep, layer)] = base + block_off + layer_off
    return mapping


def add_inner_separators(
    ax: Axes,
    primaries: Sequence[bool],
    layers: Sequence[int],
    pos_map: dict[tuple[bool, int], np.ndarray],
    n_rows: int,
) -> None:
    for r in range(n_rows):
        y_a = pos_map[(primaries[0], layers[-1])][r]
        y_b = pos_map[(primaries[1], layers[0])][r]
        ax.axhline((y_a + y_b) / 2, color="#6A6A6A", lw=0.8, zorder=0, alpha=0.7)


# ─────────────────────────── núcleo do plotter ──────────────────────────────


def plot_measure_panel(
    ax: Axes,
    df: pd.DataFrame,
    classifier_order: Sequence[int],
    measure_col: str,
    primaries: Sequence[bool],
    layers: Sequence[int],
    pos_map: dict[tuple[bool, int], np.ndarray],
    dashed_baseline: float | None,
) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#CCCCCC", ls="--", lw=0.8)
    for y in np.arange(len(classifier_order) + 1) - 0.5:
        ax.axhline(y, color="black", lw=1, zorder=0)

    for prep in primaries:
        for layer in layers:
            sub = df[(df["with_state_prep"] == prep) & (df["n_layers"] == layer)]
            for r, clf in enumerate(classifier_order):
                val = sub.loc[sub["model"] == clf, measure_col]
                if val.empty or np.isnan(val.iloc[0]):
                    continue
                ax.scatter(
                    val.iloc[0],
                    pos_map[(prep, layer)][r],
                    marker=MARKERS[layer],
                    s=60,
                    facecolor=COLOURS[prep],
                    edgecolor="black",
                    zorder=3,
                )

    if dashed_baseline is not None:
        ax.axvline(dashed_baseline, color="red", ls="--", lw=1.2, zorder=1)

    add_inner_separators(ax, primaries, layers, pos_map, len(classifier_order))

    ax.set_ylim(len(classifier_order) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_xlabel(measure_col.upper())


# ──────────────────────── gerador de figuras ────────────────────────────────


def generate_measure_figure(
    df_meas: pd.DataFrame,
    df_runs: pd.DataFrame,
    measure_col: str,  # "exp" ou "ent"
    out_name: str,
) -> None:
    """Cria uma única figura agregando todos os classificadores."""

    df = (
        df_meas.merge(
            df_runs[["model_classifier", "n_layers"]],
            left_on=["model", "n_layers"],
            right_on=["model_classifier", "n_layers"],
        )
        .drop(columns="model_classifier")
        .groupby(["model", "n_layers", "with_state_prep"], as_index=False)
        .agg({measure_col: "mean"})
    )
    df["model"] = df["model"].astype(int)

    # todos os modelos em uma única ordem
    order = sorted(df["model"].unique())
    primaries = [False, True]
    pos_map = compute_positions_points(len(order), primaries, LAYERS)

    # baseline a partir de estados de prep
    baseline_row = df_meas[df_meas["model"].map(str).str.contains("prep", case=False)]
    baseline_val = baseline_row[measure_col].mean() if not baseline_row.empty else None

    fig, ax = plt.subplots(figsize=(6, 10), dpi=300)
    plot_measure_panel(
        ax,
        df,
        order,
        measure_col,
        primaries,
        LAYERS,
        pos_map,
        dashed_baseline=baseline_val,
    )

    if measure_col == "exp":
        ax.set_xscale("log")

    # legenda única
    handles_prep = [
        Patch(facecolor=COLOURS[p], label=("Com" if p else "Sem") + " estado de prep.")
        for p in primaries
    ]
    handles_layers = [
        Line2D(
            [],
            [],
            marker=MARKERS[layer],
            color="black",
            ls="",
            markersize=8,
            label=f"{layer} camada{'s' if layer > 1 else ''}",
        )
        for layer in LAYERS
    ]
    base_line = Line2D([], [], color="red", ls="--", label="Só estado de prep.")

    ax.legend(
        handles=handles_prep + handles_layers + [base_line],
        title="Legenda",
        loc="lower left",
        fontsize="small",
        framealpha=1.0,
        edgecolor="black",
        fancybox=True,
    )

    fig.suptitle(f"{measure_col.upper()} — Todos os Classificadores")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = Path("../../figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}_all.pgf"
    plt.show()
    # fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ────────────────────────────────── main ─────────────────────────────────────
if __name__ == "__main__":
    df_meas_all = pd.read_csv(CSV_MEASURES)
    df_meas_all["model"] = df_meas_all["model"].replace(models)

    df_runs_all = read_summary()

    # apenas duas figuras: expressability e entanglement, sem separação por circuito 3
    for meas, fname in [("exp", "expressability"), ("ent", "entanglement")]:
        generate_measure_figure(
            df_meas=df_meas_all.copy(),
            df_runs=df_runs_all.copy(),
            measure_col=meas,
            out_name=fname,
        )
