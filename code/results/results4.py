"""
Unified scatter-plot figure generator for entanglement *and* expressability.

This script reproduces the visual language of the original F1-score box-and-whisker
plots, but now for the *measures* of quantum *expressability* (exp) and
*entanglement* (ent). For every classifier ID we display the mean measure value
obtained **with** and **without** the state-preparation layer, and for each case we
show the results for **1** and **10** layers of the variational circuit.

Key design choices:
    • Single panel – todos os classificadores juntos.
    • Light grey separators between os blocos sem/com estado-de-prep.
    • Vertical dotted grid only; x-axis é o valor da medida.
    • Legend: expressability no canto superior esquerdo; entanglement no canto superior direito.
    • Expressability em escala log, entanglement em escala linear.
    • Pontos de mesma cor alinhados horizontalmente, cores diferentes um acima do outro,
      e centralizados na subdivisão de camada.
    • Classificadores ID em ordem crescente de cima para baixo.

Visual encoding
───────────────
    ▸ *Colour*  → verde-claro = sem estado de preparação (`with_state_prep == False`)
                 verde-escuro = com estado de preparação (`True`).
    ▸ *Marker* → ● (círculo) = 1 camada
                 ▲ (triângulo) = 10 camadas

A dashed red vertical line indica o valor baseline da medida
calculado *apenas* no estado-de-prep (quando presente no CSV).

Author  : <your-name>
Date    : 2025-06-30
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ─────────────────────────────── imports do projecto ─────────────────────────
from utils import models  # mapeia id → label (usado no CSV de medidas)
from utils import read_summary

# ───────────────────────── CONFIGURAÇÃO GERAL ───────────────────────────────
CSV_MEASURES = Path("../../results/measures_all.csv")
LAYERS = (1, 10)
COLOURS = {False: "#03fc6b", True: "#53968d"}  # sem PE / com PE
MARKERS = {1: "o", 10: "^"}  # bolinha / triângulo

# ───────────────────── helpers (herdados e adaptados) ───────────────────────


def compute_positions_points(
    n_rows: int,
    primaries: Sequence[bool],
    layers: Sequence[int],
    gap: float = 0.5,
) -> Dict[tuple[bool, int], np.ndarray]:
    """
    Retorna y-offsets (1-D arrays) para cada (state_prep, layer), de modo que:
      • todos os pontos de uma mesma cor (prep) fiquem na mesma linha horizontal;
      • blocos verde-claro e verde-escuro fiquem verticalmente separados por gap.
    """
    mapping: Dict[tuple[bool, int], np.ndarray] = {}
    base = np.arange(n_rows)
    for pi, prep in enumerate(primaries):
        offset = (pi - (len(primaries) - 1) / 2) * gap  # ±0.25 quando gap=0.5
        for layer in layers:
            mapping[(prep, layer)] = base + offset
    return mapping


def add_inner_separators(
    ax: Axes,
    primaries: Sequence[bool],
    layers: Sequence[int],
    pos_map: Dict[tuple[bool, int], np.ndarray],
    n_rows: int,
) -> None:
    """Linhas cinza-claro a separar blocos sem/com estado-de-prep."""
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
    pos_map: Dict[tuple[bool, int], np.ndarray],
    dashed_baseline: float | None,
) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#CCCCCC", ls="--", lw=0.8)

    # desenha linhas horizontais externas entre circuitos
    for y in np.arange(len(classifier_order) + 1) - 0.5:
        ax.axhline(y, color="black", lw=1, zorder=0)

    # desenha os pontos alinhados por cor
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

    # linha tracejada vermelha (baseline)
    if dashed_baseline is not None:
        ax.axvline(dashed_baseline, color="red", ls="--", lw=1.2, zorder=1)

    # separadores internos
    add_inner_separators(ax, primaries, layers, pos_map, len(classifier_order))

    # restabelece ticks e labels y (1 no topo, maior embaixo)
    ax.set_ylim(len(classifier_order) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(classifier_order)))
    ax.set_yticklabels(classifier_order)

    # label condicional do eixo X
    if measure_col == "exp":
        ax.set_xlabel("Divergência KL")
    else:
        ax.set_xlabel("Emaranhamento")


# ──────────────────────── gerador de figuras ────────────────────────────────


def generate_measure_figure(
    df_meas: pd.DataFrame,
    df_runs: pd.DataFrame,
    measure_col: str,  # "exp" ou "ent"
    out_name: str,
) -> None:
    """Cria uma única figura agregando todos os classificadores."""

    # agrupa e calcula média
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

    order = sorted(df["model"].unique())
    primaries = [False, True]
    pos_map = compute_positions_points(len(order), primaries, LAYERS)

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

    # escala log e anotação para expressability
    if measure_col == "exp":
        ax.set_xscale("log")
        # seta horizontal abaixo do eixo X
        ax.annotate(
            "",
            xy=(1, -0.06),
            xytext=(0, -0.06),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "<->", "lw": 1.5, "color": "black"},
            annotation_clip=False,
        )
        ax.text(0, -0.08, "Alta Exp", transform=ax.transAxes, ha="left", va="top")
        ax.text(1, -0.08, "Baixa Exp", transform=ax.transAxes, ha="right", va="top")

    # prepara handles de legenda
    handles_prep = [
        Patch(facecolor=COLOURS[p], label=("Sem PE" if not p else "Com PE"))
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
    base_line = Line2D([], [], color="red", ls="--", label="PE")
    handles = handles_prep + handles_layers + [base_line]

    # posição da legenda conforme medida
    legend_loc = "upper left" if measure_col == "exp" else "upper right"
    ax.legend(
        handles=handles,
        title="Legenda",
        loc=legend_loc,
        fontsize="small",
        framealpha=1.0,
        edgecolor="black",
        fancybox=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])

    out_dir = Path("../../figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}_all.pgf"
    # plt.show()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ────────────────────────────────── main ─────────────────────────────────────
if __name__ == "__main__":
    df_meas_all = pd.read_csv(CSV_MEASURES)
    df_meas_all["model"] = df_meas_all["model"].replace(models)

    df_runs_all = read_summary()

    for meas, fname in [("exp", "expressability"), ("ent", "entanglement")]:
        generate_measure_figure(
            df_meas=df_meas_all.copy(),
            df_runs=df_runs_all.copy(),
            measure_col=meas,
            out_name=fname,
        )
