"""
Generate high-resolution box-and-whisker figures for:

1. Transformer comparison (original analysis).
2. Feature-count comparison for *tomaarsen/mpnet-base-nli-matryoshka*.

The script reproduces every visual and statistical element of the original
notebook, merely adding light horizontal separators between the colour-coded
blocks in each classifier row.

Author : <your-name>
Created: 2025-06-11
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from utils import classical_models, quantum_models, read_summary

# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────


def pairwise_wilcoxon_holm(
    df_pair: pd.DataFrame, *, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run pairwise Wilcoxon signed-rank tests across all column combinations and
    correct the *p*-values using the Holm–Bonferroni method.

    Parameters
    ----------
    df_pair:
        Wide table whose columns are matched samples.
    alpha:
        Family-wise error-rate threshold for the Holm procedure.

    Returns
    -------
    pd.DataFrame
        Long-format table with raw statistics, uncorrected and Holm-adjusted
        *p*-values, plus a *reject* indicator.
    """
    cols: List[str] = df_pair.columns.tolist()
    records: List[Tuple[str, str, float, float]] = []
    for ci, cj in itertools.combinations(cols, 2):
        x, y = df_pair[ci].values, df_pair[cj].values
        if len(x) == len(y) and (x != y).any():
            stat, p_raw = wilcoxon(x, y)
        else:  # samples identical or length mismatch
            stat, p_raw = np.nan, 1.0
        records.append((ci, cj, float(stat), float(p_raw)))

    pvals = [rec[3] for rec in records]
    reject, pvals_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")

    df_out = pd.DataFrame.from_records(
        records,
        columns=["model_i", "model_j", "statistic", "p_uncorrected"],
    )
    df_out["p_holm"] = pvals_holm
    df_out["reject"] = reject
    return df_out


def compute_positions(
    n_classifiers: int,
    primary: Sequence[str | int],
    layers: Sequence[int],
    width: float,
    block_gap: float,
) -> Tuple[List[np.ndarray], Dict[Tuple[str | int, int], np.ndarray]]:
    """
    Compute horizontal offsets (along the x-axis) for each combination of the
    primary grouping variable (transformer or n_features) and n_layers.

    Returns both the ordered list of offset arrays and a mapping
    ``(primary_value, layer) → offsets`` for quick lookup.
    """
    positions: List[np.ndarray] = []
    combos = [(pv, ly) for pv in primary for ly in layers]
    for pi in range(len(primary)):
        offset = pi * (len(layers) * width + block_gap) - block_gap
        for li in range(len(layers)):
            pos = np.arange(n_classifiers) - 0.3 + width / 2 + li * width + offset
            positions.append(pos)
    return positions, {combos[i]: positions[i] for i in range(len(combos))}


# ──────────────────────────────────────────────────────────────────────────────
# Plotters
# ──────────────────────────────────────────────────────────────────────────────


def _add_internal_separators(
    ax: Axes,
    pos_map: Dict[Tuple[str | int, int], np.ndarray],
    primary_vals: Sequence[str | int],
    layers: Sequence[int],
    n_rows: int,
) -> None:
    """
    Draw subtle horizontal lines separating consecutive primary groups
    (either transformers or n_features) for each classifier row.
    """
    for ci in range(n_rows):
        for pi in range(len(primary_vals) - 1):
            y_last_curr = pos_map[(primary_vals[pi], layers[-1])][ci]
            y_first_next = pos_map[(primary_vals[pi + 1], layers[0])][ci]
            y_sep = (y_last_curr + y_first_next) / 2
            ax.axhline(
                y=y_sep,
                color="#545454",
                linewidth=0.8,
                zorder=0,
                alpha=0.7,
            )


def plot_panel_transformer(
    ax: Axes,
    df_panel: pd.DataFrame,
    classifier_order: Sequence[int],
    transformers: Sequence[str],
    layers: Sequence[int],
    positions: List[np.ndarray],
    pos_map: Dict[Tuple[str, int], np.ndarray],
    width: float,
    color_map: Dict[str, np.ndarray],
    hatch_map: Dict[int, str],
    *,
    alpha: float = 0.05,
) -> None:
    """
    Render one panel comparing different *model_transformer* values.

    Visual grammar identical to the original notebook, with the addition of a
    light interior separator between transformer blocks.
    """
    # ― grid and outer classifier separators ―
    ax.set_axisbelow(True)
    ax.grid(
        axis="x",  # ←  apenas no eixo x → linhas verticais
        which="major",
        color="#CCCCCC",
        linestyle="--",
        linewidth=0.8,
    )
    for y in np.arange(len(classifier_order) + 1) - 0.5:
        ax.axhline(y=y, color="black", linewidth=1, zorder=0)

    # ― boxplots ―
    combos = [(tr, ly) for tr in transformers for ly in layers]
    for i, (tr, ly) in enumerate(combos):
        data = [
            df_panel[
                (df_panel["model_classifier"] == clf)
                & (df_panel["model_transformer"] == tr)
                & (df_panel["n_layers"] == ly)
            ]["f1"].values
            for clf in classifier_order
        ]
        bp = ax.boxplot(
            data,
            positions=positions[i],
            widths=width,
            vert=False,
            patch_artist=True,
            manage_ticks=False,
            zorder=1,
            boxprops={"linewidth": 1.5},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            medianprops={"linewidth": 1.5},
        )
        for box in bp["boxes"]:
            box.set_facecolor(color_map[tr])
            box.set_hatch(hatch_map[ly])

    # ― internal separators between transformer blocks ―
    _add_internal_separators(ax, pos_map, transformers, layers, len(classifier_order))

    # ― statistical annotations (× for layers, ↔ for transformers) ―
    for ci, clf in enumerate(classifier_order):
        pivot = df_panel[df_panel["model_classifier"] == clf].pivot(
            index="seed", columns=["model_transformer", "n_layers"], values="f1"
        )
        pivot.columns = pd.MultiIndex.from_tuples(pivot.columns)
        res_layers = pairwise_wilcoxon_holm(pivot, alpha=alpha)
        for _, row in res_layers[~res_layers["reject"]].iterrows():
            (t0, l0), (t1, l1) = row["model_i"], row["model_j"]
            if t0 == t1 and l0 != l1:
                y0, y1 = pos_map[(t0, l0)][ci], pos_map[(t1, l1)][ci]
                x_mid = (pivot[(t0, l0)].median() + pivot[(t1, l1)].median()) / 2
                ax.text(
                    x_mid,
                    (y0 + y1) / 2,
                    "x",
                    ha="center",
                    va="center",
                    color="#ed1fed",
                    fontsize=12,
                    fontweight="bold",
                    zorder=5,
                )

        for ly in layers:
            if ly not in pivot.columns.get_level_values(1):
                continue
            sub = pivot.xs(ly, level=1, axis=1)
            res_trans = pairwise_wilcoxon_holm(sub, alpha=alpha)
            for _, row in res_trans[~res_trans["reject"]].iterrows():
                t0, t1 = row["model_i"], row["model_j"]
                m0, m1 = pivot[(t0, ly)].median(), pivot[(t1, ly)].median()
                y0, y1 = pos_map[(t0, ly)][ci], pos_map[(t1, ly)][ci]
                ax.annotate(
                    "",
                    xy=(m1, y1),
                    xytext=(m0, y0),
                    arrowprops={
                        "arrowstyle": "<->",
                        "color": "green",
                        "linewidth": 1.5,
                    },
                    zorder=5,
                )

    # ― axis cosmetics ―
    ax.set_ylim(len(classifier_order) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_xlabel("F1 score")


def plot_panel_features(
    ax: Axes,
    df_panel: pd.DataFrame,
    classifier_order: Sequence[int],
    features: Sequence[int],
    layers: Sequence[int],
    positions: List[np.ndarray],
    pos_map: Dict[Tuple[int, int], np.ndarray],
    width: float,
    color_map: Dict[int, np.ndarray],
    hatch_map: Dict[int, str],
    *,
    alpha: float = 0.05,
) -> None:
    """
    Render one panel comparing different *n_features* values for a single
    transformer.

    Mirrors the visual grammar of `plot_panel_transformer`, substituting colours
    for feature counts instead of transformer names.
    """
    # ― grid and outer classifier separators ―
    ax.set_axisbelow(True)
    ax.grid(
        axis="x",  # ←  apenas no eixo x → linhas verticais
        which="major",
        color="#CCCCCC",
        linestyle="--",
        linewidth=0.8,
    )
    for y in np.arange(len(classifier_order) + 1) - 0.5:
        ax.axhline(y=y, color="black", linewidth=1, zorder=0)

    # ― boxplots ―
    combos = [(ft, ly) for ft in features for ly in layers]
    for i, (ft, ly) in enumerate(combos):
        data = [
            df_panel[
                (df_panel["model_classifier"] == clf)
                & (df_panel["n_features"] == ft)
                & (df_panel["n_layers"] == ly)
            ]["f1"].values
            for clf in classifier_order
        ]
        bp = ax.boxplot(
            data,
            positions=positions[i],
            widths=width,
            vert=False,
            patch_artist=True,
            manage_ticks=False,
            zorder=1,
            boxprops={"linewidth": 1.5},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            medianprops={"linewidth": 1.5},
        )
        for box in bp["boxes"]:
            box.set_facecolor(color_map[ft])
            box.set_hatch(hatch_map[ly])

    # ― internal separators between feature blocks ―
    _add_internal_separators(ax, pos_map, features, layers, len(classifier_order))

    # ― statistical annotations (× for layers, ↔ for features) ―
    for ci, clf in enumerate(classifier_order):
        pivot = df_panel[df_panel["model_classifier"] == clf].pivot(
            index="seed", columns=["n_features", "n_layers"], values="f1"
        )
        pivot.columns = pd.MultiIndex.from_tuples(pivot.columns)
        res_layers = pairwise_wilcoxon_holm(pivot, alpha=alpha)
        for _, row in res_layers[~res_layers["reject"]].iterrows():
            (f0, l0), (f1, l1) = row["model_i"], row["model_j"]
            if f0 == f1 and l0 != l1:
                y0, y1 = pos_map[(f0, l0)][ci], pos_map[(f1, l1)][ci]
                x_mid = (pivot[(f0, l0)].median() + pivot[(f1, l1)].median()) / 2
                ax.text(
                    x_mid,
                    (y0 + y1) / 2,
                    "x",
                    ha="center",
                    va="center",
                    color="#ed1fed",
                    fontsize=12,
                    fontweight="bold",
                    zorder=5,
                )

        for ly in layers:
            if ly not in pivot.columns.get_level_values(1):
                continue
            sub = pivot.xs(ly, level=1, axis=1)
            res_feat = pairwise_wilcoxon_holm(sub, alpha=alpha)
            for _, row in res_feat[~res_feat["reject"]].iterrows():
                f0, f1 = row["model_i"], row["model_j"]
                m0, m1 = pivot[(f0, ly)].median(), pivot[(f1, ly)].median()
                y0, y1 = pos_map[(f0, ly)][ci], pos_map[(f1, ly)][ci]
                ax.annotate(
                    "",
                    xy=(m1, y1),
                    xytext=(m0, y0),
                    arrowprops={
                        "arrowstyle": "<->",
                        "color": "green",
                        "linewidth": 1.5,
                    },
                    zorder=5,
                )

    # ― axis cosmetics ―
    ax.set_ylim(len(classifier_order) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_xlabel("F1 score")


# ──────────────────────────────────────────────────────────────────────────────
# Main routine – generate both figure families
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Common dataset names
    DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard")
    OUTPUT_DIR = Path("../../figures").resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Shared metadata
    layers: List[int] = sorted(read_summary()["n_layers"].unique())
    hatch_map: Dict[int, str] = {layers[0]: "", layers[1]: "//"}
    block_gap = 0.1

    # ── 1. Transformer analysis (original) ──────────────────────────────
    transformers: List[str] = read_summary()["model_transformer"].unique().tolist()
    width_tr = 0.6 / (len(transformers) * len(layers))
    colors_tr = plt.cm.tab10(np.linspace(0, 1, len(transformers)))
    color_map_tr: Dict[str, np.ndarray] = {
        tr: colors_tr[i] for i, tr in enumerate(transformers)
    }

    for ds in DATASETS:
        df = read_summary()
        df = df[
            (df["dataset"] == ds)
            & (df["n_features"] == 16)
            & (df["model_classifier"].isin(quantum_models + classical_models))
        ].copy()

        df_top = df[df["model_classifier"].isin([3, 33])]
        df_bot = df[~df["model_classifier"].isin([3, 33])]

        top_order = (
            df_top.groupby("model_classifier")["f1"].mean().sort_values().index.tolist()
        )
        bot_order = (
            df_bot.groupby("model_classifier")["f1"].mean().sort_values().index.tolist()
        )

        pos_top, pos_map_top = compute_positions(
            len(top_order), transformers, layers, width_tr, block_gap
        )
        pos_bot, pos_map_bot = compute_positions(
            len(bot_order), transformers, layers, width_tr, block_gap
        )

        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(8, 20),
            dpi=600,
            sharex=False,
            gridspec_kw={"height_ratios": [len(top_order), len(bot_order)]},
        )

        plot_panel_transformer(
            ax_top,
            df_top,
            top_order,
            transformers,
            layers,
            pos_top,
            pos_map_top,
            width_tr,
            color_map_tr,
            hatch_map,
        )
        plot_panel_transformer(
            ax_bot,
            df_bot,
            bot_order,
            transformers,
            layers,
            pos_bot,
            pos_map_bot,
            width_tr,
            color_map_tr,
            hatch_map,
        )

        ax_top.set_title("Modelos 3 e 33")
        ax_bot.set_title("Demais Modelos")

        handles_tr = [Patch(facecolor=color_map_tr[t], label=t) for t in transformers]
        handles_layers = [
            Patch(
                facecolor="white",
                hatch=hatch_map[ly],
                edgecolor="black",
                label=f"{ly} camada{'s' if ly == 10 else ''}",
            )
            for ly in layers
        ]
        x_handle = Line2D(
            [],
            [],
            marker="x",
            color="#ed1fed",
            linestyle="",
            markerfacecolor="none",
            markersize=12,
            label="Sem diferença entre camadas",
        )
        arrow_handle = Line2D(
            [],
            [],
            color="green",
            marker="None",
            linestyle="-",
            linewidth=1.5,
            label="Sem diferença entre transformers",
        )
        ax_bot.legend(
            handles=handles_tr + handles_layers + [x_handle, arrow_handle],
            title="Legenda",
            loc="lower left",
            fontsize="small",
            framealpha=1.0,  # ← opacidade total
            edgecolor="black",  # (opcional) cor da borda
            fancybox=True,  # (opcional) cantos não-arredondados
        )

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"transformers_{ds}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # ── 2. Feature-count analysis for the Matryoshka model ───────────────
    TARGET_TRANSFORMER = "tomaarsen/mpnet-base-nli-matryoshka"
    features_all = [16, 32, 768]
    width_ft = 0.6 / (len(features_all) * len(layers))
    colors_ft = plt.cm.tab10(np.linspace(0, 1, len(features_all)))
    color_map_ft: Dict[int, np.ndarray] = {
        ft: colors_ft[i] for i, ft in enumerate(features_all)
    }

    for ds in DATASETS:
        df = read_summary()
        df = df[
            (df["dataset"] == ds)
            & (df["model_transformer"] == TARGET_TRANSFORMER)
            & (df["n_features"].isin(features_all))
            & (df["model_classifier"].isin(quantum_models + classical_models))
        ].copy()

        df_top = df[df["model_classifier"].isin([3, 33])]
        df_bot = df[~df["model_classifier"].isin([3, 33])]

        top_order = (
            df_top.groupby("model_classifier")["f1"].mean().sort_values().index.tolist()
        )
        bot_order = (
            df_bot.groupby("model_classifier")["f1"].mean().sort_values().index.tolist()
        )

        pos_top, pos_map_top = compute_positions(
            len(top_order), features_all, layers, width_ft, block_gap
        )
        pos_bot, pos_map_bot = compute_positions(
            len(bot_order), features_all, layers, width_ft, block_gap
        )

        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(8, 20),
            dpi=600,
            sharex=False,
            gridspec_kw={"height_ratios": [len(top_order), len(bot_order)]},
        )

        plot_panel_features(
            ax_top,
            df_top,
            top_order,
            features_all,
            layers,
            pos_top,
            pos_map_top,
            width_ft,
            color_map_ft,
            hatch_map,
        )
        plot_panel_features(
            ax_bot,
            df_bot,
            bot_order,
            features_all,
            layers,
            pos_bot,
            pos_map_bot,
            width_ft,
            color_map_ft,
            hatch_map,
        )

        ax_top.set_title("Modelos 3 e 33")
        ax_bot.set_title("Demais Modelos")

        handles_ft = [
            Patch(facecolor=color_map_ft[ft], label=f"{ft} atributos")
            for ft in features_all
        ]
        handles_layers = [
            Patch(
                facecolor="white",
                hatch=hatch_map[ly],
                edgecolor="black",
                label=f"{ly} camada{'s' if ly == 10 else ''}",
            )
            for ly in layers
        ]
        x_handle = Line2D(
            [],
            [],
            marker="x",
            color="#ed1fed",
            linestyle="",
            markerfacecolor="none",
            markersize=12,
            label="Sem diferença entre camadas",
        )
        arrow_handle = Line2D(
            [],
            [],
            color="green",
            marker="None",
            linestyle="-",
            linewidth=1.5,
            label="Sem diferença entre n° de atributos",
        )
        ax_bot.legend(
            handles=handles_ft + handles_layers + [x_handle, arrow_handle],
            title="Legenda",
            loc="lower left",
            fontsize="small",
            framealpha=1.0,  # ← opacidade total
            edgecolor="black",  # (opcional) cor da borda
            fancybox=True,  # (opcional) cantos não-arredondados
        )

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"n_features_{ds}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
