"""
Unified box-and-whisker figure generator.

Author: <Your Name>
Date: 2025-06-11

This module generates unified box-and-whisker plots with Wilcoxon test annotations and Holm correction.
It supports two primary attributes:
- Case A: color by 'model_transformer' (e.g., n_features = 16)
- Case B: color by 'n_features' (single transformer)

Features:
- Wilcoxon + Holm annotations ('×' for no difference between layers, '↔' arrows for no difference between primaries)
- Light separators between color blocks
- Vertical dotted grid lines
- Unified x-limits across sub-panels
- Opaque legend in the lower-left corner
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
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from utils import (
    classical_ensemble_models,
    classical_models,
    quantum_ensemble_models,
    quantum_models,
    read_summary,
)

pd.set_option("future.no_silent_downcasting", True)


def pairwise_wilcoxon_holm(
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Perform all-pairs Wilcoxon test with Holm correction.

    Args:
        df: DataFrame where columns represent paired samples.
        alpha: Significance level for Holm correction.

    Returns:
        DataFrame with columns ['i', 'j', 'stat', 'p_raw', 'p_holm', 'reject'], where
        'i' and 'j' are compared column names, 'stat' is the test statistic,
        'p_raw' is the raw p-value, 'p_holm' is the Holm-adjusted p-value,
        and 'reject' indicates if the null hypothesis is rejected.
    """
    if df.shape[1] < 2:
        return pd.DataFrame(columns=["i", "j", "stat", "p_raw", "p_holm", "reject"])

    records: List[Tuple[str, str, float, float]] = []
    for a, b in itertools.combinations(df.columns, 2):
        x = df[a].values
        y = df[b].values
        if len(x) == len(y) and (x != y).any():
            stat, p = wilcoxon(x, y)
        else:
            stat, p = float("nan"), 1.0
        records.append((a, b, float(stat), float(p)))

    p_raw = [rec[3] for rec in records]
    reject, p_holm, _, _ = multipletests(p_raw, alpha=alpha, method="holm")

    result = pd.DataFrame.from_records(
        records,
        columns=["i", "j", "stat", "p_raw"],
    ).assign(p_holm=p_holm, reject=reject)
    return result


def compute_positions(
    n_rows: int,
    primaries: Sequence[str | int],
    layers: Sequence[int],
    width: float,
    gap: float,
) -> Tuple[List[np.ndarray], Dict[Tuple[str | int, int], np.ndarray]]:
    """
    Compute y-axis positions for each combination of primary value and layer.

    Args:
        n_rows: Number of classifiers (rows).
        primaries: Sequence of primary values (e.g., feature counts or transformer names).
        layers: Sequence of layer counts.
        width: Width of each boxplot.
        gap: Gap between groups of boxplots for different primaries.

    Returns:
        - positions: List of numpy arrays with y-positions for each (primary, layer) combination.
        - pos_map: Mapping from (primary, layer) to its y-position array.
    """
    combinations = [(p, layer) for p in primaries for layer in layers]
    positions: List[np.ndarray] = []
    for idx_p, _ in enumerate(primaries):
        offset = idx_p * (len(layers) * width + gap) - gap
        for idx_l, _ in enumerate(layers):
            y = np.arange(n_rows) - 0.3 + width / 2 + idx_l * width + offset
            positions.append(y)

    pos_map = {combo: positions[i] for i, combo in enumerate(combinations)}
    return positions, pos_map


def add_inner_separators(
    ax: Axes,
    pos_map: Dict[Tuple[str | int, int], np.ndarray],
    primaries: Sequence[str | int],
    layers: Sequence[int],
    n_rows: int,
) -> None:
    """
    Draw light separators between blocks of different primary values.

    Args:
        ax: Matplotlib Axes object.
        pos_map: Mapping from (primary, layer) to y-position arrays.
        primaries: Sequence of primary values.
        layers: Sequence of layer counts.
        n_rows: Number of classifiers (rows).
    """
    for row in range(n_rows):
        for i in range(len(primaries) - 1):
            y_end_first = pos_map[(primaries[i], layers[-1])][row]
            y_start_next = pos_map[(primaries[i + 1], layers[0])][row]
            mid = (y_end_first + y_start_next) / 2
            ax.axhline(mid, color="#6A6A6A", lw=0.8, zorder=0, alpha=0.7)


def plot_panel(
    ax: Axes,
    df: pd.DataFrame,
    classifier_order: Sequence[int],
    primary_col: str,
    primary_vals: Sequence[str | int],
    layers: Sequence[int],
    positions: List[np.ndarray],
    pos_map: Dict[Tuple[str | int, int], np.ndarray],
    width: float,
    color_map: Dict[str | int, np.ndarray],
    hatch_map: Dict[int, str],
    alpha: float = 0.05,
) -> None:
    """
    Plot a single panel of boxplots with Wilcoxon annotations.

    Args:
        ax: Matplotlib Axes object.
        df: DataFrame containing 'f1' scores and grouping columns.
        classifier_order: List of classifier identifiers in plotting order.
        primary_col: Column to use for primary grouping.
        primary_vals: Values of the primary grouping column.
        layers: Sequence of layer counts.
        positions: Precomputed y-positions for all boxplots.
        pos_map: Mapping from (primary_val, layer) to y-position arrays.
        width: Width of each boxplot.
        color_map: Mapping from primary_val to its color.
        hatch_map: Mapping from layer count to hatch pattern.
        alpha: Significance level for Wilcoxon tests.
    """
    # Configure grid and separators
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#CCCCCC", ls="--", lw=0.8)
    for y in np.arange(len(classifier_order) + 1) - 0.5:
        ax.axhline(y, color="black", lw=1, zorder=0)

    # Draw each boxplot group
    combos = [(p, layer) for p in primary_vals for layer in layers]
    for idx, (p, layer) in enumerate(combos):
        # Extract F1 values for each classifier
        samples = []
        for clf in classifier_order:
            mask = (
                (df["model_classifier"] == clf)
                & (df[primary_col] == p)
                & (df["n_layers"] == layer)
            )
            values = df.loc[mask, "f1"].to_numpy(dtype=float)
            samples.append(values[np.isfinite(values)])

        # Skip if no data available
        if all(len(s) == 0 for s in samples):
            continue
        samples = [s if len(s) else np.array([np.nan]) for s in samples]

        # Plot boxplots
        bp = ax.boxplot(
            samples,
            positions=positions[idx],
            widths=width,
            vert=False,
            patch_artist=True,
            manage_ticks=False,
            zorder=1,
            sym=".",
            boxprops={"lw": 1.5},
            whiskerprops={"lw": 1.5},
            capprops={"lw": 1.5},
            medianprops={"lw": 1.5},
        )
        # Style boxes
        for box, vals in zip(bp["boxes"], samples, strict=True):
            if np.isnan(vals).all():
                box.set_visible(False)
            else:
                box.set_facecolor(color_map[p])
                box.set_hatch(hatch_map[layer])

    # Add separators between primary groups
    add_inner_separators(ax, pos_map, primary_vals, layers, len(classifier_order))

    # Add Wilcoxon + Holm annotations
    for row_idx, clf in enumerate(classifier_order):
        pivot = df[df["model_classifier"] == clf].pivot(
            index="seed", columns=[primary_col, "n_layers"], values="f1"
        )
        pivot.columns = pd.MultiIndex.from_tuples(pivot.columns)

        # '×' for same primary, different layers
        results = pairwise_wilcoxon_holm(pivot, alpha=alpha)
        for _, r in results[~results["reject"]].iterrows():
            (p0, l0), (p1, l1) = r["i"], r["j"]
            if p0 == p1 and l0 != l1:
                y0 = pos_map[(p0, l0)][row_idx]
                y1 = pos_map[(p1, l1)][row_idx]
                x_mid = (pivot[(p0, l0)].median() + pivot[(p1, l1)].median()) / 2
                ax.text(
                    x_mid,
                    (y0 + y1) / 2,
                    "×",
                    ha="center",
                    va="center",
                    color="#ed1fed",
                    fontsize=12,
                    fontweight="bold",
                    zorder=5,
                )

        # '↔' for same layer, different primaries
        for layer in layers:
            if layer not in pivot.columns.get_level_values(1):
                continue
            sub = pivot.xs(layer, level=1, axis=1)
            results_layer = pairwise_wilcoxon_holm(sub, alpha=alpha)
            for _, r in results_layer[~results_layer["reject"]].iterrows():
                p0, p1 = r["i"], r["j"]
                y0 = pos_map[(p0, layer)][row_idx]
                y1 = pos_map[(p1, layer)][row_idx]
                m0 = pivot[(p0, layer)].median()
                m1 = pivot[(p1, layer)].median()
                ax.annotate(
                    "",
                    (m1, y1),
                    (m0, y0),
                    arrowprops={"arrowstyle": "<->", "color": "#0b31bf", "lw": 1.5},
                    zorder=5,
                )

    # Configure axes labels and limits
    ax.set_ylim(len(classifier_order) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_xlabel("F1")


def generate_figure(
    df: pd.DataFrame,
    dataset: str,
    primary_col: str,
    primary_vals: Sequence[str | int],
    color_map: Dict[str | int, np.ndarray],
    arrow_label: str,
    attr_label_fmt: str,
    out_name: str,
    *,
    layers: Sequence[int],
    hatch_map: Dict[int, str],
    block_gap: float = 0.1,
) -> None:
    """
    Generate and save a unified figure of boxplots and difference bars for given dataset.

    Args:
        df: DataFrame with experiment summary data.
        dataset: Dataset identifier string.
        primary_col: Column name for primary grouping ('n_features' or 'model_transformer').
        primary_vals: Values for the primary grouping.
        color_map: Mapping from primary_vals to colors.
        arrow_label: Label for the arrow annotation in legend.
        attr_label_fmt: Format string for primary values in legend.
        out_name: Base name for output files.
        layers: Sequence of layer counts.
        hatch_map: Mapping from layer to hatch pattern.
        block_gap: Gap between blocks of boxplots.
    """
    # Split classifiers into two groups: top (3,33) and others
    df_top = df[df["model_classifier"].isin([3, 33])]
    df_bot = df[~df["model_classifier"].isin([3, 33])]
    orders = [
        sorted(df_top["model_classifier"].unique()),
        sorted(df_bot["model_classifier"].unique()),
    ]

    width = 0.6 / (len(primary_vals) * len(layers))
    pos_top, map_top = compute_positions(
        len(orders[0]), primary_vals, layers, width, block_gap
    )
    pos_bot, map_bot = compute_positions(
        len(orders[1]), primary_vals, layers, width, block_gap
    )

    if primary_col == "n_features":
        # Special layout for feature count differences
        pivot = df.pivot_table(
            index=["model_classifier", "n_layers"],
            columns="n_features",
            values="f1",
            aggfunc="mean",
        )
        percent_diff = (pivot[16] - pivot[768]) / pivot[16] * 100

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(6, 10),
            dpi=300,
            sharey="row",
            gridspec_kw={
                "height_ratios": [len(orders[0]), len(orders[1])],
                "width_ratios": [3, 1],
            },
        )
        titles = ["Modelos 3 e 33", "Demais Modelos"]

        # Center titles over each row's two subplots
        for row_idx, title in enumerate(titles):
            pos0 = axes[row_idx, 0].get_position()
            pos1 = axes[row_idx, 1].get_position()
            mid_fig_x = (pos0.x0 + pos1.x1) / 2 - 0.05
            rel_x = (mid_fig_x - pos0.x0) / pos0.width
            axes[row_idx, 0].set_title(title, x=rel_x, y=1.02, fontsize="large")

        # Plot left and right panels per row
        for ax_left, ax_right, subset, order, positions, pmap in [
            (axes[0, 0], axes[0, 1], df_top, orders[0], pos_top, map_top),
            (axes[1, 0], axes[1, 1], df_bot, orders[1], pos_bot, map_bot),
        ]:
            plot_panel(
                ax_left,
                subset,
                order,
                primary_col,
                primary_vals,
                layers,
                positions,
                pmap,
                width,
                color_map,
                hatch_map,
            )

            # Right panel: difference bars
            ax_right.set_axisbelow(True)
            ax_right.grid(axis="x", color="#CCCCCC", ls="--", lw=0.8)
            for y in np.arange(len(order) + 1) - 0.5:
                ax_right.axhline(y, color="black", lw=1, zorder=0)

            y_indices = np.arange(len(order))
            bar_height = 0.3
            offsets = {layers[0]: -bar_height / 2, layers[1]: bar_height / 2}
            all_values: List[float] = []

            # Plot bars for each layer where data is present
            for layer in layers:
                try:
                    layer_series = percent_diff.xs(layer, level=1)
                except KeyError:
                    continue
                layer_full = layer_series.reindex(order)
                present = layer_full.notna()
                values = layer_full.fillna(0).values
                ys = y_indices + offsets[layer]

                ys_present = ys[present.values]
                vals_present = values[present.values]
                colors = ["green" if v > 0 else "red" for v in vals_present]

                if layer == layers[0]:
                    ax_right.barh(
                        ys_present,
                        vals_present,
                        height=bar_height,
                        facecolor=colors,
                        edgecolor="black",
                        label=f"{layer} camada",
                    )
                else:
                    ax_right.barh(
                        ys_present,
                        vals_present,
                        height=bar_height,
                        facecolor=colors,
                        edgecolor="black",
                        hatch=hatch_map[layer],
                        label=f"{layer} camadas",
                    )
                all_values.extend(vals_present.tolist())

            # Configure x-axis limits and zero line
            ax_right.axvline(0, ls="--", color="gray")
            xmin = min(0, min(all_values) if all_values else 0)
            xmax = max(0, max(all_values) if all_values else 0)
            margin = max(abs(xmin), abs(xmax)) * 0.13
            ax_right.set_xlim(xmin - margin, xmax + margin)
            ax_right.set_ylim(ax_left.get_ylim())
            ax_right.set_yticks([])
            ax_right.tick_params(axis="y", which="both", length=0)
            ax_right.set_xlabel("Δ F1 (%)\n(768→16)")

            # Add text labels on bars where appropriate
            x_range = ax_right.get_xlim()[1] - ax_right.get_xlim()[0]
            threshold = x_range * 0.05
            for layer in layers:
                try:
                    series_layer = percent_diff.xs(layer, level=1)
                except KeyError:
                    continue
                series_full = series_layer.reindex(order)
                values = series_full.fillna(0).values
                ys = y_indices + offsets[layer]
                present_mask = series_full.notna()
                for v, y, present_flag in zip(values, ys, present_mask, strict=True):
                    if not present_flag:
                        continue
                    text = f"{int(v)}"
                    if abs(v) >= threshold:
                        x_text = v / 2
                        ha = "center"
                        color = "white"
                    else:
                        x_text = v + threshold
                        ha = "left"
                        color = "black"
                    ax_right.text(
                        x_text,
                        y,
                        text,
                        va="center",
                        ha=ha,
                        fontsize="x-small",
                        color=color,
                    )

        # Set y-ticks and labels for left panels
        for idx, order in enumerate(orders):
            axes[idx, 0].set_yticks(np.arange(len(order)))
            axes[idx, 0].set_yticklabels(order)

        # Create legend for the bottom-left panel
        primary_handles = [
            Patch(facecolor=color_map[p], label=attr_label_fmt.format(p))
            for p in primary_vals
        ]
        layer_handles = [
            Patch(
                facecolor="white",
                hatch=hatch_map[layer],
                edgecolor="black",
                label=f"{layer} camada{'s' if layer > 1 else ''}",
            )
            for layer in layers
        ]
        x_handle = Line2D(
            [],
            [],
            marker="x",
            color="#ed1fed",
            ls="",
            markersize=12,
            label="Sem diferença entre camadas",
        )
        arrow_handle = Line2D([], [], color="#0b31bf", lw=1.5, label=arrow_label)
        legend_handles = primary_handles + layer_handles + [x_handle, arrow_handle]
        axes[1, 0].legend(
            handles=legend_handles,
            title="Legenda",
            loc="lower left",
            fontsize="small",
            framealpha=1.0,
            edgecolor="black",
            fancybox=True,
        )

    else:
        # Standard layout for other primary attributes
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(6, 10),
            dpi=300,
            sharex=False,
            gridspec_kw={"height_ratios": [len(orders[0]), len(orders[1])]},
        )
        for ax, subset, order, positions, pmap in [
            (ax_top, df_top, orders[0], pos_top, map_top),
            (ax_bottom, df_bot, orders[1], pos_bot, map_bot),
        ]:
            plot_panel(
                ax,
                subset,
                order,
                primary_col,
                primary_vals,
                layers,
                positions,
                pmap,
                width,
                color_map,
                hatch_map,
            )

        ax_top.set_title("Models 3 e 33")
        ax_bottom.set_title("Demais Modelos")

        primary_handles = []
        if attr_label_fmt:
            primary_handles = [
                Patch(facecolor=color_map[p], label=attr_label_fmt.format(p))
                for p in primary_vals
            ]
        layer_handles = [
            Patch(
                facecolor="white",
                hatch=hatch_map[layer],
                edgecolor="black",
                label=f"{layer} camada{'s' if layer > 1 else ''}",
            )
            for layer in layers
        ]
        x_handle = Line2D(
            [],
            [],
            marker="x",
            color="#ed1fed",
            ls="",
            markersize=12,
            label="Sem diferença entre camadas",
        )
        legend_handles = (
            (primary_handles + layer_handles + [x_handle])
            if primary_handles
            else layer_handles + [x_handle]
        )
        if arrow_label:
            legend_handles.append(
                Line2D([], [], color="#0b31bf", lw=1.5, label=arrow_label)
            )

        ax_bottom.legend(
            handles=legend_handles,
            title="Legenda",
            loc="lower left",
            fontsize="small",
            framealpha=1.0,
            edgecolor="black",
            fancybox=True,
        )

    plt.tight_layout()
    # Save figure in PNG and PGF formats
    for ext in ("png", "pgf"):
        output_path = Path("../../figures") / f"{out_name}_{dataset}.{ext}"
        # fig.savefig(output_path, bbox_inches='tight')

    plt.show()
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard")
    layers = sorted(read_summary()["n_layers"].unique())  # e.g., [1, 10]
    hatch_map = {layers[0]: "", layers[1]: "//"}

    # A) Varying number of features (n_features)
    TARGET = "tomaarsen/mpnet-base-nli-matryoshka"
    feature_values = [16, 32, 768]
    df_ft = read_summary()
    df_ft = df_ft[
        (df_ft["model_transformer"] == TARGET)
        & (df_ft["n_features"].isin(feature_values))
        & (df_ft["model_classifier"].isin(classical_models + quantum_models))
    ]
    colors_ft = plt.cm.twilight_shifted(np.linspace(0.60, 0.80, len(feature_values)))
    color_map_ft = {val: colors_ft[i] for i, val in enumerate(feature_values)}
    for ds in DATASETS:
        generate_figure(
            df=df_ft[df_ft["dataset"] == ds],
            dataset=ds,
            primary_col="n_features",
            primary_vals=feature_values,
            color_map=color_map_ft,
            arrow_label="Sem diferença entre n° de atributos",
            attr_label_fmt="{} atributos",
            out_name="n_features",
            layers=layers,
            hatch_map=hatch_map,
        )

    # B) Varying transformer models with fixed feature count
    df_tr = read_summary()
    df_tr = df_tr[
        (df_tr["n_features"] == 16)
        & (df_tr["model_classifier"].isin(classical_models + quantum_models))
    ]
    transformers = df_tr["model_transformer"].unique().tolist()
    colors_tr = plt.cm.twilight_shifted(np.linspace(0.15, 0.45, len(transformers)))
    color_map_tr = {t: colors_tr[i] for i, t in enumerate(transformers)}
    for ds in DATASETS:
        generate_figure(
            df=df_tr[df_tr["dataset"] == ds],
            dataset=ds,
            primary_col="model_transformer",
            primary_vals=transformers,
            color_map=color_map_tr,
            arrow_label="Sem diferença entre transformers",
            attr_label_fmt="{}",
            out_name="transformers",
            layers=layers,
            hatch_map=hatch_map,
        )

    # C) Final analysis: 1 vs 10 layers for all models
    DATASETS_FINAL = (*DATASETS, "sst")
    MODELS_ALL = (
        classical_models
        + quantum_models
        + classical_ensemble_models
        + quantum_ensemble_models
    )
    df_final = read_summary()
    df_final = df_final[
        (df_final["n_features"] == 16)
        & (df_final["model_transformer"] == TARGET)
        & (df_final["model_classifier"].isin(MODELS_ALL))
    ]
    color_map_final = {TARGET: "#1f77b4"}
    batch_override = {"sst": 512}
    for ds in DATASETS_FINAL:
        mask = df_final["dataset"].eq(ds)
        if ds in batch_override:
            mask &= df_final["batch_size"].eq(batch_override[ds])
        df_subset = df_final[mask]
        generate_figure(
            df=df_subset,
            dataset=ds,
            primary_col="model_transformer",
            primary_vals=[TARGET],
            color_map=color_map_final,
            arrow_label="",
            attr_label_fmt="",
            out_name="f1_layers",
            layers=layers,
            hatch_map=hatch_map,
        )
