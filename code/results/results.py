"""
Unified box-and-whisker figure generator.

* Case A – colour = ``model_transformer`` (n_features = 16)
* Case B – colour = ``n_features`` (single transformer)

Keeps:
    • Wilcoxon + Holm annotations (“×” and “↔” arrows)
    • light separators between colour blocks
    • vertical dotted grid only
    • unified x-limits across sub-panels
    • opaque legend in lower-left

Author : <your-name>
Date    : 2025-06-11
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence

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


def pairwise_wilcoxon_holm(df: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """All-pairs Wilcoxon with Holm correction."""
    if df.shape[1] < 2:
        return pd.DataFrame(columns=["i", "j", "stat", "p_raw", "p_holm", "reject"])
    records: list[tuple[str, str, float, float]] = []
    for a, b in itertools.combinations(df.columns, 2):
        x, y = df[a].values, df[b].values
        stat, p = wilcoxon(x, y) if len(x) == len(y) and (x != y).any() else (np.nan, 1)
        records.append((a, b, float(stat), float(p)))
    pvals = [r[3] for r in records]
    reject, p_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    out = pd.DataFrame.from_records(
        records, columns=["i", "j", "stat", "p_raw"]
    ).assign(p_holm=p_holm, reject=reject)
    return out


def compute_positions(
    n_rows: int,
    primaries: Sequence[str | int],
    layers: Sequence[int],
    width: float,
    gap: float,
) -> tuple[list[np.ndarray], dict[tuple[str | int, int], np.ndarray]]:
    combos = [(p, layer) for p in primaries for layer in layers]
    pos, mapping = [], {}
    for pi, _ in enumerate(primaries):
        offset = pi * (len(layers) * width + gap) - gap
        for li, _ in enumerate(layers):
            arr = np.arange(n_rows) - 0.3 + width / 2 + li * width + offset
            pos.append(arr)
    mapping = {c: pos[i] for i, c in enumerate(combos)}
    return pos, mapping


def add_inner_separators(
    ax: Axes,
    pos_map: dict[tuple[str | int, int], np.ndarray],
    primaries: Sequence[str | int],
    layers: Sequence[int],
    n_rows: int,
) -> None:
    for r in range(n_rows):
        for i in range(len(primaries) - 1):
            y_a = pos_map[(primaries[i], layers[-1])][r]
            y_b = pos_map[(primaries[i + 1], layers[0])][r]
            ax.axhline((y_a + y_b) / 2, color="#6A6A6A", lw=0.8, zorder=0, alpha=0.7)


def plot_panel(
    ax: Axes,
    df: pd.DataFrame,
    classifier_order: Sequence[int],
    primary_col: str,
    primary_vals: Sequence[str | int],
    layers: Sequence[int],
    positions: list[np.ndarray],
    pos_map: dict[tuple[str | int, int], np.ndarray],
    width: float,
    color_map: dict[str | int, np.ndarray],
    hatch_map: dict[int, str],
    *,
    alpha: float = 0.05,
) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#CCCCCC", ls="--", lw=0.8)
    for y in np.arange(len(classifier_order) + 1) - 0.5:
        ax.axhline(y, color="black", lw=1, zorder=0)

    combos = [(p, layer) for p in primary_vals for layer in layers]
    for i, (p, l) in enumerate(combos):
        raw_vecs = [
            df[
                (df["model_classifier"] == clf)
                & (df[primary_col] == p)
                & (df["n_layers"] == l)
            ]["f1"].to_numpy(dtype=float)
            for clf in classifier_order
        ]
        data = [v[np.isfinite(v)] for v in raw_vecs]
        if all(len(v) == 0 for v in data):
            continue
        data = [v if len(v) else np.array([np.nan]) for v in data]

        bp = ax.boxplot(
            data,
            positions=positions[i],
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
        for box, vec in zip(bp["boxes"], data, strict=True):
            if np.isnan(vec).all():
                box.set_visible(False)
            else:
                box.set_facecolor(color_map[p])
                box.set_hatch(hatch_map[l])

    add_inner_separators(ax, pos_map, primary_vals, layers, len(classifier_order))

    # ── RESTAURADA: ANOTAÇÕES DE WILCOXON + HOLM ────────────────────────
    for r, clf in enumerate(classifier_order):
        piv = df[df["model_classifier"] == clf].pivot(
            index="seed", columns=[primary_col, "n_layers"], values="f1"
        )
        piv.columns = pd.MultiIndex.from_tuples(piv.columns)

        # × : mesma primary, camadas diferentes
        res = pairwise_wilcoxon_holm(piv, alpha=alpha)
        for _, row in res[~res["reject"]].iterrows():
            (p0, l0), (p1, l1) = row["i"], row["j"]
            if p0 == p1 and l0 != l1:
                y0, y1 = pos_map[(p0, l0)][r], pos_map[(p1, l1)][r]
                x_mid = (piv[(p0, l0)].median() + piv[(p1, l1)].median()) / 2
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

        # ↔ : mesmo layer, primaries diferentes
        for layer in layers:
            if layer not in piv.columns.get_level_values(1):
                continue
            sub = piv.xs(layer, level=1, axis=1)
            res_p = pairwise_wilcoxon_holm(sub, alpha=alpha)
            for _, row in res_p[~res_p["reject"]].iterrows():
                p0, p1 = row["i"], row["j"]
                y0, y1 = pos_map[(p0, layer)][r], pos_map[(p1, layer)][r]
                m0, m1 = piv[(p0, layer)].median(), piv[(p1, layer)].median()
                ax.annotate(
                    "",
                    (m1, y1),
                    (m0, y0),
                    arrowprops={"arrowstyle": "<->", "color": "#0b31bf", "lw": 1.5},
                    zorder=5,
                )
    # ────────────────────────────────────────────────────────────────────

    ax.set_ylim(len(classifier_order) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_xlabel("F1")


# ──────────────────────────── figure driver ────────────────────────────────────


def generate_figure(
    df: pd.DataFrame,
    dataset: str,
    primary_col: str,
    primary_vals: Sequence[str | int],
    color_map: dict[str | int, np.ndarray],
    arrow_label: str,
    attr_label_fmt: str,
    out_name: str,
    *,
    layers: Sequence[int],
    hatch_map: dict[int, str],
    block_gap: float = 0.1,
) -> None:
    """Create one PDF (top 3/33 vs. others) for the chosen ‘primary’."""
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

    # ───── Layout especial para n_features ─────────────────────────────
    if primary_col == "n_features":
        # calcula as médias e a diferença percentual (16→768)
        pivot = df.pivot_table(
            index=["model_classifier", "n_layers"],
            columns="n_features",
            values="f1",
            aggfunc="mean",
        )
        diff = (pivot[16] - pivot[768]) / pivot[16] * 100  # MultiIndex

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
        # ── CENTRALIZA TÍTULOS SOBRE OS DOIS SUBPLOTS ────────────────────────
        for row, title in enumerate(titles):
            pos0 = axes[row, 0].get_position()
            pos1 = axes[row, 1].get_position()
            # centro em coords de figura
            mid_fig = (pos0.x0 + pos1.x1) / 2
            # converte para fração do eixo esquerdo
            x_rel = (mid_fig - pos0.x0) / pos0.width
            axes[row, 0].set_title(
                title,
                x=x_rel,
                y=1,
                fontsize="large",
            )
        # ────────────────────────────────────────────────────────────────────

        for _, (ax_l, ax_r, subset, order, pos, pmap) in enumerate(
            [
                (axes[0, 0], axes[0, 1], df_top, orders[0], pos_top, map_top),
                (axes[1, 0], axes[1, 1], df_bot, orders[1], pos_bot, map_bot),
            ]
        ):
            # Boxplot à esquerda
            plot_panel(
                ax_l,
                subset,
                order,
                primary_col,
                primary_vals,
                layers,
                pos,
                pmap,
                width,
                color_map,
                hatch_map,
            )
            # ax_l.set_title(titles[row])

            # Grid e linhas horizontais na direita
            ax_r.set_xlabel("Δ F1 (%)\n(768→16)")
            ax_r.set_axisbelow(True)
            ax_r.grid(axis="x", color="#CCCCCC", ls="--", lw=0.8)
            for y in np.arange(len(order) + 1) - 0.5:
                ax_r.axhline(y, color="black", lw=1, zorder=0)

            # Barras de diferença
            y_idx = np.arange(len(order))
            bar_h = 0.3
            offsets = {layers[0]: -bar_h / 2, layers[1]: +bar_h / 2}
            all_vals: list[float] = []

            for layer in layers:
                try:
                    series_layer = diff.xs(layer, level=1)
                except KeyError:
                    continue
                vals = series_layer.reindex(order).fillna(0).values
                ys = y_idx + offsets[layer]
                if layer == layers[0]:
                    ax_r.barh(
                        ys,
                        vals,
                        height=bar_h,
                        facecolor="red",
                        edgecolor="red",
                        label=f"{layer} camada",
                    )
                else:
                    ax_r.barh(
                        ys,
                        vals,
                        height=bar_h,
                        facecolor="red",
                        edgecolor="black",
                        hatch=hatch_map[layer],
                        label=f"{layer} camadas",
                    )
                all_vals.extend(vals.tolist())

            ax_r.axvline(0, ls="--", color="gray")
            xmin = min(0, min(all_vals)) if all_vals else 0
            xmax = max(0, max(all_vals)) if all_vals else 0
            margin = max(abs(xmin), abs(xmax)) * 0.1
            ax_r.set_xlim(xmin - margin, xmax + margin)
            ax_r.set_ylim(ax_l.get_ylim())
            ax_r.set_yticks([])
            ax_r.tick_params(
                axis="y", which="both", length=0
            )  # remove os traços do eixo y

        for idx, order in enumerate(orders):
            axes[idx, 0].set_yticks(np.arange(len(order)))
            axes[idx, 0].set_yticklabels(order)
        # 1) Legenda original na esquerda (apenas bottom-left)
        handles_primary = [
            Patch(facecolor=color_map[p], label=attr_label_fmt.format(p))
            for p in primary_vals
        ]
        handles_layers = [
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
        handles = handles_primary + handles_layers + [x_handle, arrow_handle]
        axes[1, 0].legend(
            handles=handles,
            title="Legenda",
            loc="lower left",
            fontsize="small",
            framealpha=1.0,
            edgecolor="black",
            fancybox=True,
        )

        # 2) Legenda de diferenças na direita (apenas bottom-right, mais à esquerda)
        axes[1, 1].legend(
            title="Legenda",
            loc="lower left",
            fontsize="small",
            framealpha=1.0,
            edgecolor="black",
        )

    # ───── Layout padrão para demais primaries ───────────────────────────
    else:
        fig, (ax_l, ax_b) = plt.subplots(
            2,
            1,
            figsize=(6, 10),
            dpi=300,
            sharex=False,
            gridspec_kw={"height_ratios": [len(orders[0]), len(orders[1])]},
        )
        for ax, subset, order, pos, pmap in [
            (ax_l, df_top, orders[0], pos_top, map_top),
            (ax_b, df_bot, orders[1], pos_bot, map_bot),
        ]:
            plot_panel(
                ax,
                subset,
                order,
                primary_col,
                primary_vals,
                layers,
                pos,
                pmap,
                width,
                color_map,
                hatch_map,
            )
        ax_l.set_title("Modelos 3 e 33")
        ax_b.set_title("Demais Modelos")

        handles_primary = []
        if attr_label_fmt:
            handles_primary = [
                Patch(facecolor=color_map[p], label=attr_label_fmt.format(p))
                for p in primary_vals
            ]
        handles_layers = [
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
        handles = (
            handles_primary + handles_layers + [x_handle]
            if handles_primary
            else handles_layers + [x_handle]
        )
        if arrow_label:
            handles.append(Line2D([], [], color="#0b31bf", lw=1.5, label=arrow_label))
        ax_b.legend(
            handles=handles,
            title="Legenda",
            loc="lower left",
            fontsize="small",
            framealpha=1.0,
            edgecolor="black",
            fancybox=True,
        )

    plt.tight_layout()
    out_path = Path("../../figures") / f"{out_name}_{dataset}.pgf"
    # fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────── main ─────────────────────────────────────────
if __name__ == "__main__":
    DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard")
    layers = sorted(read_summary()["n_layers"].unique())  # [1, 10]
    hatch_map = {layers[0]: "", layers[1]: "//"}

    # A) transformers (n_features=16)
    df_tr = read_summary()
    df_tr = df_tr[
        (df_tr["n_features"] == 16)
        & (df_tr["model_classifier"].isin(classical_models + quantum_models))
    ]
    transformers = df_tr["model_transformer"].unique().tolist()
    colors_tr = plt.cm.Set2(np.linspace(0, 0.5, len(transformers)))
    cmap_tr = {t: colors_tr[i] for i, t in enumerate(transformers)}
    for ds in DATASETS:
        generate_figure(
            df=df_tr[df_tr["dataset"] == ds],
            dataset=ds,
            primary_col="model_transformer",
            primary_vals=transformers,
            color_map=cmap_tr,
            arrow_label="Sem diferença entre transformers",
            attr_label_fmt="{}",
            out_name="transformers",
            layers=layers,
            hatch_map=hatch_map,
        )

    # B) n_features: 16/32/768
    TARGET = "tomaarsen/mpnet-base-nli-matryoshka"
    feats = [16, 32, 768]
    df_ft = read_summary()
    df_ft = df_ft[
        (df_ft["model_transformer"] == TARGET)
        & (df_ft["n_features"].isin(feats))
        & (df_ft["model_classifier"].isin(classical_models + quantum_models))
    ]
    colors_ft = plt.cm.Set2(np.linspace(0.5, 1, len(feats)))
    cmap_ft = {f: colors_ft[i] for i, f in enumerate(feats)}
    for ds in DATASETS:
        generate_figure(
            df=df_ft[df_ft["dataset"] == ds],
            dataset=ds,
            primary_col="n_features",
            primary_vals=feats,
            color_map=cmap_ft,
            arrow_label="Sem diferença entre n° de atributos",
            attr_label_fmt="{} atributos",
            out_name="n_features",
            layers=layers,
            hatch_map=hatch_map,
        )

    # C) análise final: 1 vs 10 camadas
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
    cmap_final = {TARGET: "#1f77b4"}
    BATCH_OVERRIDE = {"sst": 512}
    for ds in DATASETS_FINAL:
        mask = df_final["dataset"].eq(ds)
        if ds in BATCH_OVERRIDE:
            mask &= df_final["batch_size"].eq(BATCH_OVERRIDE[ds])
        df_final_figura = df_final[mask]
        generate_figure(
            df=df_final_figura,
            dataset=ds,
            primary_col="model_transformer",
            primary_vals=[TARGET],
            color_map=cmap_final,
            arrow_label="",
            attr_label_fmt="",
            out_name="f1_layers",
            layers=layers,
            hatch_map=hatch_map,
        )
