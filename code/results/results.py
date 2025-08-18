"""
Unified box-and-whisker figure generator (grupos por circuitos).

Autor: <Your Name>
Data: 2025-06-11

Este módulo gera figuras unificadas de boxplot com anotações de Wilcoxon e correção de Holm.
Suporta dois atributos principais:
- Caso A: colori por 'model_transformer' (e.g., n_features = 16)
- Caso B: colori por 'n_features' (transformer único)

Recursos:
- Anotações Wilcoxon + Holm ('×' para sem diferença entre layers, setas '↔' para sem diferença entre primários)
- Separadores claros entre blocos de cores
- Linhas de grade verticais pontilhadas
- Limites de x unificados por subpainéis
- Legenda opaca no canto inferior esquerdo
- **Novo**: gráficos gerados por grupos específicos de circuitos (IDs), sem a separação fixa de 3 e 33.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cbook import boxplot_stats
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

# >>> NOVO: grupos fixos de circuitos solicitados
CIRCUIT_GROUPS = [
    list(range(1, 7)),  # 1–6
    list(range(7, 12)),  # 7–11
    list(range(12, 16)),  # 12–15
    [32, 33, 36],  # 32, 33, 36
    [34, 35, 37, 38],  # 34, 35, 37, 38
    list(range(16, 22)),  # 16–21
    list(range(22, 32)),  # 22–31
    list(range(39, 45)),  # 39–44
]
ALLOWED_CLASSIFIERS = sorted({c for grp in CIRCUIT_GROUPS for c in grp})


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
    """
    # Grid e separadores de linhas
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#CCCCCC", ls="--", lw=0.8, zorder=0)
    for y in np.arange(len(classifier_order) + 1) - 0.5:
        ax.axhline(y, color="black", lw=1, zorder=1)

    # Desenha cada grupo de boxplots
    combos = [(p, layer) for p in primary_vals for layer in layers]
    for idx, (p, layer) in enumerate(combos):
        # Extrai F1 para cada classifier
        samples = []
        for clf in classifier_order:
            mask = (
                (df["model_classifier"] == clf)
                & (df[primary_col] == p)
                & (df["n_layers"] == layer)
            )
            values = df.loc[mask, "f1"].to_numpy(dtype=float)
            samples.append(values[np.isfinite(values)])

        if all(len(s) == 0 for s in samples):
            continue
        samples = [s if len(s) else np.array([np.nan]) for s in samples]

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
        # Estilo das caixas
        for box, vals in zip(bp["boxes"], samples, strict=True):
            if np.isnan(vals).all():
                box.set_visible(False)
            else:
                box.set_facecolor(color_map[p])
                box.set_hatch(hatch_map.get(layer, ""))

    # Separadores internos entre primários
    add_inner_separators(ax, pos_map, primary_vals, layers, len(classifier_order))

    # Anotações Wilcoxon + Holm
    for row_idx, clf in enumerate(classifier_order):
        pivot = df[df["model_classifier"] == clf].pivot(
            index="seed", columns=[primary_col, "n_layers"], values="f1"
        )
        pivot.columns = pd.MultiIndex.from_tuples(pivot.columns)

        # '×' para mesmo primário e layers diferentes
        results = pairwise_wilcoxon_holm(pivot, alpha=alpha)
        for _, r in results[~results["reject"]].iterrows():
            (p0, l0), (p1, l1) = r["i"], r["j"]
            if p0 == p1 and l0 != l1:
                y0 = pos_map[(p0, l0)][row_idx]
                y1 = pos_map[(p1, l1)][row_idx]
                m0 = pivot[(p0, l0)].median()
                m1 = pivot[(p1, l1)].median()
                x_mid = (m0 + m1) / 2
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

        # '↔' para mesmo layer e primários diferentes
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

    ax.set_ylim(len(classifier_order) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_xlabel("F1")
    ax.set_ylabel("Circuito ID")


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
    # >>> NOVO: grupos de classificadores
    classifier_groups: Sequence[Sequence[int]] | None = None,
) -> None:
    """
    Gera figuras de boxplot (e barras de diferença quando aplicável) por grupo de classificadores.
    Cada grupo produz UMA figura. A antiga separação fixa (3 e 33) foi removida.
    """
    # Filtra apenas os classificadores permitidos
    df = df[df["model_classifier"].isin(ALLOWED_CLASSIFIERS)].copy()

    # Se não vier grupos, usa um grupo único com o que houver no df
    if not classifier_groups:
        classifier_groups = [sorted(df["model_classifier"].unique())]

    width = 0.6 / (len(primary_vals) * len(layers))

    # Pré-cálculos para casos com barras de diferença
    percent_diff_features = None
    percent_diff_transformers = None
    best_ref_transformers = None

    if primary_col == "n_features":
        piv = df.pivot_table(
            index=["model_classifier", "n_layers"],
            columns="n_features",
            values="f1",
            aggfunc="mean",
        )
        if 16 in piv.columns and 768 in piv.columns:
            percent_diff_features = (piv[16] - piv[768]) / piv[16] * 100

    if primary_col == "model_transformer" and len(primary_vals) > 1:
        piv = df.pivot_table(
            index=["model_classifier", "n_layers"],
            columns="model_transformer",
            values="f1",
            aggfunc="mean",
        )
        try:
            global TARGET  # usa a variável global já presente no script
            if TARGET in piv.columns:
                best_ref_transformers = piv.idxmax(axis=1)
                percent_diff_transformers = (
                    (piv[TARGET] - piv.max(axis=1)) / piv.max(axis=1) * 100
                )
        except NameError:
            # Caso TARGET não exista, não cria painel de diferença
            percent_diff_transformers = None
            best_ref_transformers = None

    # Loop: uma figura por grupo
    for g_idx, group_ids in enumerate(classifier_groups, start=1):
        order = [c for c in sorted(df["model_classifier"].unique()) if c in group_ids]
        if not order:
            continue

        positions, pos_map = compute_positions(
            n_rows=len(order),
            primaries=primary_vals,
            layers=layers,
            width=width,
            gap=block_gap,
        )

        has_diff_panel = (
            primary_col == "n_features" and percent_diff_features is not None
        ) or (
            primary_col == "model_transformer"
            and len(primary_vals) > 1
            and percent_diff_transformers is not None
        )

        height = max(12, 1.1 * len(order) + 2.0)
        if has_diff_panel:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(7.2, height),
                dpi=300,
                gridspec_kw={"width_ratios": [3, 1]},
            )
            ax_left, ax_right = axes
        else:
            fig, ax_left = plt.subplots(1, 1, figsize=(7.2, height), dpi=300)
            ax_right = None

        # Painel esquerdo: boxplots
        plot_panel(
            ax_left,
            df,
            order,
            primary_col,
            primary_vals,
            layers,
            positions,
            pos_map,
            width,
            color_map,
            hatch_map,
        )
        # ax_left.set_title(f"Grupo {g_idx}: {', '.join(map(str, group_ids))}")

        # Painel direito: barras de diferença (quando aplicável)
        if has_diff_panel and ax_right is not None:
            ax_right.set_axisbelow(True)
            ax_right.grid(axis="x", color="#CCCCCC", ls="--", lw=0.8)
            for y in np.arange(len(order) + 1) - 0.5:
                ax_right.axhline(y, color="black", lw=1, zorder=0)

            y_idx = np.arange(len(order))
            bar_h = 0.3
            offsets = (
                {layers[0]: -bar_h / 2, layers[1]: bar_h / 2}
                if len(layers) >= 2
                else {layers[0]: 0.0}
            )
            all_vals: List[float] = []

            if primary_col == "n_features" and percent_diff_features is not None:
                for layer in layers:
                    try:
                        layer_series = percent_diff_features.xs(layer, level=1)
                    except KeyError:
                        continue
                    layer_full = layer_series.reindex(order)
                    present = layer_full.notna()
                    vals = layer_full.fillna(0).values
                    ys = y_idx + offsets.get(layer, 0.0)

                    ys_p = ys[present.values]
                    vals_p = vals[present.values]
                    colors = ["green" if v > 0 else "red" for v in vals_p]

                    ax_right.barh(
                        ys_p,
                        vals_p,
                        height=bar_h,
                        facecolor=colors,
                        edgecolor="black",
                        hatch=hatch_map.get(layer, ""),
                        label=f"{layer} camada{'s' if layer>1 else ''}",
                    )
                    all_vals.extend(vals_p.tolist())

                ax_right.axvline(0, ls="--", color="gray")
                xmin = min(0, min(all_vals) if all_vals else 0)
                xmax = max(0, max(all_vals) if all_vals else 0)
                margin = max(abs(xmin), abs(xmax)) * 0.13
                ax_right.set_xlim(xmin - margin, xmax + margin)
                ax_right.set_ylim(ax_left.get_ylim())
                ax_right.set_yticks([])
                ax_right.tick_params(axis="y", which="both", length=0)
                ax_right.set_xlabel("Δ F1 (%)\n(768→16)")

                # Rótulos de texto nos bares (como no original)
                x_range = ax_right.get_xlim()[1] - ax_right.get_xlim()[0]
                threshold = x_range * 0.05
                for layer in layers:
                    try:
                        series_layer = percent_diff_features.xs(layer, level=1)
                    except KeyError:
                        continue
                    series_full = series_layer.reindex(order)
                    values = series_full.fillna(0).values
                    ys = y_idx + offsets.get(layer, 0.0)
                    present_mask = series_full.notna().values
                    for v, y, present_flag in zip(
                        values, ys, present_mask, strict=True
                    ):
                        if not present_flag:
                            continue
                        text = f"{int(np.round(v))}"
                        if abs(v) >= threshold:
                            x_text = v / 2
                            ha = "center"
                            color_txt = "white"
                        else:
                            x_text = v + threshold
                            ha = "left"
                            color_txt = "black"
                        ax_right.text(
                            x_text,
                            y,
                            text,
                            va="center",
                            ha=ha,
                            fontsize="x-small",
                            color=color_txt,
                        )

            elif (
                primary_col == "model_transformer"
                and len(primary_vals) > 1
                and percent_diff_transformers is not None
            ):
                for layer in layers:
                    try:
                        layer_diff = percent_diff_transformers.xs(layer, level=1)
                        ref_series = best_ref_transformers.xs(layer, level=1)
                    except KeyError:
                        continue
                    layer_full = layer_diff.reindex(order)
                    ref_full = ref_series.reindex(order)
                    present = layer_full.notna()
                    vals = layer_full.fillna(0).values
                    ys = y_idx + offsets.get(layer, 0.0)

                    ys_p = ys[present.values]
                    vals_p = vals[present.values]
                    # Cor da barra pelo "best model"
                    colors = [color_map[model] for model in ref_full[present].tolist()]

                    ax_right.barh(
                        ys_p,
                        vals_p,
                        height=bar_h,
                        facecolor=colors,
                        edgecolor="black",
                        hatch=hatch_map.get(layer, ""),
                        label=f"{layer} camada{'s' if layer>1 else ''}",
                    )
                    all_vals.extend(vals_p.tolist())

                ax_right.axvline(0, ls="--", color="gray")
                xmin = min(0, min(all_vals) if all_vals else 0)
                xmax = max(0, max(all_vals) if all_vals else 0)
                margin = max(abs(xmin), abs(xmax)) * 0.13
                ax_right.set_xlim(xmin - margin, xmax + margin)
                ax_right.set_ylim(ax_left.get_ylim())
                ax_right.set_yticks([])
                ax_right.tick_params(axis="y", which="both", length=0)
                ax_right.set_xlabel("Δ F1 (%)\n(Melhor → Matryoshka)")

                # Rótulos de texto nos bares (caso transformers)
                x_range = ax_right.get_xlim()[1] - ax_right.get_xlim()[0]
                threshold = x_range * 0.05
                for layer in layers:
                    try:
                        layer_diff = percent_diff_transformers.xs(layer, level=1)
                        ref_series = best_ref_transformers.xs(
                            layer, level=1
                        )  # vencedores por (classifier, layer)
                    except KeyError:
                        continue

                    layer_full = layer_diff.reindex(order)
                    ref_full = ref_series.reindex(order)

                    ys = y_idx + offsets.get(layer, 0.0)
                    present_mask = layer_full.notna().values
                    vals = layer_full.fillna(0).values
                    refs = (
                        ref_full.fillna("").astype(str).values
                    )  # nomes dos vencedores

                    for v, y, present_flag, ref_name in zip(
                        vals, ys, present_mask, refs, strict=True
                    ):
                        if not present_flag:
                            continue

                        text = f"{int(np.round(v))}"
                        wide = abs(v) >= threshold

                        # posição do texto
                        if wide:
                            x_text = v / 2
                            ha = "center"
                        else:
                            x_text = v + threshold
                            ha = "left"

                        # cor do texto:
                        # - barras estreitas -> preto
                        # - barras largas -> branco, EXCETO se o vencedor for all-mpnet -> preto
                        if wide and "all-mpnet" in ref_name.lower():
                            color_txt = "black"
                        else:
                            color_txt = "white" if wide else "black"

                        ax_right.text(
                            x_text,
                            y,
                            text,
                            va="center",
                            ha=ha,
                            fontsize="x-small",
                            color=color_txt,
                        )

            # Legenda no painel esquerdo
            primary_handles = (
                [
                    Patch(facecolor=color_map[p], label=attr_label_fmt.format(p))
                    for p in primary_vals
                ]
                if attr_label_fmt
                else []
            )
            layer_handles = [
                Patch(
                    facecolor="white",
                    hatch=hatch_map.get(layer, ""),
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
                else (layer_handles + [x_handle])
            )
            if arrow_label:
                legend_handles.append(
                    Line2D([], [], color="#0b31bf", lw=1.5, label=arrow_label)
                )
            ax_left.legend(
                handles=legend_handles,
                title="Legenda",
                loc="lower left",
                fontsize="small",
                framealpha=0.85,
                edgecolor="black",
                fancybox=True,
            )
        else:
            # Legenda quando não há painel direito
            primary_handles = (
                [
                    Patch(facecolor=color_map[p], label=attr_label_fmt.format(p))
                    for p in primary_vals
                ]
                if attr_label_fmt
                else []
            )
            layer_handles = [
                Patch(
                    facecolor="white",
                    hatch=hatch_map.get(layer, ""),
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
                else (layer_handles + [x_handle])
            )
            if arrow_label:
                legend_handles.append(
                    Line2D([], [], color="#0b31bf", lw=1.5, label=arrow_label)
                )
            ax_left.legend(
                handles=legend_handles,
                title="Legenda",
                loc="lower left",
                fontsize="small",
                framealpha=0.85,
                edgecolor="black",
                fancybox=True,
            )

        plt.tight_layout()

        # Salva um arquivo por grupo (g1, g2, …)
        for ext in ("png", "pgf", "pdf"):
            output_path = Path("../../figures") / f"{out_name}_{dataset}_g{g_idx}.{ext}"
            fig.savefig(output_path, bbox_inches="tight")
        # plt.show()
        plt.close(fig)
        print(f"Saved: {output_path}")


def transcribe_figure_to_csv(
    df: pd.DataFrame,
    dataset: str,
    primary_col: str,
    primary_vals: Sequence[str | int],
    layers: Sequence[int],
    out_csv: str,
    alpha: float = 0.05,
) -> None:
    """
    Gera um CSV com uma linha para cada combinação
    (model_classifier, primary_val, n_layers) contendo:
      - estatísticas de boxplot (whislo, q1, med, q3, whishi)
      - outliers à esquerda e à direita
      - observações sobre testes de Wilcoxon+Holm (não-diferenças)
      - valor da diferença percentual (direito) e, no caso de transformer,
        qual o modelo de referência melhor.
    """
    records = []
    # prepara análises estatísticas globais por classifier
    for clf in sorted(df["model_classifier"].unique()):
        sub = df[df["model_classifier"] == clf]
        # Pivot para Wilcoxon
        pivot = sub.pivot_table(
            index="seed",
            columns=[primary_col, "n_layers"],
            values="f1",
        ).dropna(how="all", axis=1)

        tests = pairwise_wilcoxon_holm(pivot, alpha=alpha)
        tests = tests[["i", "j", "reject"]].to_dict(orient="records")
        tests = [[row["i"], row["j"], row["reject"]] for row in tests]

        # calcula percent_diff e best_ref se for caso transformer
        if primary_col == "model_transformer" and len(primary_vals) > 1:
            piv2 = sub.pivot_table(
                index=["model_classifier", "n_layers"],
                columns="model_transformer",
                values="f1",
                aggfunc="mean",
            )
            best_ref = piv2.idxmax(axis=1)
            pct_diff = (piv2[TARGET] - piv2.max(axis=1)) / piv2.max(axis=1) * 100
        elif primary_col == "n_features":
            piv2 = sub.pivot_table(
                index=["model_classifier", "n_layers"],
                columns="n_features",
                values="f1",
                aggfunc="mean",
            )
            pct_diff = (
                (piv2[primary_vals[0]] - piv2[primary_vals[-1]])
                / piv2[primary_vals[0]]
                * 100
            )
            best_ref = None
        else:
            pct_diff = pd.Series(dtype=float)
            best_ref = None

        # percorre grupos
        for p in primary_vals:
            for layer in layers:
                vals = (
                    sub.loc[(sub[primary_col] == p) & (sub["n_layers"] == layer), "f1"]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if vals.size == 0:
                    continue
                # boxplot_stats
                stats = boxplot_stats(vals, whis=1.5)[0]
                whislo = round(stats["whislo"], 4)
                q1 = round(stats["q1"], 4)
                med = round(stats["med"], 4)
                q3 = round(stats["q3"], 4)
                whishi = round(stats["whishi"], 4)
                fliers = [round(float(x), 4) for x in stats["fliers"]]
                left = [x for x in fliers if x < whislo]
                right = [x for x in fliers if x > whishi]

                # observações
                obs = []
                for i_col, j_col, reject in tests:
                    # mesmo primary, testes de camada
                    if (
                        not reject
                        and i_col[0] == p == j_col[0]
                        and i_col[1] != j_col[1]
                    ):
                        obs.append("sem diferença entre layers (n_layers = 1 ou = 10)")
                    # mesmo layer, primaries
                    if (
                        not reject
                        and i_col[1] == layer == j_col[1]
                        and i_col[0] != j_col[0]
                    ):
                        obs.append(
                            f"sem diferença entre {i_col[0]} vs {j_col[0]} com n_layer = {layer}"
                        )
                obs_txt = "; ".join(sorted(set(obs)))

                # valor de diferença
                diff = pct_diff.get((clf, layer), np.nan)
                diff = round(float(diff), 4) if not np.isnan(diff) else ""
                best = best_ref.get((clf, layer), "") if best_ref is not None else ""

                records.append(
                    {
                        "model_classifier": clf,
                        primary_col: p,
                        "n_layers": layer,
                        "whislo": whislo,
                        "q1": q1,
                        "median": med,
                        "q3": q3,
                        "whishi": whishi,
                        "outliers_left": json.dumps(left, ensure_ascii=False),
                        "outliers_right": json.dumps(right, ensure_ascii=False),
                        "observacoes": obs_txt,
                        "pct_diff": diff,
                        **({"best_model": best} if best_ref is not None else {}),
                    }
                )

    df_out = pd.DataFrame.from_records(records)
    # salva CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"[transcribe] salvo em {out_csv}")


if __name__ == "__main__":
    DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard")
    layers = sorted(read_summary()["n_layers"].unique())  # e.g., [1, 10]
    hatch_map = (
        {layers[0]: "", layers[1]: "//"} if len(layers) >= 2 else {layers[0]: ""}
    )
    BASE_MODELS = sorted(set(classical_models + quantum_models))
    ALLOWED_BASE = sorted(set(ALLOWED_CLASSIFIERS) & set(BASE_MODELS))

    # A) Varying number of features (n_features)
    TARGET = "tomaarsen/mpnet-base-nli-matryoshka"
    feature_values = [16, 32, 768]
    df_ft = read_summary()
    df_ft = df_ft[
        (df_ft["model_transformer"] == TARGET)
        & (df_ft["n_features"].isin(feature_values))
        # somente os circuitos especificados
        & (df_ft["model_classifier"].isin(ALLOWED_BASE))
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
            # passa os grupos de circuitos
            classifier_groups=CIRCUIT_GROUPS,
        )
        transcribe_figure_to_csv(
            df=df_ft[df_ft["dataset"] == ds],
            dataset=ds,
            primary_col="n_features",
            primary_vals=feature_values,
            layers=layers,
            out_csv=f"../../figures/n_features_{ds}.csv",
        )

    # B) Varying transformer models with fixed feature count
    df_tr = read_summary()
    df_tr = df_tr[
        (df_tr["n_features"] == 16)
        # somente os circuitos especificados
        & (df_tr["model_classifier"].isin(ALLOWED_BASE))
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
            classifier_groups=CIRCUIT_GROUPS,
        )
        transcribe_figure_to_csv(
            df=df_tr[df_tr["dataset"] == ds],
            dataset=ds,
            primary_col="model_transformer",
            primary_vals=transformers,
            layers=layers,
            out_csv=f"../../figures/transformers_{ds}.csv",
        )

    # C) Final analysis: 1 vs 10 layers for all models (apenas os circuitos listados)
    DATASETS_FINAL = (*DATASETS, "sst")
    MODELS_ALL = (
        classical_models
        + quantum_models
        + classical_ensemble_models
        + quantum_ensemble_models
    )
    # restringe aos circuitos permitidos
    MODELS_ALL = [m for m in MODELS_ALL if m in ALLOWED_CLASSIFIERS]

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
            classifier_groups=CIRCUIT_GROUPS,
        )
        transcribe_figure_to_csv(
            df=df_subset,
            dataset=ds,
            primary_col="model_transformer",
            primary_vals=[TARGET],
            layers=layers,
            out_csv=f"../../figures/f1_layers_{ds}.csv",
        )
