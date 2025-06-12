import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from utils import classical_models, quantum_models, read_summary


def pairwise_wilcoxon_holm(df_pair, alpha=0.05):
    """
    Perform pairwise Wilcoxon signed‐rank tests with Holm correction.
    """
    models = df_pair.columns.tolist()
    records = []
    for i, j in itertools.combinations(models, 2):
        x = df_pair[i].values
        y = df_pair[j].values
        if len(x) == len(y) and (x != y).any():
            stat, p_uncorrected = wilcoxon(x, y)
        else:
            stat, p_uncorrected = np.nan, 1.0
        records.append((i, j, stat, p_uncorrected))
    pvals = [r[3] for r in records]
    reject, pvals_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    df_out = pd.DataFrame(
        records, columns=["model_i", "model_j", "statistic", "p_uncorrected"]
    )
    df_out["p_holm"] = pvals_holm
    df_out["reject"] = reject
    return df_out


def compute_positions(n_classifiers, transformers, layers, width, block_gap):
    """
    Compute horizontal offsets for each (transformer, layer) combination.
    """
    positions = []
    combos = [(tr, ly) for tr in transformers for ly in layers]
    for ti in range(len(transformers)):
        offset = ti * (len(layers) * width + block_gap) - block_gap
        for li in range(len(layers)):
            pos = np.arange(n_classifiers) - 0.3 + width / 2 + li * width + offset
            positions.append(pos)
    return positions, {combos[i]: positions[i] for i in range(len(combos))}


def plot_panel(
    ax, df_panel, classifier_order, transformers, layers, positions, pos_map, width
):
    """
    Draw a panel of boxplots with statistical annotations.
    """
    # draw grid and separator lines
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#CCCCCC", linestyle="--", linewidth=0.8)
    for y in np.arange(len(classifier_order) + 1) - 0.5:
        ax.axhline(y=y, color="black", linewidth=1, zorder=0)

    # draw boxplots
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

    alpha = 0.05

    # annotate non-differences between layers with 'x'
    for ci, clf in enumerate(classifier_order):
        pivot = df_panel[df_panel["model_classifier"] == clf].pivot(
            index="seed", columns=["model_transformer", "n_layers"], values="f1"
        )
        pivot.columns = pd.MultiIndex.from_tuples(pivot.columns)
        res = pairwise_wilcoxon_holm(pivot, alpha)
        for _, row in res[~res["reject"]].iterrows():
            (t0, l0), (t1, l1) = row["model_i"], row["model_j"]
            if t0 == t1 and l0 != l1:
                y0 = pos_map[(t0, l0)][ci]
                y1 = pos_map[(t1, l1)][ci]
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

    # annotate non-differences between transformers with green bidirectional arrows
    for ci, clf in enumerate(classifier_order):
        pivot = df_panel[df_panel["model_classifier"] == clf].pivot(
            index="seed", columns=["model_transformer", "n_layers"], values="f1"
        )
        pivot.columns = pd.MultiIndex.from_tuples(pivot.columns)
        for ly in layers:
            if ly not in pivot.columns.get_level_values(1):
                continue
            sub = pivot.xs(ly, level=1, axis=1)
            res = pairwise_wilcoxon_holm(sub, alpha)
            for _, row in res[~res["reject"]].iterrows():
                t0, t1 = row["model_i"], row["model_j"]
                m0 = pivot[(t0, ly)].median()
                m1 = pivot[(t1, ly)].median()
                y0 = pos_map[(t0, ly)][ci]
                y1 = pos_map[(t1, ly)][ci]
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

    # finalize axes
    ax.set_ylim(len(classifier_order) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_xlabel("F1 Score")


# common configuration
layers = sorted(read_summary()["n_layers"].unique())
transformers = read_summary()["model_transformer"].unique()
width = 0.6 / (len(transformers) * len(layers))
block_gap = 0.1

# global color maps
colors = plt.cm.tab10(np.linspace(0, 1, len(transformers)))
color_map = {tr: colors[i] for i, tr in enumerate(transformers)}
hatch_map = {layers[0]: "", layers[1]: "//"}

# iterate over datasets
for dataset_name in ["chatgpt_easy", "chatgpt_medium", "chatgpt_hard"]:
    # load and filter data
    df = read_summary()
    df = df[df["dataset"] == dataset_name]
    df = df[df["n_features"] == 16]
    df = df[df["model_classifier"].isin(quantum_models + classical_models)]

    # split into top and bottom panels
    df_top = df[df["model_classifier"].isin([3, 33])]
    df_bot = df[~df["model_classifier"].isin([3, 33])]

    top_order = (
        df_top.groupby("model_classifier")["f1"].mean().sort_values().index.tolist()
    )
    bot_order = (
        df_bot.groupby("model_classifier")["f1"].mean().sort_values().index.tolist()
    )

    # compute positions for each panel
    pos_top, pos_map_top = compute_positions(
        len(top_order), transformers, layers, width, block_gap
    )
    pos_bot, pos_map_bot = compute_positions(
        len(bot_order), transformers, layers, width, block_gap
    )

    # create figure (dpi=600 for high resolution)
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8, 20),
        dpi=600,
        sharex=False,
        gridspec_kw={"height_ratios": [len(top_order), len(bot_order)]},
    )

    plot_panel(
        ax_top, df_top, top_order, transformers, layers, pos_top, pos_map_top, width
    )
    plot_panel(
        ax_bot, df_bot, bot_order, transformers, layers, pos_bot, pos_map_bot, width
    )

    # set titles (remain in Portuguese)
    ax_top.set_title("Modelos 3 e 33")
    ax_bot.set_title("Demais Modelos")

    # legend entries
    transformer_handles = [
        Patch(facecolor=color_map[tr], label=tr) for tr in transformers
    ]
    layer_handles = [
        Patch(
            facecolor="white", hatch=hatch_map[ly], edgecolor="black", label=f"L = {ly}"
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
        label="Sem diferença entre layers",
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
        handles=transformer_handles + layer_handles + [x_handle, arrow_handle],
        title="Legenda",
        loc="best",
        fontsize="small",
    )

    plt.tight_layout()
    # save as PDF for Overleaf inclusion
    fig.savefig(f"../../figures/transformers_{dataset_name}.pdf", bbox_inches="tight")
    plt.close(fig)
