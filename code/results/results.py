import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from utils import classical_models, quantum_models, read_summary


def pairwise_wilcoxon_holm(df_pair, alpha=0.05):
    models = df_pair.columns.tolist()
    results = []
    for i, j in itertools.combinations(models, 2):
        x = df_pair[i].values
        y = df_pair[j].values
        if len(x) == len(y) and (x != y).any():
            stat, p_unc = wilcoxon(x, y)
        else:
            stat, p_unc = np.nan, 1.0
        results.append((i, j, stat, p_unc))
    pvals = [r[3] for r in results]
    reject, pvals_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    out = pd.DataFrame(
        results, columns=["model_i", "model_j", "statistic", "p_uncorrected"]
    )
    out["p_holm"] = pvals_holm
    out["reject"] = reject
    return out


# --- 1) Lê e filtra seus dados ---
df = read_summary()
df = df[df["dataset"] == "chatgpt_easy"]
df = df[df["n_features"] == 16]
df = df[df["model_classifier"].isin(quantum_models + classical_models)]

# --- 2) Separa em top (modelos 3 e 33) e bottom (restantes) ---
special = [3, 33]
df_top = df[df["model_classifier"].isin(special)]
df_bot = df[~df["model_classifier"].isin(special)]

top_order = df_top.groupby("model_classifier")["f1"].mean().sort_values().index.tolist()
bot_order = df_bot.groupby("model_classifier")["f1"].mean().sort_values().index.tolist()

# parâmetros comuns
layers = sorted(df["n_layers"].unique())
transformers = df["model_transformer"].unique()
combos = [(tr, ly) for tr in transformers for ly in layers]
n_comb = len(combos)

# largura e gap entre blocos de transformers
width = 0.6 / n_comb
block_gap = 0.1


def compute_positions(n_cl):
    positions = []
    for ti in range(len(transformers)):
        base = ti * (len(layers) * width + block_gap) - block_gap
        for li in range(len(layers)):
            p = np.arange(n_cl) - 0.3 + width / 2 + li * width + base
            positions.append(p)
    return positions, {combos[i]: positions[i] for i in range(n_comb)}


positions_top, pos_map_top = compute_positions(len(top_order))
positions_bot, pos_map_bot = compute_positions(len(bot_order))

colors = plt.cm.tab10(np.linspace(0, 1, len(transformers)))
color_map = {tr: colors[i] for i, tr in enumerate(transformers)}
hatch_map = {layers[0]: "", layers[1]: "//"}


# --- 3) Função genérica que plota um painel num eixo `ax` ---
def plot_panel(ax, df_panel, classifier_order, positions, pos_map):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#CCCCCC", linestyle="--", linewidth=0.8)

    # linhas horizontais de limite de cada classifier
    for y in np.arange(len(classifier_order) + 1) - 0.5:
        ax.axhline(y=y, color="black", linewidth=1, zorder=0)

    # desenha boxplots
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

    # ① “x”: não-diferença entre layers dentro de cada transformer
    for ci, clf in enumerate(classifier_order):
        sub = df_panel[df_panel["model_classifier"] == clf].pivot(
            index="seed", columns=["model_transformer", "n_layers"], values="f1"
        )
        sub.columns = pd.MultiIndex.from_tuples(sub.columns)
        res = pairwise_wilcoxon_holm(sub, alpha)
        for _, row in res[res["reject"] == False].iterrows():
            (t0, l0), (t1, l1) = row["model_i"], row["model_j"]
            if t0 == t1 and l0 != l1:
                y0 = pos_map[(t0, l0)][ci]
                y1 = pos_map[(t1, l1)][ci]
                x_mid = (sub[(t0, l0)].median() + sub[(t1, l1)].median()) / 2
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

    # ② seta bidirecional verde: não-diferença entre transformers para cada layer
    for ci, clf in enumerate(classifier_order):
        sub = df_panel[df_panel["model_classifier"] == clf].pivot(
            index="seed", columns=["model_transformer", "n_layers"], values="f1"
        )
        sub.columns = pd.MultiIndex.from_tuples(sub.columns)

        for ly in layers:
            if ly not in sub.columns.get_level_values(1):
                continue

            sub_ly = sub.xs(ly, level=1, axis=1)
            res = pairwise_wilcoxon_holm(sub_ly, alpha)
            for _, row in res[res["reject"] == False].iterrows():
                t0, t1 = row["model_i"], row["model_j"]
                # medianas
                m0 = sub[(t0, ly)].median()
                m1 = sub[(t1, ly)].median()
                # posições verticais
                y0 = pos_map[(t0, ly)][ci]
                y1 = pos_map[(t1, ly)][ci]
                # desenha seta bidirecional verde entre as medianas
                ax.annotate(
                    "",
                    xy=(m1, y1),
                    xytext=(m0, y0),
                    arrowprops=dict(arrowstyle="<->", color="green", linewidth=1.5),
                    zorder=5,
                )

    ax.set_ylim(len(classifier_order) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(classifier_order)))
    ax.set_yticklabels(classifier_order)
    ax.set_xlabel("F1 Score")


# --- 4) Monta a figura com 2 subplots ---
fig, (ax_top, ax_bot) = plt.subplots(
    2,
    1,
    figsize=(8, 24),
    dpi=600,  # aumento de dpi
    sharex=False,
    gridspec_kw={"height_ratios": [len(top_order), len(bot_order)]},
)

plot_panel(ax_top, df_top, top_order, positions_top, pos_map_top)
plot_panel(ax_bot, df_bot, bot_order, positions_bot, pos_map_bot)

ax_top.set_title("Modelos 3 e 33")
ax_bot.set_title("Demais Modelos")

# legenda unificada
t_handles = [Patch(facecolor=color_map[tr], label=tr) for tr in transformers]
l_handles = [
    Patch(facecolor="white", hatch=hatch_map[ly], edgecolor="black", label=f"L = {ly}")
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
    marker=r"$\leftrightarrow$",
    linestyle="None",
    markersize=10,
    label="Sem diferença entre transformers",
)

ax_bot.legend(
    handles=t_handles + l_handles + [x_handle, arrow_handle],
    title="Legenda",
    loc="best",
    fontsize="small",
)

plt.tight_layout()
plt.show()
