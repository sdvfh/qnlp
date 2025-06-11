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


# --- Lê e filtra seus dados ---
df = read_summary()
df = df[df["dataset"] == "chatgpt_easy"]
df = df[df["n_features"] == 16]
df = df[df["model_classifier"].isin(quantum_models + classical_models)]


# --- Configura plotagem ---
classifier_order = (
    df.groupby("model_classifier")["f1"]
    .mean()
    .sort_values(ascending=True)
    .index.tolist()
)
layers = sorted(df["n_layers"].unique())  # ex: [1,10]
transformers = df["model_transformer"].unique()
combos = [(tr, ly) for tr in transformers for ly in layers]

n_cl = len(classifier_order)
n_comb = len(combos)

# largura de cada boxplot
width = 0.6 / n_comb

# gap extra entre blocos de transformers distintos (em unidades de F1 score)
block_gap = 0.1  # ajuste entre 0.02 e 0.1 conforme necessidade

# calcula posições incluindo gap entre transformers
positions = []
for ti, _ in enumerate(transformers):
    # deslocamento acumulado por bloco completo de layers + gap
    base_offset = ti * (len(layers) * width + block_gap) - block_gap
    for li, _ in enumerate(layers):
        pos = np.arange(n_cl) - 0.3 + width / 2 + li * width + base_offset
        positions.append(pos)

# mapeia cada combo (transformer, n_layers) à sua posição
pos_map = {combos[i]: positions[i] for i in range(n_comb)}

# cores e hatches
colors = plt.cm.tab10(np.linspace(0, 1, len(transformers)))
color_map = {tr: colors[i] for i, tr in enumerate(transformers)}
hatch_map = {1: "", 10: "//"}

# --- Desenha boxplots ---
fig, ax = plt.subplots(figsize=(8, 20), dpi=300)
ax.set_axisbelow(True)
ax.grid(True, which="major", color="#CCCCCC", linestyle="--", linewidth=0.8)
for y in np.arange(n_cl + 1) - 0.5:
    ax.axhline(y=y, color="black", linewidth=1, zorder=0)

for i, (tr, ly) in enumerate(combos):
    data_plot = [
        df[
            (df["model_classifier"] == clf)
            & (df["model_transformer"] == tr)
            & (df["n_layers"] == ly)
        ]["f1"].values
        for clf in classifier_order
    ]
    bp = ax.boxplot(
        data_plot,
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

# --- Adiciona asteriscos de não-diferença ---
alpha = 0.05
for ci, clf in enumerate(classifier_order):
    sub = df[df["model_classifier"] == clf].pivot(
        index="seed", columns=["model_transformer", "n_layers"], values="f1"
    )
    sub.columns = pd.MultiIndex.from_tuples(sub.columns)
    res = pairwise_wilcoxon_holm(sub, alpha)
    for _, row in res[res["reject"] == False].iterrows():
        col_i = row["model_i"]
        col_j = row["model_j"]
        if col_i[0] == col_j[0] and col_i[1] != col_j[1]:
            y1 = pos_map[col_i][ci]
            y2 = pos_map[col_j][ci]
            y_ast = np.mean([y1, y2])
            med1 = sub[col_i].median()
            med2 = sub[col_j].median()
            x_ast = (med1 + med2) / 2
            ax.text(
                x_ast,
                y_ast,
                "x",
                ha="center",
                va="center",
                color="#ed1fed",
                fontsize=12,
                fontweight="bold",
                zorder=5,
            )

# --- Finaliza e legenda ---
ax.set_ylim(n_cl - 0.5, -0.5)
ax.set_yticks(np.arange(n_cl))
ax.set_yticklabels(classifier_order)
ax.set_xlabel("F1 Score")
ax.set_title(
    "Boxplot F1 por Classifier, Transformer e N_Layers\n"
    "'x' indica sem diferença estatística (Wilcoxon+Holm)"
)

t_handles = [Patch(facecolor=color_map[tr], label=tr) for tr in transformers]
l_handles = [
    Patch(
        facecolor="white",
        hatch=hatch_map[ly],
        edgecolor="black",
        label=f"n_layers={ly}",
    )
    for ly in layers
]
s_handle = Line2D(
    [0],
    [0],
    marker="x",
    color="#ed1fed",
    linestyle="",
    markerfacecolor="none",
    markersize=12,
    label="semelhantes",
)
ax.legend(handles=t_handles + l_handles + [s_handle], title="Legenda", loc="best")

plt.tight_layout()
plt.show()
