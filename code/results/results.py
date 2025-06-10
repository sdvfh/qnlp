import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from utils import (
    classical_ensemble_models,
    classical_models,
    quantum_ensemble_models,
    quantum_models,
    read_summary,
)

df = read_summary()
df = df[df["dataset"] == "chatgpt_easy"]
df = df[df["n_features"] == 16]
df = df[df["model_classifier"].isin(quantum_models + classical_models)]

classifier_order = (
    df.groupby("model_classifier")["f1"]
    .mean()
    .sort_values(ascending=True)
    .index.tolist()
)

layers = sorted(df["n_layers"].unique())
transformers = df["model_transformer"].unique()
combos = [(tr, ly) for tr in transformers for ly in layers]

n_tr = len(transformers)
n_cl = len(classifier_order)
n_combos = len(combos)

# Posições
width = 0.6 / n_combos
positions = [np.arange(n_cl) - 0.3 + width / 2 + i * width for i in range(n_combos)]

# Cores e hatches
colors = plt.cm.tab10(np.linspace(0, 1, n_tr))
color_map = {tr: colors[i] for i, tr in enumerate(transformers)}
hatch_map = {1: "", 10: "//"}

# Plot
fig, ax = plt.subplots(figsize=(8, 20), dpi=300)
ax.set_axisbelow(True)
# Ativa grid maior
ax.grid(
    True, which="major", color="#CCCCCC", linestyle="--", linewidth=0.8  # cinza claro
)

# Linhas delimitadoras (sem espaço extra)
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
    )
    for box in bp["boxes"]:
        box.set_facecolor(color_map[tr])
        box.set_hatch(hatch_map[ly])

# Ajuste fino dos limites sem margens
ax.set_ylim(n_cl - 0.5, -0.5)
ax.margins(y=0)

# Eixos e legendas
ax.set_yticks(np.arange(n_cl))
ax.set_yticklabels(classifier_order)
ax.set_xlabel("F1 Score")
ax.set_title("Boxplot F1 por Classifier, Transformer e N_Layers")

transformer_handles = [Patch(facecolor=color_map[tr], label=tr) for tr in transformers]
layer_handles = [
    Patch(
        facecolor="white",
        hatch=hatch_map[ly],
        edgecolor="black",
        label=f"n_layers={ly}",
    )
    for ly in layers
]
ax.legend(
    handles=transformer_handles + layer_handles,
    title="Transformer / N_Layers",
    loc="best",
)

plt.tight_layout()
plt.show()
