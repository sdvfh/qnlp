"""
Gera, para cada dataset, uma “matriz de cliques” colorida pelo rank geral dos
modelos dentro de cada clique identificado pelo teste de Wilcoxon + Holm,
ignorando a diagonal principal.
"""

import matplotlib.pyplot as plt
import numpy as np
from aeon.visualisation import plot_critical_difference
from aeon.visualisation.results._critical_difference import _build_cliques
from scipy.stats import rankdata
from utils import read_summary

# ─────────── Configuração ───────────
DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard", "sst")
ALPHA = 0.05
WIDTH = 6
TEXTSPACE = 1.5
REVERSE = True

# ─────────── Carrega tudo ───────────
df = read_summary()

for ds in DATASETS:
    # 1) filtra dataset + transformer + n_qubits conforme seu exemplo
    df_ds = df[
        (df["dataset"] == ds)
        & (df["model_transformer"] == "tomaarsen/mpnet-base-nli-matryoshka")
        & (df["n_qubits"] == 4)
    ]

    # 2) pivot para (seeds × estimadores)
    pivot = df_ds.pivot_table(
        index="seed",
        columns=["model_classifier", "n_layers"],
        values="f1",
        dropna=False,
    ).dropna(axis=1, how="all")

    scores = pivot.values  # (n_seeds, n_estimators)
    labels = [
        f"{int(mc)}_{int(nl)}"  # rótulos “ID_CAMADAS”
        for mc, nl in pivot.columns.to_list()
    ]

    # 3) pega p‐values
    fig, ax, pvalues = plot_critical_difference(
        scores,
        labels,
        lower_better=False,
        test="wilcoxon",
        correction="holm",
        alpha=ALPHA,
        width=WIDTH,
        textspace=TEXTSPACE,
        reverse=REVERSE,
        return_p_values=True,
    )
    plt.close(fig)

    # 4) calcula ranks e média de ranks
    ranks = rankdata(-scores, axis=1)
    avg_ranks = ranks.mean(axis=0)

    # 5) ordena
    order = np.argsort(avg_ranks)
    ordered_labels = [labels[i] for i in order]
    ordered_avg_ranks = avg_ranks[order]
    pmat = pvalues[np.ix_(order, order)]
    m = len(ordered_labels)

    # 6) cria matriz binária de não‐diferença
    threshold = ALPHA / (m - 1)
    pairwise = pmat > threshold

    # 7) obtém cliques
    cliques = _build_cliques(pairwise)

    # 8) monta matriz M sem preencher a diagonal
    M = np.full((m, m), np.nan)
    # para cada clique, preenche sub‐bloco com o melhor rank
    for clique in cliques:
        members = [i for i, in_clique in enumerate(clique) if in_clique]
        best = ordered_avg_ranks[members].min()
        for i in members:
            for j in members:
                if i != j:
                    M[i, j] = best

    # 9) máscara para ignorar diagonal
    M_masked = np.ma.masked_invalid(M)

    # 10) plota matriz colorida
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.viridis_r
    im = ax2.imshow(M_masked, cmap=cmap, origin="lower")
    ax2.set_xticks(np.arange(m))
    ax2.set_xticklabels(ordered_labels, rotation=90)
    ax2.set_yticks(np.arange(m))
    ax2.set_yticklabels(ordered_labels)
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Rank (menor = melhor)")
    ax2.set_title(f"Matriz de Cliques – {ds}")
    plt.tight_layout()

    # 11) exibe e salva
    plt.show()
    # fig2.savefig(f"clique_matrix_{ds}.pdf", bbox_inches="tight")
    print(f"Salvo: clique_matrix_{ds}.pdf")
