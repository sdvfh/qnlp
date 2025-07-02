"""
Gera, para cada dataset, um diagram de diferença crítica filtrado para exibir
apenas os estimadores que participam de cliques contendo ao menos um modelo clássico.
"""

import matplotlib.pyplot as plt
import numpy as np
from aeon.visualisation import plot_critical_difference
from aeon.visualisation.results._critical_difference import _build_cliques
from scipy.stats import rankdata
from utils import classical_ensemble_models, classical_models, read_summary

# ─────────── Configuração ───────────
DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard", "sst")
ALPHA = 0.05
WIDTH = 6
TEXTSPACE = 1.5
REVERSE = True

# ─────────── Carrega todos os resultados ───────────
df_all = read_summary()

for ds in DATASETS:
    # 1) filtra pelo dataset, transformer e n_qubits
    df = df_all[
        (df_all["dataset"] == ds)
        & (df_all["model_transformer"] == "tomaarsen/mpnet-base-nli-matryoshka")
        & (df_all["n_qubits"] == 4)
    ]

    # 2) pivot para (seeds × estimadores)
    pivot = df.pivot_table(
        index="seed",
        columns=["model_classifier", "n_layers"],
        values="f1",
        dropna=False,
    ).dropna(axis=1, how="all")

    # 3) extrai scores e labels
    scores = pivot.values
    labels = [f"{int(mc)}_{int(nl)}" for mc, nl in pivot.columns.to_list()]

    # 4) obtenha p‐values via CD plot
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

    # 5) calcula ranks e médias de ranks
    ranks = rankdata(-scores, axis=1)  # maior F1 = menor rank
    avg_ranks = ranks.mean(axis=0)

    # 6) ordena estimadores por avg_rank
    order = np.argsort(avg_ranks)
    ordered_labels = [labels[i] for i in order]
    ordered_avg_ranks = avg_ranks[order]
    pmat = pvalues[np.ix_(order, order)]
    m = len(ordered_labels)

    # 7) constrói a matriz de “não‐diferença” e extrai cliques
    threshold = ALPHA / (m - 1)
    pairwise = pmat > threshold
    cliques = _build_cliques(pairwise)

    # 8) filtra apenas cliques que contenham pelo menos um modelo clássico
    def is_classical_idx(ordered_labels, idx: int) -> bool:
        model_id = int(ordered_labels[idx].split("_")[0])
        return model_id in classical_models + classical_ensemble_models

    filtered_cliques = [
        clique
        for clique in cliques
        if any(
            is_classical_idx(ordered_labels, i) for i, flag in enumerate(clique) if flag
        )
    ]

    # 9) reúne o conjunto de índices de todos os estimadores nesses cliques
    keep_idxs = sorted(
        {i for clique in filtered_cliques for i, flag in enumerate(clique) if flag}
    )

    # 10) reordena e filtra o pivot original
    ordered_cols = list(pivot.columns[order])
    pivot_ordered = pivot.loc[:, ordered_cols]
    pivot_filtered = pivot_ordered.iloc[:, keep_idxs]

    scores_filt = pivot_filtered.values
    labels_filt = [
        f"{int(mc)}_{int(nl)}" for mc, nl in pivot_filtered.columns.to_list()
    ]

    # 11) gera o diagram de diferença crítica final só com os estimadores filtrados
    fig2, ax2 = plot_critical_difference(
        scores_filt,
        labels_filt,
        lower_better=False,
        test="wilcoxon",
        correction="holm",
        alpha=ALPHA,
        width=WIDTH,
        textspace=TEXTSPACE,
        reverse=REVERSE,
    )

    # 12) salva e fecha
    # fig2.savefig(f"cd_filtered_{ds}.pdf", bbox_inches="tight")
    fig2.subplots_adjust(bottom=0.8)
    plt.show()
    plt.close(fig2)
    print(f"Salvo: cd_filtered_{ds}.pdf")
