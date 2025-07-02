import matplotlib.pyplot as plt
import numpy as np
from aeon.visualisation import plot_critical_difference
from aeon.visualisation.results._critical_difference import _build_cliques
from scipy.stats import rankdata
from utils import classical_ensemble_models, classical_models, read_summary

# ─────────── Configuração ───────────
DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard", "sst")
ALPHA = 0.05
WIDTH = 10  # aumentamos a largura para dar mais espaço à régua de rank
TEXTSPACE = 1.9
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

    # 8) filtra apenas cliques que contenham modelo clássico
    def is_classical_idx(idx: int) -> bool:
        model_id = int(ordered_labels[idx].split("_")[0])
        return model_id in classical_models + classical_ensemble_models

    filtered_cliques = [
        clique
        for clique in cliques
        if any(is_classical_idx(i) for i, flag in enumerate(clique) if flag)
    ]

    # 9) índices dos estimadores a manter
    keep_idxs = sorted(
        {i for clique in filtered_cliques for i, flag in enumerate(clique) if flag}
    )

    # 10) filtra o pivot original
    ordered_cols = list(pivot.columns[order])
    pivot_filtered = pivot.loc[:, ordered_cols].iloc[:, keep_idxs]

    scores_filt = pivot_filtered.values
    labels_filt = [
        f"{int(mc)}_{int(nl)}" for mc, nl in pivot_filtered.columns.to_list()
    ]

    # 11) gera o CD plot final só com os estimadores filtrados
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
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height - 0.05])
    # 12) ajusta margens para não cortar nada
    # fig2.subplots_adjust(top=0.8)
    # fig2.tight_layout()

    plt.show()
    print(f"Salvo: cd_filtered_{ds}.pdf")
