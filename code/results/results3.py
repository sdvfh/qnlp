import matplotlib.pyplot as plt
import numpy as np
from aeon.visualisation import plot_critical_difference
from aeon.visualisation.results._critical_difference import _build_cliques
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from utils import classical_ensemble_models, classical_models, read_summary

# ─────────── Configuração ───────────
DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard", "sst")
ALPHA = 0.05
WIDTH = 10  # largura maior para desafogar a régua
TEXTSPACE = 1.9
REVERSE = True

df_all = read_summary()


def format_label(mc: int, nl: int) -> str:
    """Remove “_1” dos clássicos, mantém para quânticos."""
    if mc in classical_models + classical_ensemble_models:
        return f"{mc}"
    else:
        return f"{mc}_{nl}"


for ds in DATASETS:
    # 1) filtra
    df = df_all[
        (df_all["dataset"] == ds)
        & (df_all["model_transformer"] == "tomaarsen/mpnet-base-nli-matryoshka")
        & (df_all["n_qubits"] == 4)
    ]

    # 2) pivot
    pivot = df.pivot_table(
        index="seed",
        columns=["model_classifier", "n_layers"],
        values="f1",
        dropna=False,
    ).dropna(axis=1, how="all")

    # 3) scores e labels
    scores = pivot.values
    labels = [format_label(mc, nl) for mc, nl in pivot.columns.to_list()]

    # 4) pega p‐values crus
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

    # 5) ranks médios
    ranks = rankdata(-scores, axis=1)
    avg_ranks = ranks.mean(axis=0)

    # 6) ordenação
    order = np.argsort(avg_ranks)
    ordered_labels = [labels[i] for i in order]
    ordered_avg_ranks = avg_ranks[order]
    raw_pmat = pvalues[np.ix_(order, order)]
    m = len(ordered_labels)

    # 7) correção Holm via multipletests
    # extrai apenas a parte acima da diagonal (k=1)
    iu = np.triu_indices(m, k=1)
    raw = raw_pmat[iu]
    # aplica Holm
    _, p_holm, _, _ = multipletests(raw, alpha=ALPHA, method="holm")
    # remonta a matriz simétrica de pval corrigidos
    pmat_holm = np.ones_like(raw_pmat)
    pmat_holm[iu] = p_holm
    pmat_holm[(iu[1], iu[0])] = p_holm

    # 8) monta a matriz booleana de “não diferença”
    pairwise = pmat_holm > ALPHA

    # 9) forma cliques
    cliques = _build_cliques(pairwise)

    # 10) filtra cliques que incluam algum clássico
    def is_classical_idx(idx: int) -> bool:
        mc = int(ordered_labels[idx].split("_")[0])
        return mc in classical_models + classical_ensemble_models

    filtered = [
        clique
        for clique in cliques
        if any(is_classical_idx(i) for i, flag in enumerate(clique) if flag)
    ]

    # 11) índices a manter
    keep = sorted({i for clique in filtered for i, flag in enumerate(clique) if flag})

    # 12) filtra o pivot
    cols_ordered = [pivot.columns[i] for i in order]
    pivot_filtered = pivot.loc[:, cols_ordered].iloc[:, keep]

    scores_filt = pivot_filtered.values
    labels_filt = [format_label(mc, nl) for mc, nl in pivot_filtered.columns.to_list()]

    # 13) gera o CD‐plot final
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

    # 14) ajusta posição para não cortar nada
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height - 0.05])
    # aumenta dpi
    fig2.set_dpi(300)

    plt.show()
    print(f"Salvo: cd_filtered_{ds}.pdf")
