from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cd import plot_critical_difference
from aeon.visualisation.results._critical_difference import _build_cliques
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from utils import classical_ensemble_models, classical_models, read_summary

import matplotlib as mpl

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": (
        # tudo numa única string, não numa lista
        r"\usepackage[T1]{fontenc} "
        r"\usepackage[utf8]{inputenc} "
        # tratar '_' como caractere normal
        r"\catcode`\_=12\relax "
        # definir macro \_ para imprimir underscore
        r"\def\_{\_}"
    ),
})

# ─────────── Configuração ───────────
DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard", "sst")
ALPHA = 0.05
WIDTH = 8  # largura maior para desafogar a régua
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

    # 4) pega p‐values brutos via CD plot
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
    plt.close("all")

    # 5) calcula ranks e médias de ranks
    ranks = rankdata(-scores, axis=1)
    avg_ranks = ranks.mean(axis=0)

    # 6) ordena estimadores
    order = np.argsort(avg_ranks)
    ordered_labels = [labels[i] for i in order]
    ordered_avg_ranks = avg_ranks[order]
    raw_pmat = pvalues[np.ix_(order, order)]
    m = len(ordered_labels)

    # 7) Holm correction
    iu = np.triu_indices(m, k=1)
    raw = raw_pmat[iu]
    _, p_holm, _, _ = multipletests(raw, alpha=ALPHA, method="holm")
    pmat_holm = np.ones_like(raw_pmat)
    pmat_holm[iu] = p_holm
    pmat_holm[(iu[1], iu[0])] = p_holm

    # 8) boolean de “não diferença”
    pairwise = pmat_holm > ALPHA

    # 9) forma cliques
    cliques = _build_cliques(pairwise)

    # 10) filtra cliques size>1 com pelo menos um clássico
    def is_classical_idx(idx: int) -> bool:
        mc = int(ordered_labels[idx].split("_")[0])
        return mc in classical_models + classical_ensemble_models

    filtered_cliques = [
        cl
        for cl in cliques
        if sum(cl) > 1 and any(is_classical_idx(i) for i, flag in enumerate(cl) if flag)
    ]

    # 11) índices iniciais a manter
    keep = sorted({i for cl in filtered_cliques for i, flag in enumerate(cl) if flag})

    # 11.1) adiciona complemento de camadas para quânticos
    ordered_cols = [pivot.columns[i] for i in order]
    layer_values = sorted({nl for _, nl in ordered_cols})
    mapping = {(mc, nl): idx for idx, (mc, nl) in enumerate(ordered_cols)}
    extended = set(keep)
    for idx in keep:
        mc, nl = ordered_cols[idx]
        if mc not in classical_models + classical_ensemble_models:
            for nl2 in layer_values:
                if nl2 != nl and (mc, nl2) in mapping:
                    extended.add(mapping[(mc, nl2)])
    keep = sorted(extended)

    # 12) filtra o pivot
    pivot_filtered = pivot.loc[:, ordered_cols].iloc[:, keep]
    scores_filt = pivot_filtered.values
    labels_filt = [format_label(mc, nl) for mc, nl in pivot_filtered.columns.to_list()]

    # ── Remoção específica para chatgpt_hard ────────────────────────────
    if ds == "chatgpt_hard":
        to_drop = {"26_10", "26_1"}
        mask = [lbl not in to_drop for lbl in labels_filt]
        scores_filt = scores_filt[:, mask]
        labels_filt = [lbl for lbl in labels_filt if lbl not in to_drop]

    labels_filt = [lbl.replace('_', r'\_') for lbl in labels_filt]

    # 13) gera o CD‐plot final
    fig2, ax2 = plot_critical_difference(
        scores_filt,
        labels_filt,
        lower_better=False,
        test="wilcoxon",
        correction=None,
        alpha=ALPHA,
        width=WIDTH,
        textspace=TEXTSPACE,
        reverse=REVERSE,
    )

    # 14) ajusta posição e DPI
    pos = ax2.get_position()
    ax2.set_position([pos.x0 - 0.08, pos.y0 + 0.05, pos.width + 0.13, pos.height - 0.05])
    # ax2.set_position([0.00, 0.00, 1.00, 0.95])
    fig2.set_size_inches(7, 7, forward=True)
    fig2.set_dpi(300)

    out_dir = Path("../../figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    pgf_path = out_dir / f"cd_{ds}.pgf"
    fig2.tight_layout()
    fig2.savefig(pgf_path, bbox_inches="tight",  pad_inches=0,)
    # plt.show()
    print(f"Salvo: cd_{ds}.pdf")

