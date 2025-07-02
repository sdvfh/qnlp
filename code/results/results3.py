from aeon.visualisation import plot_critical_difference
from matplotlib import pyplot as plt
from utils import read_summary

# 1) Carrega o summary com todos os seeds, modelos e datasets
df = read_summary()

# 2) Define as bases de dados de interesse
DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard", "sst")

for ds in DATASETS:
    # 3) Filtra apenas as execuções daquele dataset
    df_ds = df[
        (df["dataset"] == ds)
        & (df["model_transformer"] == "tomaarsen/mpnet-base-nli-matryoshka")
        & (df["n_qubits"] == 4)
    ]

    # 4) Pivot para ter seeds nas linhas e (modelo, camadas) nas colunas
    pivot = df_ds.pivot_table(
        index="seed",
        columns=["model_classifier", "n_layers"],
        values="f1",
        dropna=False,  # manter todas as colunas mesmo que falte algum valor
    )

    # 5) Remove colunas que não têm nenhum valor
    pivot = pivot.dropna(axis=1, how="all")

    # 6) Extrai a matriz de scores e monta os labels
    scores = pivot.values  # shape = (n_seeds, n_estimators)
    labels = [f"{int(mc)}_{int(nl)}" for mc, nl in pivot.columns.to_list()]

    # 7) Gera o diagram de diferença crítica
    #    - lower_better=False pois F1 maior é melhor
    #    - alpha=0.05 (por ex.) para teste de Wilcoxon + Holm
    fig, ax = plot_critical_difference(
        scores,
        labels,
        lower_better=False,
        test="wilcoxon",
        correction="holm",
        alpha=0.05,
        reverse=True,  # melhor rank à esquerda
        width=6,
        textspace=1.5,
    )

    # 8) Salva a figura
    # fig.savefig(f"cd_{ds}.pdf", bbox_inches="tight")
    plt.show()
    print(f"Salvo: cd_{ds}.pdf")
    break
