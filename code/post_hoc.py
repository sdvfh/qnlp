import pickle
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# Suponha que temos um DataFrame df_accuracy com colunas = modelos e linhas = valores de acurácia em 30 execuções.
# (Faça o mesmo para df_precision, df_sensitivity, df_f1, df_auc se tiver cada métrica separada.)

models = {
    "matryoshka_5_32": "/home/sergio/repositories/mestrado/qnlp-2024/results/matryoshka/5/32",
    "mpnet_5_32": "/home/sergio/repositories/mestrado/qnlp-2024/results/mpnet/5/32",
    "mpnet_10_32": "/home/sergio/repositories/mestrado/qnlp-2024/results/mpnet/10/32",
    "mpnet_10_768": "/home/sergio/repositories/mestrado/qnlp-2024/results/mpnet/10/768",
    "nomic_5_32": "/home/sergio/repositories/mestrado/qnlp-2024/results/nomic/5/32",
}
level = "easy"
ansatz = "AnsatzRotCNOT"
n_layers = 1

df = []
for model_name, model_path in models.items():
    file_results_path = list(Path(model_path).rglob("*.pkl"))
    if not file_results_path:
        raise ValueError("No results found")

    results = []

    for file in file_results_path:
        with open(file, "rb") as f:
            metrics = pickle.load(f)
            metrics["ansatz"] = file.stem.split("_")[0]
            # results.append(metrics)
            if (
                metrics["level"] == level
                and metrics["ansatz"] == ansatz
                and metrics["n_layers"] == n_layers
            ):
                results.append([metrics["seed"], metrics["accuracy"]])
    column = pd.DataFrame.from_records(
        results, columns=["seed", model_name + "_" + ansatz]
    )
    column = column.set_index("seed")
    df.append(column)

df = pd.concat(df, axis=1)

df_accuracy = df

# 1. Teste de Friedman para Acurácia
stat, p = friedmanchisquare(*(df_accuracy[col] for col in df_accuracy.columns))
print(f"Friedman Acurácia: estatística={stat:.3f}, p-valor={p:.4f}")
if p < 0.05:
    print(
        "Diferença estatisticamente significativa encontrada entre os modelos (Acurácia)."
    )

    # 2. Testes de Wilcoxon pareados entre cada par de modelos (post-hoc)
    p_vals = []
    comparisons = []
    modelos = df_accuracy.columns
    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            stat_w, p_w = wilcoxon(df_accuracy[modelos[i]], df_accuracy[modelos[j]])
            p_vals.append(p_w)
            comparisons.append(f"{modelos[i]} vs {modelos[j]}")

    # 3. Correção de múltiplas comparações (Holm-Bonferroni neste caso)
    reject, p_vals_corr, _, _ = multipletests(p_vals, alpha=0.05, method="holm")

    # Exibir resultados post-hoc:
    print("Comparações pareadas (Acurácia) com p-valor ajustado (Holm):")
    for comp, p_raw, p_adj, sig in zip(
        comparisons, p_vals, p_vals_corr, reject, strict=True
    ):
        status = "Significativo" if sig else "Não significativo"
        print(f" - {comp}: p-valor bruto={p_raw:.4f}, ajustado={p_adj:.4f} -> {status}")

    # 4. Ranking médio dos modelos para Acurácia
    # Calcula o rank em cada repetição (1 = melhor desempenho na repetição)
    ranks = df_accuracy.rank(
        axis=1, method="average", ascending=False
    )  # ascending=False pois valor maior = melhor
    avg_ranks = ranks.mean()  # média dos ranks por coluna
    print("Rank médio dos modelos (Acurácia):")
    for modelo, rank in avg_ranks.sort_values().items():
        print(f" - {modelo}: rank médio = {rank:.2f}")
else:
    print("Nenhuma diferença significativa entre os modelos (Acurácia).")
