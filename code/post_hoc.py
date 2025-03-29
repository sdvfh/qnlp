import pickle
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from cdd import draw_cd_diagram

# Suponha que temos um DataFrame df_accuracy com colunas = modelos e linhas = valores de acurácia em 30 execuções.
# (Faça o mesmo para df_precision, df_sensitivity, df_f1, df_auc se tiver cada métrica separada.)

models = {
    "matryoshka_5_32": "/home/sergio/repositories/mestrado/qnlp-2024/results/matryoshka/5/32",
    "mpnet_5_32": "/home/sergio/repositories/mestrado/qnlp-2024/results/mpnet/5/32",
    # "mpnet_10_32": "/home/sergio/repositories/mestrado/qnlp-2024/results/mpnet/10/32",
    # "mpnet_10_768": "/home/sergio/repositories/mestrado/qnlp-2024/results/mpnet/10/768",
    "nomic_5_32": "/home/sergio/repositories/mestrado/qnlp-2024/results/nomic/5/32",
}
level = "easy"

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
            if metrics["level"] == level:
                results.append([metrics["seed"], metrics["f1"], metrics["ansatz"], metrics["n_layers"]])
    column = pd.DataFrame.from_records(
        results, columns=["seed", model_name, "ansatz", "n_layers"]
    )
    column = column.pivot_table(
        index='seed',
        columns=['ansatz', 'n_layers'],
        values=model_name,
        aggfunc='first'  # Usar 'first' para evitar duplicatas
    )
    column.columns = [f"{model_name}_{ansatz}_{layers}" for ansatz, layers in column.columns]
    df.append(column)

df = pd.concat(df, axis=1)

df_accuracy = df
df_accuracy_2 = df.copy()
df_accuracy_2 = df_accuracy_2.reset_index()
df_accuracy_2 = df_accuracy_2.melt(id_vars=["seed"])
df_accuracy_2.columns = ["dataset_name", "classifier_name", "accuracy"]

draw_cd_diagram(df_perf=df_accuracy_2, title='F1', labels=True)
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
        model_i_parts = modelos[i].split("_")
        for j in range(i + 1, len(modelos)):
            model_j_parts = modelos[j].split("_")
            stat_w, p_w = wilcoxon(df_accuracy[modelos[i]], df_accuracy[modelos[j]])
            p_vals.append(p_w)
            comparisons.append([
                f"{modelos[i]} vs {modelos[j]}",
                *model_i_parts[1:],
                *model_j_parts[1:],
            ])

    # 3. Correção de múltiplas comparações (Holm-Bonferroni neste caso)
    reject, p_vals_corr, _, _ = multipletests(p_vals, alpha=0.05, method="holm")

    # Exibir resultados post-hoc:
    print("Comparações pareadas (Acurácia) com p-valor ajustado (Holm):")
    for comp_list, p_raw, p_adj, sig in zip(
        comparisons, p_vals, p_vals_corr, reject, strict=True
    ):
        comp = comp_list[0]
        if (comp_list[2] == comp_list[6]) and (comp_list[3] == comp_list[7]) and (comp_list[4] == comp_list[8]):
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
