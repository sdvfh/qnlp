from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

# Mapas solicitados
BASE_MAP = {
    "easy": "Fácil",
    "medium": "Médio",
    "hard": "Difícil",
}

# Agora com 8 grupos, conforme solicitado
CIRCUIT_GROUPS: dict[int, Tuple[str, str]] = {
    1: (
        "1 ao 6",
        "demonstram a crescente complexidade ao utilizar portas quânticas e portas CNOT",
    ),
    2: ("7 ao 11", "implementam diferentes formas de emaranhamento quântico"),
    3: (
        "12 ao 15",
        "implementam diferentes tipos de conectividade entre \\textit{qubits}",
    ),
    4: ("32, 33 e 36", "implementam modelos clássicos lineares"),
    5: ("34, 35, 37 e 38", "implementam modelos clássicos não-lineares"),
    6: (
        "16 ao 21",
        "implementam modelos de ensemble através de \\textit{AdaBoost} ou \\textit{Bagging} de circuitos quânticos",
    ),
    7: (
        "22 ao 31",
        "implementam modelos de ensemble através de \\textit{hard} ou \\textit{soft voting} de circuitos quânticos",
    ),
    8: ("39 ao 44", "implementam modelos de ensemble clássicos"),
}

# Templates LaTeX com chaves escapadas para .format()
TEX_N_FEATURES = r"""\begin{{figure}}
\caption{{Resultados do desempenho F1 dos circuitos {LISTA_CIRCUITOS} com diferentes tamanhos de vetor de entrada na base \textit{{ChatGPT}} - {TIPO_BASE_TRADUZIDA}. Esses modelos {DESCRICAO_CIRCUITOS}. As barras comparativas do lado direito exibem a diferença percentual da métrica F1 ao se reduzir de 768 atributos para 16 atributos, divididos pela profundidade de cada circuito. As marcações que indicam semelhança estatística são advindas do teste pareado de Wilcoxon com correção de p-valor de Holm.}}
\label{{fig:n_features:{TIPO_BASE}:g{NUMERO}}}
     \centering
    \centering
    \resizebox{{!}}{{0.90\textheight}}{{%  <-- Ajuste a altura aqui
        \import{{capitulos/4_resultados/figuras}}{{n_features_chatgpt_{TIPO_BASE}_g{NUMERO}.pgf}}
    }}
      \par\medskip\ABNTEXfontereduzida\selectfont\textbf{{Fonte:}} Elaborada pelo autor (2025) \par\medskip
\end{{figure}}"""

TEX_TRANSFORMERS = r"""\begin{{figure}}
\caption{{Resultados do desempenho F1 dos circuitos {LISTA_CIRCUITOS} com os modelos geradores de \textit{{embeddings}} na base \textit{{ChatGPT}} - {TIPO_BASE_TRADUZIDA}. Esses modelos {DESCRICAO_CIRCUITOS}. As barras comparativas do lado direito exibem a diferença percentual da métrica F1 entre o modelo com a maior média F1 e o modelo \textit{{Matryoshka}}, divididos pela profundidade de cada circuito. As barras de diferença são coloridas com a mesma cor do melhor modelo em questão. As marcações que indicam semelhança estatística são advindas do teste pareado de Wilcoxon com correção de p-valor de Holm.}}
\label{{fig:model_transformer:{TIPO_BASE}:g{NUMERO}}}
     \centering
    \centering
    \resizebox{{!}}{{0.90\textheight}}{{%  <-- Ajuste a altura aqui
        \import{{capitulos/4_resultados/figuras}}{{transformers_chatgpt_{TIPO_BASE}_g{NUMERO}.pgf}}
    }}

      \par\medskip\ABNTEXfontereduzida\selectfont\textbf{{Fonte:}} Elaborada pelo autor (2025) \par\medskip
\end{{figure}}"""

# --- NOVOS templates F1 ---
# Variante ChatGPT (mantém "ChatGPT - {TIPO_BASE_TRADUZIDA}" e importa f1_layers_chatgpt_{TIPO_BASE}_g{NUMERO}.pgf)
TEX_F1_CHATGPT = r"""\begin{{figure}}
\caption{{Resultados do desempenho F1 dos circuitos {LISTA_CIRCUITOS} na base \textit{{ChatGPT}} - {TIPO_BASE_TRADUZIDA}. Esses algoritmos classificadores {DESCRICAO_CIRCUITOS}. As marcações que indicam semelhança estatística são advindas do teste pareado de Wilcoxon.}}
\label{{fig:f1:{TIPO_BASE}:g{NUMERO}}}
     \centering
    \centering
    \resizebox{{!}}{{0.95\textheight}}{{%  <-- Ajuste a altura aqui
        \import{{capitulos/4_resultados/figuras}}{{f1_layers_chatgpt_{TIPO_BASE}_g{NUMERO}.pgf}}
    }}

      \par\medskip\ABNTEXfontereduzida\selectfont\textbf{{Fonte:}} Elaborada pelo autor (2025) \par\medskip
\end{{figure}}"""

# Variante SST (muda legenda para "\GLS{SST}" e troca chatgpt_{TIPO_BASE} -> sst no import e no label)
TEX_F1_SST = r"""\begin{{figure}}
\caption{{Resultados do desempenho F1 dos circuitos {LISTA_CIRCUITOS} na base \gls{{SST}}. Esses algoritmos classificadores {DESCRICAO_CIRCUITOS}. As marcações que indicam semelhança estatística são advindas do teste pareado de Wilcoxon.}}
\label{{fig:f1:sst:g{NUMERO}}}
     \centering
    \centering
    \resizebox{{!}}{{0.95\textheight}}{{%  <-- Ajuste a altura aqui
        \import{{capitulos/4_resultados/figuras}}{{f1_layers_sst_g{NUMERO}.pgf}}
    }}

      \par\medskip\ABNTEXfontereduzida\selectfont\textbf{{Fonte:}} Elaborada pelo autor (2025) \par\medskip
\end{{figure}}"""


def _common_for(tipo_base: str, numero: int) -> tuple[str, str, str, str, int]:
    tb = tipo_base.lower()
    if tb not in BASE_MAP:
        raise ValueError(
            f"TIPO_BASE inválido: {tipo_base!r}. Use: easy, medium ou hard."
        )
    if numero not in CIRCUIT_GROUPS:
        raise ValueError(f"NUMERO inválido: {numero!r}. Use um inteiro de 1 a 8.")
    lista, descricao = CIRCUIT_GROUPS[numero]
    return lista, descricao, BASE_MAP[tb], tb, numero


def render_tex(
    fig_kind: str, tipo_base: str | None, numero: int, base_scope: str = "both"
) -> list[tuple[str, str]]:
    """
    fig_kind: 'n_features' | 'transformers' | 'f1'
    base_scope (para 'f1'): 'chatgpt' | 'sst' | 'both'
    Retorna lista de pares (nome_arquivo, conteudo).
    """
    out: list[tuple[str, str]] = []

    if fig_kind in ("n_features", "transformers"):
        if tipo_base is None:
            raise ValueError("Este tipo de figura requer --tipo (easy|medium|hard).")
        lista, descricao, tipo_base_traduzida, tb, num = _common_for(tipo_base, numero)
        template = TEX_N_FEATURES if fig_kind == "n_features" else TEX_TRANSFORMERS
        content = template.format(
            LISTA_CIRCUITOS=lista,
            DESCRICAO_CIRCUITOS=descricao,
            TIPO_BASE_TRADUZIDA=tipo_base_traduzida,
            TIPO_BASE=tb,
            NUMERO=num,
        )
        prefix = (
            "n_features_chatgpt" if fig_kind == "n_features" else "transformers_chatgpt"
        )
        filename = f"{prefix}_{tb}_g{num}.tex"
        out.append((filename, content))
        return out

    if fig_kind == "f1":
        # ChatGPT (depende de --tipo) e/ou SST (independe)
        if base_scope in ("chatgpt", "both"):
            if tipo_base is None:
                raise ValueError(
                    "Para f1 com base ChatGPT, informe --tipo (easy|medium|hard)."
                )
            lista, descricao, tipo_base_traduzida, tb, num = _common_for(
                tipo_base, numero
            )
            content_cgpt = TEX_F1_CHATGPT.format(
                LISTA_CIRCUITOS=lista,
                DESCRICAO_CIRCUITOS=descricao,
                TIPO_BASE_TRADUZIDA=tipo_base_traduzida,
                TIPO_BASE=tb,
                NUMERO=num,
            )
            out.append((f"f1_layers_chatgpt_{tb}_g{num}.tex", content_cgpt))

        if base_scope in ("sst", "both"):
            # Para SST, não depende de tipo_base: usa \GLS{SST} e f1_layers_sst_g{NUMERO}.pgf
            if numero not in CIRCUIT_GROUPS:
                raise ValueError(
                    f"NUMERO inválido: {numero!r}. Use um inteiro de 1 a 8."
                )
            lista, descricao = CIRCUIT_GROUPS[numero]
            content_sst = TEX_F1_SST.format(
                LISTA_CIRCUITOS=lista,
                DESCRICAO_CIRCUITOS=descricao,
                NUMERO=numero,
            )
            out.append((f"f1_layers_sst_g{numero}.tex", content_sst))
        return out

    raise ValueError("fig_kind inválido. Use 'n_features', 'transformers' ou 'f1'.")


def write_many(pairs: list[tuple[str, str]], outdir: Path) -> list[Path]:
    out_paths: list[Path] = []
    for filename, content in pairs:
        p = outdir / filename
        p.write_text(content, encoding="utf-8")
        out_paths.append(p)
    return out_paths


def main():
    parser = argparse.ArgumentParser(
        description="Gera .tex: n_features, transformers e f1 (ChatGPT e/ou SST)."
    )
    parser.add_argument(
        "--fig",
        choices=["n_features", "transformers", "f1", "both", "allfigs"],
        default="both",
        help="Tipo de figura: 'both' = n_features+transformers; 'allfigs' = todos (inclui f1).",
    )
    parser.add_argument(
        "--bases",
        choices=["chatgpt", "sst", "both"],
        default="both",
        help="Para --fig f1: quais bases gerar (padrão: both). Ignorado para os demais.",
    )
    parser.add_argument("--tipo", help="easy | medium | hard")
    parser.add_argument("--numero", type=int, choices=range(1, 9), help="Grupo (1..8)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Gerar para todas as bases e grupos. Para f1 e --bases=both, inclui ChatGPT e SST.",
    )
    parser.add_argument("--outdir", default=".", help="Diretório de saída (padrão: .)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Quais 'fig_kinds' efetivamente gerar
    if args.fig == "both":
        fig_kinds = ["n_features", "transformers"]
    elif args.fig == "allfigs":
        fig_kinds = ["n_features", "transformers", "f1"]
    else:
        fig_kinds = [args.fig]

    generated: list[Path] = []

    if args.all:
        # Todos os grupos 1..8; para ChatGPT, todas as bases easy/medium/hard
        for n in range(1, 9):
            for fig in fig_kinds:
                if fig in ("n_features", "transformers"):
                    for tb in ("easy", "medium", "hard"):
                        pairs = render_tex(fig_kind=fig, tipo_base=tb, numero=n)
                        generated += write_many(pairs, outdir)
                elif fig == "f1":
                    # ChatGPT e/ou SST conforme --bases
                    if args.bases in ("chatgpt", "both"):
                        for tb in ("easy", "medium", "hard"):
                            pairs = render_tex(
                                fig_kind="f1",
                                tipo_base=tb,
                                numero=n,
                                base_scope="chatgpt",
                            )
                            generated += write_many(pairs, outdir)
                    if args.bases in ("sst", "both"):
                        pairs = render_tex(
                            fig_kind="f1", tipo_base=None, numero=n, base_scope="sst"
                        )
                        generated += write_many(pairs, outdir)
    else:
        # Execução pontual
        if not args.numero:
            parser.error("Informe --numero (1..8) ou use --all.")
        for fig in fig_kinds:
            if fig in ("n_features", "transformers"):
                if not args.tipo:
                    parser.error(
                        "Para n_features/transformers, informe --tipo (easy|medium|hard) ou use --all."
                    )
                pairs = render_tex(
                    fig_kind=fig, tipo_base=args.tipo, numero=args.numero
                )
                generated += write_many(pairs, outdir)
            elif fig == "f1":
                # f1 pode gerar ChatGPT (requer --tipo) e/ou SST (não requer)
                if args.bases in ("chatgpt", "both"):
                    if not args.tipo:
                        parser.error(
                            "Para f1 com base ChatGPT, informe --tipo (easy|medium|hard) ou use --bases sst."
                        )
                    generated += write_many(
                        render_tex(
                            fig_kind="f1",
                            tipo_base=args.tipo,
                            numero=args.numero,
                            base_scope="chatgpt",
                        ),
                        outdir,
                    )
                if args.bases in ("sst", "both"):
                    generated += write_many(
                        render_tex(
                            fig_kind="f1",
                            tipo_base=None,
                            numero=args.numero,
                            base_scope="sst",
                        ),
                        outdir,
                    )

    for path in generated:
        print(f"✔ Gerado: {path}")


if __name__ == "__main__":
    main()
