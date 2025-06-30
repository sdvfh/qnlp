from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from utils import models, read_summary

# --------------------------------------------------------------------- #
#                             CONFIGURAÇÃO                              #
# --------------------------------------------------------------------- #
CSV_MEASURES = Path("../../results/measures_all.csv")
DATASETS = ("chatgpt_easy", "chatgpt_medium", "chatgpt_hard", "sst")
EPS = 1e-3

MARKERS = {1: "o", 10: "^"}
OFFSETS = [(9, -5)]

ZOOMS = [
    (0.784, 0.830, 0.265e-2, 0.470e-2, (0.75, -0.65, 0.35, 0.35)),
    (0.381, 0.3865, 1.20e-2, 0.400e0, (0.05, -0.65, 0.35, 0.35)),
]


# --------------------------- FUNÇÕES AUXILIARES --------------------------- #
def place_label(
    ax, x, y, label, xs_others, ys_others, *, offsets=OFFSETS, min_sep=1, fontsize=7
):
    """Rótulo que evita colisão grosseira com outros pontos."""
    xd, yd = ax.transData.transform(np.column_stack([xs_others, ys_others])).T
    for dx, dy in offsets:
        off = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
        xt, yt = off.transform((x, y))
        if (np.hypot(xd - xt, yd - yt) > min_sep).all():
            ax.annotate(
                label,
                (x, y),
                (dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                ha="center",
                va="center",
            )
            return
    dx, dy = offsets[0]
    ax.annotate(
        label,
        (x, y),
        (dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "gray"},
    )


def scatter_by_layer(ax, data):
    """Espalhamento por nº de camadas.
    Devolve o primeiro PathCollection (necessário p/ o colorbar)."""
    sc_main = None
    for n_layers, marker in MARKERS.items():
        sub = data[data["n_layers"] == n_layers]
        sc = ax.scatter(
            sub["ent"],
            sub["exp"].clip(lower=EPS),
            c=sub["f1"],
            cmap="viridis",
            marker=marker,
            s=80,
            edgecolors="black",
            label=f"{n_layers} camada{'s' if n_layers > 1 else ''}",
            zorder=2,
        )
        if sc_main is None:
            sc_main = sc
    return sc_main


# --------------------------------- MAIN ---------------------------------- #
def plot_dataset(dataset: str) -> None:
    """Gera PDF **e** PGF (+ PNG) para um dataset."""
    df_meas = pd.read_csv(CSV_MEASURES)
    df_meas["model"] = df_meas["model"].replace(models)
    df_meas = df_meas[df_meas["with_state_prep"]]

    df_runs = read_summary().query("dataset == @dataset")

    df = (
        df_meas.merge(
            df_runs[["model_classifier", "n_layers", "f1"]],
            left_on=["model", "n_layers"],
            right_on=["model_classifier", "n_layers"],
        )
        .drop(columns="model_classifier")
        .groupby(["model", "n_layers"], as_index=False, sort=False)
        .agg(ent=("ent", "mean"), exp=("exp", "mean"), f1=("f1", "mean"))
    )

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sc_main = scatter_by_layer(ax, df)

    # rótulos fora dos zooms
    mask_in_zoom = sum(
        df["ent"].between(x0, x1) & df["exp"].between(y0, y1)
        for x0, x1, y0, y1, _ in ZOOMS
    ).astype(bool)
    for _, r in df[~mask_in_zoom].iterrows():
        others = df.loc[df.index != r.name]
        place_label(
            ax,
            r["ent"],
            max(r["exp"], EPS),
            str(int(r["model"])),
            others["ent"].values,
            others["exp"].clip(lower=EPS).values,
        )

    # eixo principal
    safe_title = dataset.replace("_", r"\_").replace("%", r"\%")  # ← escapa LaTeX
    ax.set(
        yscale="log",
        xlabel="Emaranhamento (média)",
        ylabel="Expressabilidade (média, log)",
        title=f"Panorama das medidas — {safe_title}",
    )
    ax.grid(True, ls="--", lw=0.6, alpha=0.5)
    ax.legend(title="Legenda")

    # colorbar – usa o PathCollection real → gradiente OK em PDF e PGF
    fig.colorbar(sc_main, ax=ax, pad=0.02).set_label("F1-score (média)")

    # janelas de zoom
    for x0, x1, y0, y1, bbox in ZOOMS:
        axins = inset_axes(
            ax,
            "120%",
            "120%",
            bbox_to_anchor=bbox,
            bbox_transform=ax.transAxes,
            loc="lower left",
            borderpad=0,
        )
        axins.set(yscale="log", xlim=(x0, x1), ylim=(y0, y1))
        scatter_by_layer(axins, df)

        sub = df[df["ent"].between(x0, x1) & df["exp"].between(y0, y1)]
        for _, r in sub.iterrows():
            others = sub.drop(r.name)
            place_label(
                axins,
                r["ent"],
                max(r["exp"], EPS),
                str(int(r["model"])),
                others["ent"].values,
                others["exp"].clip(lower=EPS).values,
            )

        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="black", ls="--", lw=0.8)

    fig.subplots_adjust(bottom=0.40, right=1)

    out_dir = Path("../../figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------- salva PGF + PNG (para Overleaf) ----
    pgf_path = out_dir / f"ent_exp_f1_{dataset}.pgf"
    fig.savefig(pgf_path, bbox_inches="tight")
    # todos os PNGs gerados ficam no mesmo diretório de pgf_path

    plt.close(fig)
    print(f"Saved: {pgf_path.name} (+ PNGs)")


if __name__ == "__main__":
    for ds in DATASETS:
        plot_dataset(ds)
