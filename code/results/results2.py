from pathlib import Path

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


# --------------------------------------------------------------------- #
#                        FUNÇÕES AUXILIARES                             #
# --------------------------------------------------------------------- #
def place_label(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    xs_others: np.ndarray,
    ys_others: np.ndarray,
    *,
    offsets=OFFSETS,
    min_sep: float = 1,
    fontsize: int = 7,
) -> None:
    """Anota *label* perto de (x, y) evitando colidir com outros pontos."""
    xd, yd = ax.transData.transform(np.column_stack([xs_others, ys_others])).T
    for dx, dy in offsets:
        off = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
        xt, yt = off.transform((x, y))
        if (np.hypot(xd - xt, yd - yt) > min_sep).all():
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                ha="center",
                va="center",
            )
            return
    dx, dy = offsets[0]
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "gray"},
    )


def scatter_by_layer(ax: plt.Axes, data: pd.DataFrame):
    """
    Plota o espalhamento por número de camadas
    e devolve o primeiro PathCollection (para o colorbar).
    """
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
            sc_main = sc  # guarda o primeiro PathCollection
    return sc_main


# --------------------------------------------------------------------- #
#                                MAIN                                   #
# --------------------------------------------------------------------- #
def plot_dataset(dataset: str) -> None:
    """Gera e salva a figura para um *dataset*."""
    df_meas = pd.read_csv(CSV_MEASURES)
    df_meas["model"] = df_meas["model"].replace(models)
    df_meas = df_meas[df_meas["with_state_prep"]]

    df_runs = read_summary()
    df_runs = df_runs[df_runs["dataset"] == dataset]

    df = (
        df_meas.merge(
            df_runs[["model_classifier", "n_layers", "f1"]],
            left_on=["model", "n_layers"],
            right_on=["model_classifier", "n_layers"],
            how="inner",
        )
        .drop(columns="model_classifier")
        .groupby(["model", "n_layers"], as_index=False, sort=False)
        .agg(ent=("ent", "mean"), exp=("exp", "mean"), f1=("f1", "mean"))
    )

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sc_main = scatter_by_layer(ax, df)  # ← capturamos o mappable real

    # rótulos apenas fora das janelas de zoom
    mask_in_zoom = sum(
        df["ent"].between(x_min, x_max) & df["exp"].between(y_min, y_max)
        for x_min, x_max, y_min, y_max, _ in ZOOMS
    ).astype(bool)

    for idx, row in df[~mask_in_zoom].iterrows():
        others = df.loc[df.index != idx]
        place_label(
            ax,
            row["ent"],
            max(row["exp"], EPS),
            str(int(row["model"])),
            others["ent"].values,
            others["exp"].clip(lower=EPS).values,
        )

    ax.set(
        yscale="log",
        xlabel="Emaranhamento (média)",
        ylabel="Expressabilidade (média, log)",
        title=f"Panorama das medidas — {dataset}",
    )
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(title="Legenda")

    # --------- COLORBAR (agora usando o PathCollection real) -----------
    fig.colorbar(sc_main, ax=ax, pad=0.02).set_label("F1-score (média)")

    # --- janelas de zoom --------------------------------------------- #
    for x_min, x_max, y_min, y_max, bbox in ZOOMS:
        axins = inset_axes(
            ax,
            width="120%",
            height="120%",
            bbox_to_anchor=bbox,
            bbox_transform=ax.transAxes,
            loc="lower left",
            borderpad=0,
        )
        axins.set(yscale="log", xlim=(x_min, x_max), ylim=(y_min, y_max))
        scatter_by_layer(axins, df)

        mask = df["ent"].between(x_min, x_max) & df["exp"].between(y_min, y_max)
        sub = df[mask]
        for idx, row in sub.iterrows():
            others = sub.drop(idx)
            place_label(
                axins,
                row["ent"],
                max(row["exp"], EPS),
                str(int(row["model"])),
                others["ent"].values,
                others["exp"].clip(lower=EPS).values,
            )

        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="black", ls="--", lw=0.8)

    fig.subplots_adjust(bottom=0.40, right=1)

    out_dir = Path("../../figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"ent_exp_f1_{dataset}.pgf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for ds in DATASETS:  # loop por datasets
        plot_dataset(ds)
