import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("/results/kl_values.csv")
df = df[df["circuit_name"] != "single_rot_z"]
df_1 = df[df["n_layers"] == 1]
df_10 = df[df["n_layers"] == 10]
df = pd.merge(df_1, df_10, on="circuit_name", suffixes=("_1", "_10"))
df = df.sort_values("KL_value_1", ascending=False)

fig, ax = plt.subplots(dpi=300)
# ax = plt.gca()
plt.plot("circuit_name", "KL_value_1", "or", data=df, mfc="none", label="L = 1")
plt.plot("circuit_name", "KL_value_10", "sb", data=df, mfc="none", label="L = 10")
plt.xticks(rotation=90)
ax.set_xlabel("Circuit name")
ax.set_ylabel("Expr, $D_{KL}$")
plt.subplots_adjust(left=0.18)

ax.annotate(
    "",
    xy=(-0.15, 1),
    xytext=(-0.15, 0),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "<->", "lw": 1.5, "color": "black"},
)
ax.text(-0.15, 1.02, "Low Expr", transform=ax.transAxes, ha="center", va="bottom")

ax.text(-0.15, -0.02, "High Expr", transform=ax.transAxes, ha="center", va="top")
ax.grid(which="major", linestyle="--", color="gray", alpha=0.5)
# ax.minorticks_on()
# ax.grid(
#     which="minor",
#     linestyle="--",
#     color="gray",
#     alpha=0.2
# )
plt.title("$D_{KL}$ values per circuit and layer")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("results/kl_values_diagram.png")
# plt.show()
