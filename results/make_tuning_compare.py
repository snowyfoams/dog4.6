# Stock (DOG3.0 tuning, gait 0.34) vs platform-retuned (gait 0.42) on DOG4.6.
# Proves the "only ETH survives" picture was largely a shared-tuning artifact.
# Reads retuned data from results/ and stock data from results/stock_tuning_backup/.
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(RES, "paper_figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "axes.grid": True, "grid.alpha": 0.3,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

STACKS = [("MIT (", "MIT", "#1f77b4", "o"), ("ETH (", "ETH", "#d62728", "s"),
          ("X1 (", "X1 (cvx+WBC)", "#2ca02c", "^"), ("X2 (", "X2 (SLQ+leg)", "#9467bd", "D")]
SETS = [(os.path.join(RES, "stock_tuning_backup"), "(a) stock tuning (gait 0.34 s, from DOG3.0)"),
        (RES, "(b) platform retune (gait 0.42 s)")]


def rows(d):
    with open(os.path.join(d, "s1_s2_metrics.csv"), newline="") as fp:
        return list(csv.DictReader(fp))


def s1(d, prefix):
    ok, fell = [], None
    for r in rows(d):
        if r["stack"].startswith(prefix) and r["scenario"] == "S1":
            tup = (float(r["vx_cmd"]), float(r["vel_rmse"]), float(r["vel_mean"]), float(r["att_rmse"]))
            if int(r["fell"]):
                fell = tup
            else:
                ok.append(tup)
    return sorted(ok), fell


# 2 rows (stock/retuned) x 3 cols (vel mean-vs-cmd, RMSE, att); fall marked with X
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
cols = [("forward mean speed [m/s]", 2), ("velocity RMSE [m/s]", 1), ("roll/pitch RMSE [deg]", 3)]
for i, (d, title) in enumerate(SETS):
    for prefix, label, color, mk in STACKS:
        ok, fell = s1(d, prefix)
        if ok:
            x = [p[0] for p in ok]
            for j, (ylab, idx) in enumerate(cols):
                axes[i, j].plot(x, [p[idx] for p in ok], marker=mk, ms=4, color=color,
                                label=label if j == 0 else None)
        if fell is not None:                       # mark the fall point with a hollow X
            for j, (ylab, idx) in enumerate(cols):
                axes[i, j].plot(fell[0], fell[idx], "x", color=color, ms=7, mew=1.6)
    axes[i, 0].set_title(title, loc="left", fontsize=8.5)
    for j, (ylab, _) in enumerate(cols):
        axes[i, j].set_ylabel(ylab)
# ideal tracking reference on the mean-speed panels
for i in range(2):
    axes[i, 0].plot([0.1, 0.5], [0.1, 0.5], "k--", lw=0.7, alpha=0.5, zorder=0)
for j in range(3):
    axes[1, j].set_xlabel(r"commanded $v_x$ [m/s]")
    axes[0, j].set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
axes[0, 0].legend(frameon=False, fontsize=7)
axes[0, 0].text(0.12, 0.46, "ideal", fontsize=6, rotation=33, alpha=0.6)
fig.suptitle("DOG4.6 velocity staircase: stock vs platform-retuned gait  (× = fell)", y=1.01, fontsize=9.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_tuning_compare." + ext))
plt.close(fig)
print("wrote fig_tuning_compare.{png,pdf}")
