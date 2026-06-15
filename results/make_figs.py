# Publication figures for the DEFINITIVE platform DOG4.6 (correct current-controlled
# servo envelope). Single platform, the 2x2 control-stack ablation:
#   MIT = convex MPC + J^T leg ctrl     X1 = convex MPC + QP-WBC
#   ETH = SLQ-MPC + QP-WBC              X2 = SLQ-MPC + J^T leg ctrl
# High level: convex (MIT,X1) vs SLQ (ETH,X2). Low level: leg (MIT,X2) vs WBC (ETH,X1).
# Outputs (PNG 300dpi + PDF): fig_tracking, fig_robustness, fig_compute, table.tex
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(RES, "paper_figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.3,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

# (csv-prefix, short label, color, marker, high-level, low-level)
STACKS = [("MIT (", "MIT", "#1f77b4", "o", "convex", "leg"),
          ("ETH (", "ETH", "#d62728", "s", "SLQ", "WBC"),
          ("X1 (",  "X1 (cvx+WBC)", "#2ca02c", "^", "convex", "WBC"),
          ("X2 (",  "X2 (SLQ+leg)", "#9467bd", "D", "SLQ", "leg")]
MAGS = [10, 20, 30, 40, 50]
DIRS = ["+x", "-x", "+y", "-y"]


def rows(name):
    with open(os.path.join(RES, name), newline="") as fp:
        return list(csv.DictReader(fp))


def s1(prefix):
    out = []
    for r in rows("s1_s2_metrics.csv"):
        if r["stack"].startswith(prefix) and r["scenario"] == "S1" and not int(r["fell"]):
            out.append((float(r["vx_cmd"]), float(r["vel_rmse"]),
                        float(r["att_rmse"]), float(r["cot"])))
    return sorted(out)


def s2(prefix):
    for r in rows("s1_s2_metrics.csv"):
        if r["stack"].startswith(prefix) and r["scenario"] == "S2":
            return float(r["max_speed"]), r["failure_mode"]
    return np.nan, ""


def push_grid(prefix):
    g = np.full((len(DIRS), len(MAGS)), np.nan)
    for r in rows("s5_push.csv"):
        if r["stack"].startswith(prefix):
            i, j = DIRS.index(r["direction"]), MAGS.index(int(r["push_N"]))
            g[i, j] = int(r["success"])
    return g


def solve(prefix):
    for r in rows("solve_times.csv"):
        if r["stack"].startswith(prefix):
            return float(r["mean_ms"]), float(r["p95_ms"])
    return np.nan, np.nan


# ------------------------------------------------ fig 1: S1 tracking 1x3
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharex=True)
cols = [("velocity RMSE [m/s]", 1), ("roll/pitch RMSE [deg]", 2), ("CoT [-]", 3)]
for prefix, label, color, mk, _, _ in STACKS:
    d = s1(prefix)
    if not d:
        continue
    x = [p[0] for p in d]
    for j, (ylab, idx) in enumerate(cols):
        axes[j].plot(x, [p[idx] for p in d], marker=mk, ms=4, color=color, label=label)
for j, (ylab, _) in enumerate(cols):
    axes[j].set_xlabel(r"commanded $v_x$ [m/s]"); axes[j].set_ylabel(ylab)
    axes[j].set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
axes[0].legend(frameon=False)
fig.suptitle("S1 velocity staircase (DOG4.6 servo envelope)", y=1.02, fontsize=9.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_tracking." + ext))
plt.close(fig)

# ----------------------------- fig 2: S2 bars + S5 push heatmaps (1 + 4 panels)
fig = plt.figure(figsize=(7.2, 2.7))
gs = fig.add_gridspec(1, 5, width_ratios=[2.0, 1, 1, 1, 1], wspace=0.35)
ax0 = fig.add_subplot(gs[0, 0])
xs = np.arange(len(STACKS))
vs, fms = [], []
FM = {"roll-over": "RO", "height collapse": "HC", "pitch-over": "PO"}
for prefix, _, _, _, _, _ in STACKS:
    v, fm = s2(prefix); vs.append(v); fms.append(FM.get(fm, ""))
bars = ax0.bar(xs, vs, color=[c for _, _, c, _, _, _ in STACKS],
               edgecolor="black", linewidth=0.4)
for rect, v, fm in zip(bars, vs, fms):
    ax0.text(rect.get_x() + rect.get_width() / 2, v + 0.012, "%.2f\n%s" % (v, fm),
             ha="center", va="bottom", fontsize=6.5)
ax0.set_xticks(xs, [l for _, l, _, _, _, _ in STACKS], rotation=15)
ax0.set_ylabel("max stable speed [m/s]")
ax0.set_title("(a) S2 ramp-to-failure", loc="left", fontsize=8.5)
ax0.set_ylim(0, max(0.75, np.nanmax(vs) + 0.12))

for k, (prefix, label, color, _, _, _) in enumerate(STACKS):
    ax = fig.add_subplot(gs[0, k + 1])
    g = push_grid(prefix)
    ax.imshow(g, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MAGS)), MAGS, fontsize=6)
    ax.set_yticks(range(len(DIRS)), DIRS if k == 0 else [""] * len(DIRS), fontsize=6)
    ax.set_title("%s (%d/20)" % (label.split(" ")[0], int(np.nansum(g))),
                 fontsize=7, color=color)
    ax.set_xlabel("push [N]", fontsize=6.5)
    ax.tick_params(length=0)
fig.text(0.62, 1.0, "(b) S5 push recovery (green = recovered, 0.1 s impulse)",
         ha="center", fontsize=8.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_robustness." + ext))
plt.close(fig)

# ------------------------------------------------------- fig 3: solve time
fig, ax = plt.subplots(figsize=(3.3, 2.6))
means = [solve(p[0])[0] for p in STACKS]
p95s = [solve(p[0])[1] for p in STACKS]
ax.bar(xs, means, color=[c for _, _, c, _, _, _ in STACKS], edgecolor="black",
       linewidth=0.4, label="mean")
ax.errorbar(xs, means, yerr=[np.zeros(len(xs)), np.array(p95s) - np.array(means)],
            fmt="none", ecolor="black", capsize=3, lw=0.8, label="p95")
for x, m in zip(xs, means):
    ax.text(x, m + 0.4, "%.1f" % m, ha="center", fontsize=7)
ax.set_xticks(xs, [l.split(" ")[0] for _, l, _, _, _, _ in STACKS], rotation=15)
ax.set_ylabel("MPC solve time [ms]")
ax.set_title("(c) compute (convex=OSQP/C, SLQ=numpy)", loc="left", fontsize=8.5)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_compute." + ext))
plt.close(fig)

# ----------------------------------------------------------- LaTeX table
def best_cot(prefix):
    d = s1(prefix)
    return min(p[3] for p in d) if d else float("nan")


def max_tracked(prefix):
    best = 0.0
    for r in rows("s1_s2_metrics.csv"):
        if r["stack"].startswith(prefix) and r["scenario"] == "S1" and not int(r["fell"]):
            if float(r["vel_mean"]) >= 0.65 * float(r["vx_cmd"]):
                best = max(best, float(r["vx_cmd"]))
    return best


lines = [
    r"\begin{table}[t]\centering",
    r"\caption{The $2\times2$ control-stack ablation on DOG4.6 (physically correct",
    r"current-controlled actuator envelope). High level: convex MPC vs SLQ-MPC;",
    r"low level: $J^\top$ leg controller vs QP whole-body controller.}",
    r"\label{tab:ablation_4p6}",
    r"\begin{tabular}{llccccc}",
    r"\toprule",
    r"stack & high$\,/\,$low & $v_\mathrm{max}$ & $v_\mathrm{track}$ & push & best & solve \\",
    r" & & [m/s] & [m/s] & /20 & CoT & [ms] \\",
    r"\midrule",
]
for prefix, label, _, _, hi, lo in STACKS:
    v, _ = s2(prefix)
    vt = max_tracked(prefix)
    pu = int(np.nansum(push_grid(prefix)))
    cot = best_cot(prefix)
    sm, _ = solve(prefix)
    lines.append("%s & %s/%s & %.2f & %.2f & %d & %.2f & %.1f \\\\" % (
        label.replace("+", r"$+$"), hi, lo, v, vt, pu, cot, sm))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "table.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("wrote:", sorted(os.listdir(OUT)))
