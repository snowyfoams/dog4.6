# Gait timing diagram: edition 1 (period 0.34 s) vs edition 2 (0.42 s).
# Gantt bars of stance/swing for the 4 legs over ~2.5 cycles, diagonal pairing
# (FR+RL offset 0, FL+RR offset 0.5), duty 0.60. Highlights the +48 ms stance.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.bbox": "tight",
                     "savefig.dpi": 300})

LEGS = ["FR", "FL", "RR", "RL"]
OFFSET = {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0}   # matches TrotGait.offset
DUTY = 0.60
C_ST, C_SW = "#3b7dd8", "#e8a33d"      # stance / swing


def stance_intervals(period, offset, t_end):
    """List of (start, dur) stance intervals for one leg over [0, t_end]."""
    out = []
    k = -1
    while True:
        # cycle k starts (in this leg's local phase) at t where local phase hits 0
        t0 = (k - offset) * period
        if t0 > t_end:
            break
        if t0 + DUTY * period > 0:
            out.append((max(t0, 0.0), min(t0 + DUTY * period, t_end) - max(t0, 0.0)))
        k += 1
    return [(s, d) for s, d in out if d > 0]


def draw(ax, period, title, t_end):
    for i, leg in enumerate(LEGS):
        y = len(LEGS) - 1 - i
        # full strip = swing background
        ax.broken_barh([(0, t_end)], (y - 0.4, 0.8), facecolors=C_SW, edgecolor="none")
        # stance blocks on top
        ax.broken_barh(stance_intervals(period, OFFSET[leg], t_end),
                       (y - 0.4, 0.8), facecolors=C_ST, edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(LEGS)))
    ax.set_yticklabels(LEGS[::-1])
    ax.set_xlim(0, t_end)
    ax.set_ylim(-0.75, len(LEGS) + 0.15)
    ax.set_xlabel("time [s]")
    ax.set_title(title, loc="left", fontsize=9.5, pad=16)
    # period bracket above the top leg (FR, y=3)
    ax.annotate("", xy=(0, 3.6), xytext=(period, 3.6),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(period / 2, 3.68, "period %.2f s" % period, ha="center", va="bottom", fontsize=7)
    # stance-duration label on FR's first stance
    ax.text(DUTY * period / 2, 3, "%.0f ms" % (DUTY * period * 1000),
            ha="center", va="center", fontsize=6.5, color="white")


T_END = 1.0
fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.4), sharex=True)
draw(axes[0], 0.34, "(a) edition 1 — DOG3.0 inherited (period 0.34 s, stance 204 ms)", T_END)
draw(axes[1], 0.42, "(b) edition 2 — platform retune (period 0.42 s, stance 252 ms)", T_END)

# highlight the extra 48 ms of stance under edition 2's FR first block
x1 = 0.34 * DUTY        # 0.204
x2 = 0.42 * DUTY        # 0.252
axes[1].axvspan(x1, x2, ymin=0.0, ymax=1.0, color="#2ca02c", alpha=0.20, zorder=0)
axes[1].annotate("+48 ms stance\nper step vs ed.1", xy=(x2, -0.42), xytext=(x2 + 0.07, -0.66),
                 ha="left", va="center", fontsize=7, color="#1d6b1d",
                 arrowprops=dict(arrowstyle="->", color="#1d6b1d", lw=0.9))

handles = [Patch(facecolor=C_ST, label="stance (foot down)"),
           Patch(facecolor=C_SW, label="swing (foot up)"),
           Patch(facecolor="#2ca02c", alpha=0.25, label="added stance time")]
fig.legend(handles=handles, ncol=3, frameon=False, fontsize=7.5,
           loc="upper center", bbox_to_anchor=(0.5, 1.05))
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_gait_schedule." + ext))
print("wrote fig_gait_schedule.{png,pdf}")
