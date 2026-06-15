# Extra results from logged trots (results/logs/*.npz): compute timing, joint
# trajectories, torque-speed vs actuator envelope, dynamic tracking, foot + GRF.
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RES)
sys.path.insert(0, os.path.join(ROOT, "scene"))
import actuator_envelope as env
LOG = os.path.join(RES, "logs")
OUT = os.path.join(RES, "paper_figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
                     "legend.fontsize": 7.5, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "savefig.dpi": 300})

STK = [("mit", "MIT", "#1f77b4"), ("eth", "ETH", "#d62728"),
       ("cvxwbc", "X1 (cvx+WBC)", "#2ca02c"), ("slqjt", "X2 (SLQ+leg)", "#9467bd")]
D = {k: dict(np.load(os.path.join(LOG, "%s.npz" % k), allow_pickle=True)) for k, _, _ in STK}
DT = 0.002
JCOL = {"abd": 0, "thigh": 1, "knee": 2}     # FR leg columns are 0,1,2
JC = {"abd": "#1f77b4", "thigh": "#ff7f0e", "knee": "#2ca02c"}


def steady(d):
    m = (d["vx_cmd"] > 0.19) & (d["vx_cmd"] < 0.21)
    idx = np.where(m)[0]
    return idx


def cyc_window(d, n_cyc=2.0, period=0.42):
    idx = steady(d)
    n = int(n_cyc * period / DT)
    return idx[:n] if len(idx) > n else idx


# ============================================================ 1. timing
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for k, lab, c in STK:
    mpc = D[k]["mpc_ms"]; mpc = mpc[np.isfinite(mpc)]
    xs = np.sort(mpc); ys = np.arange(1, len(xs) + 1) / len(xs)
    a1.plot(xs, ys, color=c, label=lab, lw=1.4)
a1.axvline(16, color="grey", ls=":", lw=0.9); a1.text(16, 0.05, " 16 ms\n MPC tick", fontsize=6)
a1.set_xscale("log"); a1.set_xlabel("MPC solve time [ms]"); a1.set_ylabel("CDF")
a1.set_title("(a) high-level solve-time CDF", loc="left", fontsize=8.5); a1.legend(frameon=False)
xs = np.arange(len(STK)); means = [np.mean(D[k]["loop_ms"]) for k, _, _ in STK]
p95 = [np.percentile(D[k]["loop_ms"], 95) for k, _, _ in STK]
a2.bar(xs, means, color=[c for _, _, c in STK], edgecolor="black", linewidth=0.4)
a2.errorbar(xs, means, yerr=[np.zeros(len(xs)), np.array(p95) - np.array(means)],
            fmt="none", ecolor="black", capsize=3, lw=0.8)
a2.axhline(2.0, color="k", ls="--", lw=0.8); a2.text(0.0, 2.1, "2 ms control period", fontsize=6.5)
for x, m in zip(xs, means):
    a2.text(x, m + 0.15, "%.1f" % m, ha="center", fontsize=7)
a2.set_xticks(xs, [l.split(" ")[0] for _, l, _ in STK], rotation=12)
a2.set_ylabel("full control-loop time [ms]")
a2.set_title("(b) per-tick loop wall time (mean, p95)", loc="left", fontsize=8.5)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_timing." + e))
plt.close(fig)

# timing table
lines = [r"\begin{table}[t]\centering",
         r"\caption{Compute on DOG4.6 (control period 2 ms; MPC re-plans every 8 ticks = 16 ms"
         r" budget). Convex = OSQP (compiled C); SLQ = numpy -- absolute times are"
         r" language-bound.}", r"\label{tab:timing}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"stack & MPC mean & MPC p95 & MPC max & loop mean & loop$>$2ms & MPC$>$16ms \\",
         r" & [ms] & [ms] & [ms] & [ms] & [\%] & [\%] \\", r"\midrule"]
for k, lab, _ in STK:
    mpc = D[k]["mpc_ms"]; mpc = mpc[np.isfinite(mpc)]; loop = D[k]["loop_ms"]
    lines.append("%s & %.2f & %.2f & %.2f & %.2f & %.1f & %.1f \\\\" % (
        lab.split(" ")[0], mpc.mean(), np.percentile(mpc, 95), mpc.max(),
        loop.mean(), 100 * np.mean(loop > 2.0), 100 * np.mean(mpc > 16.0)))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "table_timing.tex"), "w").write("\n".join(lines) + "\n")

# ============================================ 2. joint trajectories ETH vs X2
fig, axes = plt.subplots(3, 2, figsize=(7.0, 5.2), sharex="col")
for col, (k, lab) in enumerate([("eth", "ETH"), ("slqjt", "X2 (SLQ+leg)")]):
    d = D[k]; w = cyc_window(d)
    t = d["t"][w] - d["t"][w][0]
    con = d["contact"][w, 0]            # FR contact
    for ax in axes[:, col]:
        sw = ~con
        ax.fill_between(t, 0, 1, where=sw, transform=ax.get_xaxis_transform(),
                        color="orange", alpha=0.10, lw=0, step="mid")
    for j, c in JC.items():
        axes[0, col].plot(t, np.degrees(d["q"][w, JCOL[j]]), color=c, lw=1.2, label=j)
        axes[1, col].plot(t, d["qd"][w, JCOL[j]], color=c, lw=1.2)
        axes[2, col].plot(t, d["tau_app"][w, JCOL[j]], color=c, lw=1.2)
    axes[2, col].axhline(7, color="k", ls="--", lw=0.7); axes[2, col].axhline(-7, color="k", ls="--", lw=0.7)
    axes[0, col].set_title("%s — FR leg (orange = swing)" % lab, fontsize=8.5)
    axes[2, col].set_xlabel("time [s]")
axes[0, 0].set_ylabel("joint angle [deg]"); axes[1, 0].set_ylabel("joint vel [rad/s]")
axes[2, 0].set_ylabel("applied torque [N·m]"); axes[0, 0].legend(frameon=False, ncol=3)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_joint_traj." + e))
plt.close(fig)

# ===================================== 3. torque-speed vs envelope (2x2)
ws = np.linspace(0, 36, 300)
hi = np.array([env._hi(w) for w in ws])
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
for ax, (k, lab, c) in zip(axes.flat, STK):
    d = D[k]; w = steady(d)
    sp = np.abs(d["qd"][w]).ravel(); tq = np.abs(d["tau_app"][w]).ravel()
    ax.scatter(sp, tq, s=2, alpha=0.18, color=c, edgecolors="none")
    ax.plot(ws, hi, "k-", lw=1.2)
    ax.axvline(22.9, color="grey", ls=":", lw=0.8)
    ax.set_title("%s  (peak %.0f rpm)" % (lab.split(" ")[0], np.abs(d["qd"][w]).max() * 60 / (2 * np.pi)),
                 fontsize=8.5, color=c)
    ax.set_xlim(0, 36); ax.set_ylim(0, 7.6)
for ax in axes[1, :]:
    ax.set_xlabel("joint speed |ω| [rad/s]")
for ax in axes[:, 0]:
    ax.set_ylabel("applied |τ| [N·m]")
fig.suptitle("Torque–speed operating points vs actuator envelope (320 rpm cliff)", y=1.01, fontsize=9.5)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_torque_speed." + e))
plt.close(fig)

# ===================================== 4. dynamic tracking ETH vs X2
fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.0), sharex=True)
for k, lab, c in [("eth", "ETH", "#d62728"), ("slqjt", "X2 (SLQ+leg)", "#9467bd")]:
    d = D[k]; t = d["t"] - d["t"][0]
    axes[0].plot(t, d["com_vel"][:, 0], color=c, lw=1.1, label=lab)
    axes[1].plot(t, d["h"], color=c, lw=1.1)
    axes[2].plot(t, np.degrees(np.abs(d["rpy"][:, :2]).max(axis=1)), color=c, lw=1.1)
d0 = D["eth"]; t0 = d0["t"] - d0["t"][0]
axes[0].plot(t0, d0["vx_cmd"], "k--", lw=0.9, label="command")
axes[0].set_ylabel("forward speed [m/s]"); axes[0].legend(frameon=False, ncol=3)
axes[1].set_ylabel("body height [m]"); axes[1].axhline(0.07, color="grey", ls=":", lw=0.8)
axes[2].set_ylabel("max |roll,pitch| [deg]"); axes[2].set_xlabel("time [s]")
axes[0].set_title("Dynamic tracking over the velocity staircase", loc="left", fontsize=8.5)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_tracking_dyn." + e))
plt.close(fig)

# ===================================== 5. foot swing + GRF ETH vs X2
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
for col, (k, lab) in enumerate([("eth", "ETH"), ("slqjt", "X2 (SLQ+leg)")]):
    d = D[k]; w = cyc_window(d, 2.5)
    t = d["t"][w] - d["t"][w][0]
    # (top) FR foot height desired vs actual
    axes[0, col].plot(t, d["foot_pos"][w, 0, 2], color="#1f77b4", lw=1.3, label="actual")
    axes[0, col].plot(t, d["foot_des"][w, 0, 2], color="#d62728", lw=1.0, ls="--", label="desired")
    axes[0, col].set_title("%s — FR foot height" % lab, fontsize=8.5)
    # (bottom) vertical GRF, diagonal pair FR (idx0) + RL (idx3) vs FL(1)+RR(2)
    axes[1, col].plot(t, d["grf"][w, 0, 2], color="#d62728", lw=1.0, label="FR")
    axes[1, col].plot(t, d["grf"][w, 3, 2], color="#9467bd", lw=1.0, label="RL")
    axes[1, col].plot(t, d["grf"][w, 1, 2], color="#1f77b4", lw=1.0, ls="--", label="FL")
    axes[1, col].set_xlabel("time [s]")
axes[0, 0].set_ylabel("foot z [m]"); axes[0, 0].legend(frameon=False, fontsize=7)
axes[1, 0].set_ylabel("vertical GRF [N]"); axes[1, 0].legend(frameon=False, ncol=3, fontsize=7)
fig.suptitle("Swing-foot tracking and ground reaction forces", y=1.01, fontsize=9.5)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig_foot_grf." + e))
plt.close(fig)

print("wrote:", sorted(f for f in os.listdir(OUT) if "timing" in f or "joint" in f
                        or "torque_speed" in f or "tracking_dyn" in f or "foot_grf" in f))
