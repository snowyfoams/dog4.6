r"""Dissertation Ch.6 comparison benchmark — four stacks, same robot, same
shared modules (gait clock, contact-aware layer, Raibert/Bezier swing
trajectories, reference generation, station keeping, ground-truth state):

  mit    : convex MPC (one-shot QP, yaw-linearised SRB) + J^T leg controller
  eth    : SLQ-MPC (iterated DP / discrete iLQR, nonlinear SRB) + QP-WBC
  cvxwbc : convex MPC + QP-WBC          (RQ4 cross-combination)
  slqjt  : SLQ-MPC + J^T leg controller (RQ4 cross-combination)

Scenarios (Ch.6 sec 6.4):  S1 velocity staircase | S2 ramp to failure | S5 push recovery
Metrics  (Ch.6 sec 6.5):  velocity RMSE, attitude RMSE, CoT, torque saturation,
                          max stable speed + failure mode, push success, solve times.

Usage (slices append/replace rows in results/*.csv, so runs are resumable):
  python -u tests/compare_stacks.py run mit s1s2
  python -u tests/compare_stacks.py run mit s5
  ... (each stack x {s1s2, s5}) ...
  python    tests/compare_stacks.py report     # summary md + figures from CSVs

NOTE on solve times (RQ3 caveat): convex MPC solves with OSQP (compiled C), the
SLQ solver is pure numpy — absolute times are implementation-language-bound.
"""
import os
import sys
import csv
import time
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scene'))
import actuator_envelope
from srb_mpc.controller import LocomotionController
from srb_mpc.eth_controller import LocomotionControllerETH

SCENE = r"D:\mujoco\DOG4.6_description\scene\DOG4.6_scene.xml"
RESULTS = r"D:\mujoco\DOG4.6_description\results"
os.makedirs(RESULTS, exist_ok=True)

M = mujoco.MjModel.from_xml_path(SCENE)
actuator_envelope.install()   # must come AFTER model load (compile-time callback crash)
TAU_MAX = float(M.actuator_ctrlrange[0, 1])

STACKS = {
    "mit": ("MIT (convexMPC + leg ctrl)",
            lambda: LocomotionController(M, stand=True)),
    "eth": ("ETH (SLQ-MPC + QP-WBC)",
            lambda: LocomotionControllerETH(M, stand=True)),
    "cvxwbc": ("X1 (convexMPC + QP-WBC)",
               lambda: LocomotionControllerETH(M, stand=True,
                                               high_level="convex", low_level="wbc")),
    "slqjt": ("X2 (SLQ-MPC + leg ctrl)",
              lambda: LocomotionControllerETH(M, stand=True,
                                              high_level="slq", low_level="jt")),
}

S1_SPEEDS = [0.1, 0.2, 0.3, 0.4, 0.5]
S1_HOLD = 3.0
S2_STEP, S2_HOLD = 0.05, 2.0
S5_MAGS = [10, 20, 30, 40, 50]
S5_DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}

CSV_MAIN = os.path.join(RESULTS, "s1_s2_metrics.csv")
CSV_PUSH = os.path.join(RESULTS, "s5_push.csv")
CSV_SOLVE = os.path.join(RESULTS, "solve_times.csv")
MAIN_KEYS = ["stack", "scenario", "vx_cmd", "vel_rmse", "vel_mean", "att_rmse",
             "cot", "sat_rate", "fell", "max_speed", "failure_mode"]


def fresh(make):
    d = mujoco.MjData(M)
    ctrl = make()
    ctrl.reset(d)
    return ctrl, d


def fallen(rob, d):
    return (not np.isfinite(d.qpos).all()
            or np.degrees(np.abs(rob.rpy(d)[:2])).max() > 45
            or rob.com(d)[2] < 0.07)


def failure_mode(rob, d):
    if rob.com(d)[2] < 0.07:
        return "height collapse"
    r, p = np.abs(rob.rpy(d)[:2])
    return "roll-over" if r > p else "pitch-over"


# ---------------------------------------------------------------- S1 staircase
def s1_staircase(name, make):
    ctrl, d = fresh(make)
    rob = ctrl.rob
    dt = M.opt.timestep
    rows = []
    for _ in range(int(1.0 / dt)):
        ctrl.gait.stand = True
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    ctrl.gait.stand = False
    for _ in range(int(1.0 / dt)):
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    for phase_v in S1_SPEEDS:
        ctrl.cmd.vx = phase_v
        vels, atts, sats = [], [], []
        E = 0.0
        p0 = rob.com(d).copy()
        n, n_settle = int(S1_HOLD / dt), int(0.5 / dt)
        fell = False
        for i in range(n):
            tau = ctrl(d)
            d.ctrl[:] = tau
            mujoco.mj_step(M, d)
            if i >= n_settle:
                vels.append(rob.com_vel(d)[0])
                atts.append(np.degrees(rob.rpy(d)[:2]))
                sats.append(np.mean(np.abs(tau) >= 0.98 * TAU_MAX))
                E += np.sum(np.abs(tau * d.qvel[6:])) * dt
            if fallen(rob, d):
                fell = True
                break
        dist = max(rob.com(d)[0] - p0[0], 1e-6)
        vels = np.asarray(vels); atts = np.asarray(atts)
        rows.append(dict(
            stack=name, scenario="S1", vx_cmd=phase_v,
            vel_rmse=float(np.sqrt(np.mean((vels - phase_v) ** 2))) if len(vels) else np.nan,
            vel_mean=float(vels.mean()) if len(vels) else np.nan,
            att_rmse=float(np.sqrt(np.mean(atts ** 2))) if len(atts) else np.nan,
            cot=float(E / (rob.mass * rob.g * dist)),
            sat_rate=float(np.mean(sats)) if len(sats) else np.nan,
            fell=int(fell)))
        print(f"  S1 vx={phase_v:.1f}: rmse={rows[-1]['vel_rmse']:.3f} "
              f"mean={rows[-1]['vel_mean']:.2f} att={rows[-1]['att_rmse']:.1f}deg "
              f"CoT={rows[-1]['cot']:.2f} sat={rows[-1]['sat_rate']*100:.1f}% fell={fell}")
        if fell:
            break
    return rows


# --------------------------------------------------------------- S2 ramp test
def s2_ramp(name, make):
    ctrl, d = fresh(make)
    rob = ctrl.rob
    dt = M.opt.timestep
    for _ in range(int(1.0 / dt)):
        ctrl.gait.stand = True
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    ctrl.gait.stand = False
    for _ in range(int(1.0 / dt)):
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    v, vmax, mode = 0.0, 0.0, "none (reached ramp end)"
    while v < 1.0:
        v = round(v + S2_STEP, 3)
        ctrl.cmd.vx = v
        fell = False
        for _ in range(int(S2_HOLD / dt)):
            d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
            if fallen(rob, d):
                fell = True
                mode = failure_mode(rob, d)
                break
        if fell:
            break
        vmax = v
    print(f"  S2: max stable speed = {vmax:.2f} m/s, failure mode: {mode}")
    return [dict(stack=name, scenario="S2", vx_cmd=vmax, vel_rmse=np.nan,
                 vel_mean=np.nan, att_rmse=np.nan, cot=np.nan, sat_rate=np.nan,
                 fell=0, max_speed=vmax, failure_mode=mode)]


# ------------------------------------------------------------ solve-time pass
def solve_pass(name, make):
    ctrl, d = fresh(make)
    dt = M.opt.timestep
    for k in range(int(4.0 / dt)):
        ctrl.gait.stand = k * dt < 1.0
        ctrl.cmd.vx = 0.3 if k * dt >= 1.0 else 0.0
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    st = np.array(ctrl.solve_times) * 1e3
    print(f"  MPC solve: mean={st.mean():.2f} ms  p95={np.percentile(st,95):.2f} ms")
    return dict(stack=name, mean_ms=float(st.mean()),
                p95_ms=float(np.percentile(st, 95)), n=len(st))


# ----------------------------------------------------------- S5 push recovery
def s5_push(name, make):
    rows = []
    dt = M.opt.timestep
    trunk = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    for dname, (dx_, dy_) in S5_DIRS.items():
        for mag in S5_MAGS:
            ctrl, d = fresh(make)
            rob = ctrl.rob
            for _ in range(int(1.0 / dt)):
                ctrl.gait.stand = True
                d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
            ctrl.gait.stand = False
            ok = True
            for k in range(int(4.5 / dt)):
                t = k * dt
                if 1.0 <= t < 1.1:
                    d.xfrc_applied[trunk, 0] = mag * dx_
                    d.xfrc_applied[trunk, 1] = mag * dy_
                else:
                    d.xfrc_applied[trunk, :2] = 0.0
                d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
                if fallen(rob, d):
                    ok = False
                    break
            rows.append(dict(stack=name, scenario="S5", direction=dname,
                             push_N=mag, success=int(ok)))
        succ = [r["success"] for r in rows if r["direction"] == dname]
        print(f"  S5 {dname}: {''.join('Y' if s else 'n' for s in succ)} (mags {S5_MAGS})")
    return rows


# ------------------------------------------------------------------- CSV I/O
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fp:
        return list(csv.DictReader(fp))


def replace_rows(path, keys, new_rows, drop):
    rows = [r for r in load_csv(path) if not drop(r)]
    rows += [{k: r.get(k, "") for k in keys} for r in new_rows]
    with open(path, "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


# -------------------------------------------------------------------- report
def report():
    main = load_csv(CSV_MAIN)
    push = load_csv(CSV_PUSH)
    solve = load_csv(CSV_SOLVE)
    names = [n for _, (n, _) in STACKS.items()
             if any(r["stack"] == n for r in main + push)]

    with open(os.path.join(RESULTS, "comparison_summary.md"), "w", encoding="utf-8") as fp:
        fp.write("# Stack comparison summary (S1/S2/S5)\n\n")
        fp.write(f"Same robot, shared modules, ground-truth state; MuJoCo "
                 f"{mujoco.__version__}, dt={M.opt.timestep}. Cross-combinations "
                 f"X1/X2 answer RQ4 (attribution).\n\n")
        fp.write("## RQ1 — tracking (S1 staircase): vel RMSE [m/s] / att RMSE [deg] / CoT\n\n")
        fp.write("| vx cmd | " + " | ".join(names) + " |\n")
        fp.write("|---" * (1 + len(names)) + "|\n")
        for v in S1_SPEEDS:
            cells = []
            for n in names:
                r = next((x for x in main if x["stack"] == n and x["scenario"] == "S1"
                          and abs(float(x["vx_cmd"]) - v) < 1e-9), None)
                if r is None:
                    cells.append("—")
                elif int(r["fell"]):
                    cells.append(f"fell ({float(r['vel_mean']):.2f})")
                else:
                    cells.append(f"{float(r['vel_rmse']):.3f} / "
                                 f"{float(r['att_rmse']):.1f} / {float(r['cot']):.2f}")
            fp.write(f"| {v:.1f} | " + " | ".join(cells) + " |\n")
        fp.write("\n## RQ1 — max stable speed (S2, ramp to failure)\n\n")
        for n in names:
            r = next((x for x in main if x["stack"] == n and x["scenario"] == "S2"), None)
            if r:
                fp.write(f"- **{n}**: {float(r['max_speed']):.2f} m/s ({r['failure_mode']})\n")
        fp.write("\n## RQ2 — push recovery (S5; Y per magnitude "
                 f"{S5_MAGS} N for 0.1 s)\n\n")
        fp.write("| dir | " + " | ".join(names) + " |\n")
        fp.write("|---" * (1 + len(names)) + "|\n")
        for dname in S5_DIRS:
            cells = []
            for n in names:
                s = ["Y" if int(r["success"]) else "·" for r in push
                     if r["stack"] == n and r["direction"] == dname]
                cells.append(" ".join(s) if s else "—")
            fp.write(f"| {dname} | " + " | ".join(cells) + " |\n")
        fp.write("\n## RQ3 — computation (caveat: OSQP is compiled C; SLQ is numpy)\n\n")
        for r in solve:
            fp.write(f"- **{r['stack']}**: solve mean {float(r['mean_ms']):.2f} ms, "
                     f"p95 {float(r['p95_ms']):.2f} ms\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for n in names:
        rs = [r for r in main if r["stack"] == n and r["scenario"] == "S1"
              and not int(r["fell"])]
        vx = [float(r["vx_cmd"]) for r in rs]
        axes[0].plot(vx, [float(r["vel_rmse"]) for r in rs], "o-", label=n)
        axes[1].plot(vx, [float(r["att_rmse"]) for r in rs], "o-", label=n)
        axes[2].plot(vx, [float(r["cot"]) for r in rs], "o-", label=n)
    for ax, ttl, yl in zip(axes, ["Velocity tracking", "Attitude error", "Cost of transport"],
                           ["vx RMSE [m/s]", "roll/pitch RMSE [deg]", "CoT [-]"]):
        ax.set_xlabel("commanded vx [m/s]"); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "s1_tracking.png"), dpi=130)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.4))
    for ax, n in zip(np.atleast_1d(axes), names):
        grid = np.full((len(S5_DIRS), len(S5_MAGS)), np.nan)
        for i, dname in enumerate(S5_DIRS):
            for j, mag in enumerate(S5_MAGS):
                r = next((x for x in push if x["stack"] == n
                          and x["direction"] == dname and int(x["push_N"]) == mag), None)
                if r:
                    grid[i, j] = int(r["success"])
        ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(S5_MAGS)), S5_MAGS)
        ax.set_yticks(range(len(S5_DIRS)), list(S5_DIRS))
        ax.set_xlabel("push [N, 0.1 s]"); ax.set_title(n, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "s5_push.png"), dpi=130)
    print("report -> comparison_summary.md, s1_tracking.png, s5_push.png")


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "report":
        report()
        sys.exit(0)
    if len(sys.argv) != 4 or sys.argv[1] != "run":
        print(__doc__)
        sys.exit(1)
    key, scen = sys.argv[2], sys.argv[3]
    name, make = STACKS[key]
    t0 = time.perf_counter()
    print(f"=== {name} | {scen} ===")
    if scen == "s1s2":
        rows = s1_staircase(name, make) + s2_ramp(name, make)
        replace_rows(CSV_MAIN, MAIN_KEYS, rows, lambda r: r["stack"] == name)
        st = solve_pass(name, make)
        replace_rows(CSV_SOLVE, ["stack", "mean_ms", "p95_ms", "n"], [st],
                     lambda r: r["stack"] == name)
    elif scen == "s5":
        rows = s5_push(name, make)
        replace_rows(CSV_PUSH, ["stack", "scenario", "direction", "push_N", "success"],
                     rows, lambda r: r["stack"] == name)
    else:
        raise SystemExit(f"unknown scenario {scen}")
    print(f"[{time.perf_counter()-t0:.0f} s wall]")
