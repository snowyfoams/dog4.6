# DOG4.6 — additional dynamic results (logged trot, retuned gait, servo envelope)

Logged per control step from a stepped-command trot (0.1→0.2→0.3 m/s) on DOG4.6 with the
platform gait (period 0.42) and the actuator envelope active. Logger:
`scene/log_run.py` → `results/logs/<stack>.npz`; figures: `results/make_extra_figs.py`.
Steady analysis uses the 0.2 m/s window; ETH vs X2 (SLQ+leg) are the featured contrast.

## 1. Compute timing — `fig_timing`, `table_timing.tex`

Control period 2 ms (500 Hz); the MPC re-plans every 8 ticks → 16 ms budget per solve.

| stack | MPC mean | MPC p95 | MPC max | loop mean | loop>2ms | MPC>16ms |
|---|---|---|---|---|---|---|
| MIT (cvx/leg) | 5.2 | 15.2 | 45.3 | 0.9 ms | 13 % | 4.9 % |
| **ETH (SLQ/WBC)** | 24.0 | 25.1 | 29.0 | 4.2 ms | 12.5 % | **100 %** |
| X1 (cvx/WBC) | 6.5 | 23.3 | 47.7 | 2.0 ms | 13 % | 9.8 % |
| X2 (SLQ/leg) | 23.9 | 25.3 | 29.9 | 3.4 ms | 13 % | **100 %** |

- **The high level sets compute.** Convex MPC (OSQP/C) solves in ~5–6 ms and meets the
  16 ms budget ~90–95 % of the time; SLQ-MPC (numpy) takes ~24 ms and **misses the budget
  on every tick**. Caveat (as in RQ3): OSQP is compiled C, SLQ is pure numpy, so absolute
  times are language-bound — but the structural gap (one QP vs iterated Riccati sweeps) is
  real. A C/JAX SLQ would shrink this; as implemented, only the convex high level is
  real-time on this loop.
- The full control loop (MPC + WBC + swing) averages 0.9–4.2 ms; the ~13 % of ticks over
  2 ms are the replan ticks. WBC adds ~1.5 ms/tick (X1 2.0 vs MIT 0.9).

## 2. Joint trajectories — `fig_joint_traj` (ETH vs X2, FR leg)

- **ETH**: clean, small-amplitude periodic joint angles; peak joint speed ~15 rad/s in
  swing; applied torque stays comfortably inside ±7 N·m — no saturation.
- **X2**: the J^T leg controller drives a **much larger knee excursion (folds toward
  ~150°, near the joint limit)** and noisier torques. The leg crouches to hold the foot
  arc — a worse-conditioned gait than the WBC produces for the identical swing targets.

## 3. Torque–speed vs envelope — `fig_torque_speed` (all 4)

Every joint's (|ω|, |τ_applied|) over the 0.2 m/s window, on the flat/power/cliff curve:
- At 0.2 m/s **ETH/X1/X2 operate entirely inside the flat region** (peaks 144–189 rpm, far
  below the 320 rpm cliff): the envelope does **not** bind at this speed — consistent with
  the probe, which only saw envelope contact above ~0.5 m/s for the leg stacks.
- **MIT is the outlier**: it thrashes to 317 rpm with torques pinned at ±7 — the
  signature of its rock-over instability, not productive locomotion.
- Takeaway: the actuator envelope is a *high-speed* constraint; at the usable ≤0.2 m/s
  operating point it is slack for every stable stack, so the early-fall behaviour is a
  control/tuning effect, not an actuator-limit effect.

## 4. Dynamic tracking — `fig_tracking_dyn` (ETH vs X2)

- **ETH** tracks the command staircase tightly, holds body height 0.19–0.20 m, keeps
  roll/pitch < 3°.
- **X2** lags and oscillates in forward speed, **sags progressively to ~0.17 m**, and its
  attitude oscillation **grows to 6–8°** — a degraded, marginally-stable gait that stays
  upright over the 8 s window but is visibly worse. This is the dynamic face of the static
  result: the WBC low level owns attitude and economy.

## 5. Swing-foot tracking + GRF — `fig_foot_grf` (ETH vs X2)

- **Foot height**: actual follows the desired Bezier arc for both stacks, with a small
  apex undershoot (~0.05 vs 0.055 m).
- **Vertical GRF**: clear diagonal pairing (FR+RL load while FL+RR swing), force ≈ 0 in
  swing, ~36 N per foot in stance (×2 stance feet ≈ 72 N = body weight, a physics check),
  with sharp ~150–200 N touchdown impact spikes. X2's stance forces are choppier,
  matching its noisier body state.

## Synthesis

The dynamic logs reinforce the static benchmark and the layer attribution:
- **Compute is owned by the high level** (convex real-time, SLQ not in numpy).
- **Attitude, height, tracking smoothness and economy are owned by the low level**: the WBC
  (ETH) yields a clean gait; the J^T leg controller (X2) yields a crouching, rocking gait
  for the *same* high level and swing targets.
- **The actuator envelope is slack at the usable speed** — the binding constraints here are
  controller stability and tuning, not motor torque–speed, at ≤0.2 m/s.

Files: `scene/log_run.py`, `results/make_extra_figs.py`, `results/logs/*.npz`,
`results/paper_figs/fig_{timing,joint_traj,torque_speed,tracking_dyn,foot_grf}.{png,pdf}`,
`table_timing.tex`.
