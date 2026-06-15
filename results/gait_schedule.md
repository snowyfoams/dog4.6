# Gait schedule: edition 1 (DOG3.0-inherited) → edition 2 (platform-tuned)

The trot schedule is a **shared module** — the same clock drives the MPC contact
constraints and the swing-leg controller for all four stacks (MIT, ETH, X1, X2), so a gait
change is part of the fair-comparison backbone, not a per-controller tweak. This note
documents the one schedule change made for the new robot and why it matters for anyone
running the ETH or MIT method on this hardware.

## The schedule (`srb_mpc/gait.py`, `TrotGait`)

Trot = diagonal pairs (FR+RL / FL+RR) move together; a leg is in stance while its local
phase < `duty`, else swinging. Two numbers define it:

| param | edition 1 (DOG3.0) | edition 2 (this robot) |
|---|---|---|
| `period` | 0.34 s | **0.42 s** |
| `duty`   | 0.60 | 0.60 |
| → stance time `t_stance = period·duty` | 0.204 s | 0.252 s |
| → swing time  `t_swing = period·(1−duty)` | 0.136 s | 0.168 s |

Only `period` changed; `duty` is unchanged, so the stance/swing split stays 60/40 — the
legs just get **more time per cycle**, lengthening stance by 48 ms. The schedule and the
added stance time are shown in `paper_figs/fig_gait_schedule.png` (Gantt bars for the four
legs, diagonal pairs FR+RL / FL+RR, the +48 ms stance band highlighted).

## Why edition 1 fails on this robot — the diagnosis

Edition 1 was tuned on DOG3.0 (5.57 kg, 18 N·m). The new robot is **heavier and weaker**
(7.334 kg, 7 N·m output). Held at 0.2 m/s, the failure mode is **lateral rock-over**, not
torque saturation (`scene/diagnose_fall.py`):
- torque saturation stays at 0 % until the instant of the fall,
- the body roll grows oscillation-by-oscillation (MIT: −9.7° → −13.8° → −27.4° → collapse),
- a stack that survives the steady 0.2 m/s (X2: 6 s upright) still falls in the staircase
  the moment the command **jumps** 0.1 → 0.2.

Mechanism: in a fast trot the single supporting diagonal pair has too little stance time to
arrest the roll that builds up while the other pair is airborne. On a heavier robot the
roll momentum per step is larger and the weaker actuators apply less corrective hip moment,
so the margin that DOG3.0 had at 0.34 s is gone. Lengthening stance gives the support pair
the time to null the roll before the next swing.

## The sweep (`scene/sweep_retune.py`) — why 0.42, and the conflict it exposed

Six shared-tuning combos, each applied identically to all four stacks, scored by how far an
S1-style staircase survives (2 s holds):

| combo | MIT | X1 | X2 | ETH | sum |
|---|---|---|---|---|---|
| stock (period 0.34) | 0.30 | 0.50 | 0.10 | 0.50 | 1.4 |
| **period 0.42** | 0.10 | 0.50 | **0.50** | 0.50 | **1.6** |
| period 0.50 | 0.10 | 0.40 | 0.50 | 0.50 | 1.5 |
| 0.46 + raibert_k 0.16 | 0.10 | 0.20 | 0.50 | 0.40 | 1.2 |
| 0.46 + softer swing + rai 0.16 | 0.20 | 0.20 | 0.20 | 0.40 | 1.0 |
| 0.50 + rai 0.18 + low step | 0.10 | 0.30 | 0.50 | 0.40 | 1.3 |

Findings:
- **Slowing the gait is the single effective knob.** 0.42 s rescues the SLQ+leg stack (X2)
  outright (0.10 → 0.50) while keeping the WBC stacks at 0.50.
- **More aggressive foot placement (`raibert_k` ↑) hurts the WBC stacks** — every raibert
  combo dropped ETH/X1, so it was rejected.
- **The stacks have conflicting optima.** MIT (convex + leg) actually prefers the *fast*
  0.34 gait; X2 (SLQ + leg) needs the *slow* one. No single shared schedule is optimal for
  all four — an inherent limit of one-schedule fair comparison, and a caution against
  reading absolute max-speeds as controller properties.

`period = 0.42` was chosen as the best total and the one that rescues the worst-off stack
without regressing the canonical ETH stack.

## Effect on the benchmark (full S1/S2/S5, `fig_tuning_compare.png`)

| stack | edition 1 | edition 2 |
|---|---|---|
| ETH | survives staircase but **station-keeps** (mean ≈ 0.08 at cmd 0.5, RMSE 0.41, att 12.9°) | **tracks** (mean = cmd 0.1–0.5, att < 0.8°, CoT 1.70 → **0.43**, push 15 → 18) |
| X1  | **falls at 0.2** | survives the whole staircase to 0.5 |
| MIT | falls at 0.2 | falls at 0.3 |
| X2  | falls at 0.2 | falls at 0.3; fine-ramp S2 0.40 → 0.10 (profile-sensitive) |

Edition 2 improved tracking, push and CoT across the board. The cost is a mild
profile-sensitivity in the leg-controller stacks (X2's fine-ramp S2 number drops) — the
robot sits near its stability boundary, so absolute speeds remain tuning-sensitive while
the **ranking structure is stable** (ETH best, MIT worst, WBC = robustness).

## Practical note for running ETH / MIT on this robot

- Start from `period ≈ 0.42 s, duty 0.60`. If pushing higher speed, lengthen stance first
  (raise `period`, keep `duty`) before touching swing gains.
- Avoid step changes in commanded velocity; ramp instead — the worst falls here come from
  command jumps at a fixed gait phase, not from steady-state speed.
- Leave `raibert_k` at 0.10; raising it destabilises the WBC low level on this platform.
- The remaining ceiling is set by the high level (SLQ > convex for survival) and the floor
  by the low level (WBC >> J^T leg for robustness) — tune the plant, not the controllers.

Files: `srb_mpc/gait.py` (the schedule), `scene/diagnose_fall.py` (failure mode),
`scene/sweep_retune.py` (the sweep), `results/stock_tuning_backup/` (edition-1 data).
