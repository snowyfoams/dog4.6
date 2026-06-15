# Actuator model correction (DOG4.6) — response to the back-EMF damping review

## 1. The review's three verification points, answered

1. **Joint placement**: joints are gearbox-OUTPUT side (geometric nesting, `gear="1"`), so
   `damping`, `armature`, and any velocity bias all act joint-side. `armature=0.0085`
   = rotor 8.5e-5 kg·m² × N² — correctly reflected. (Review case C excluded.)
2. **The "nominal" 0.01 damping is a placeholder** inherited from the DOG3.0 template —
   the review's suspicion B is correct on this point. It has no measured provenance.
3. **The 0.0657 provenance is documented but its shape was wrong**: it is the secant
   through the spec points (0 rpm, 7 N·m) and (320 rpm, 160 W → 4.77 N·m) — never a
   kt²/R derivation. Its flaw is applying torque reduction at ALL speeds.

## 2. Where the review itself errs: current control cancels back-EMF damping

b_joint = N²·kt²/R = 100 × 0.1²/0.466 = **2.146 N·m·s/rad** is the slope of the
VOLTAGE-limited line. It acts as a viscous damper only for a voltage-driven motor.
The MG5010-i10 runs FOC torque (current) mode: the current loop raises voltage to cancel
back-EMF, so **below voltage saturation the net EMF damping is ≈ 0** — not 2.146 and not
0.0657. Counter-check: 2.146 N·m·s/rad everywhere would consume 21 N·m at a 10 rad/s leg
swing — 3× peak torque; the robot could not take a single step.

2.146 is, however, exactly the **cliff slope** of the true envelope:

```
tau_lim(w) = min( 7,            current limit            (flat to 22.9 rad/s)
                  160 / w,      power limit              (22.9 – 31.1 rad/s)
                  2.146·(33.51 − w) )   voltage cliff    (zero at 320 rpm)
```

Braking quadrant is regenerative → only the 7 N·m current limit binds.
Implemented as a MuJoCo user-bias callback (`scene/actuator_envelope.py`); the XML alone
falls back to the ideal ±7 source. Verified by 6-quadrant unit assertions and by probe:
with the envelope active, peak joint speed self-limits at exactly 33.4–33.5 rad/s (320 rpm).

## 3. Probe: does the envelope ever bind?

`scene/probe_joint_speeds.py`, trot 0.3→0.5→0.7 ramp (ideal mode = worst case):

| stack | peak |qvel| ideal | time > 22.9 rad/s | peak with envelope |
|---|---|---|---|
| MIT | 37.2 rad/s | 2.4 % | 33.4 (capped) |
| ETH | 22.2 rad/s | 0 % | 22.2 (untouched) |
| X2  | 38.7 rad/s | 1.9 % | 33.5 (capped) |

The envelope binds **rarely and selectively** — only the J^T-leg stacks' fast swings.

## 4. Benchmark: correct model (4.6) vs ideal (4.0) vs over-damped (4.5)

S2 max stable speed [m/s]:

| stack | 4.0 ideal | **4.6 correct** | 4.5 (+6.6× damping) |
|---|---|---|---|
| MIT | 0.05 | 0.00 | 0.50 |
| ETH | 0.65 | **0.65** (bit-identical) | 0.75 |
| X1  | 0.15 | **0.15** (bit-identical) | 0.80 |
| X2  | 0.65 | **0.40** | 0.55 |

Push /20: MIT 5→5, ETH 15→15, X1 17→17 (all identical), X2 13→14.
ETH and X1 reproduce 4.0 exactly (their joints never leave the flat region — the runs are
deterministic). The envelope's entire effect lands on the leg-controller stacks, X2 above all:
its S1 staircase now falls at 0.2 (was 0.4) and S2 drops 0.65 → 0.40.

## 5. Revised conclusions for this 7 N·m platform

1. **With the physically correct actuator model, the canonical ETH stack (SLQ + QP-WBC)
   wins outright** (0.65 m/s, 15/20 pushes, no envelope contact). X2's earlier tie was an
   ideal-actuator artifact: its gait exploited swing speeds at which the real motor cannot
   deliver torque.
2. The high level still gates survival (same low level: ETH 0.65 vs X1 0.15; X2 0.40 vs
   MIT 0.00); the low level decides whether the actuator envelope is even entered —
   the WBC's acceleration-bounded swing tasks keep joints in the flat region, the
   Cartesian-PD J^T swing does not.
3. **DOG4.5 is hereby relabelled a damping-sensitivity arm**, not a fidelity model: +6.6×
   joint damping flips the ranking entirely (X1 0.80, perfect tracking, CoT 0.40). Together
   with 4.6 it brackets the real robot: the true joint-side mechanical damping (gearbox) is
   unmeasured, sits somewhere between 0.01 and ~0.07 N·m·s/rad, and the ranking is
   hypersensitive to it. **Identifying joint damping on hardware is the single highest-value
   experiment before any controller conclusion is transferred to the real robot.**

## 6. Files
- `scene/actuator_envelope.py` (callback + constants kt=0.1, R=0.466 Ω, N=10)
- `scene/probe_joint_speeds.py` (speed-vs-corner probe)
- raw: `s1_s2_metrics.csv`, `s5_push.csv`, `solve_times.csv`, `comparison_summary.md`
- siblings: `..\..\DOG4.0_description\results\` (ideal), `..\..\DOG4.5_description\results\` (damped)
