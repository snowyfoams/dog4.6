# DOG4.6 — definitive results, with the tuning-artifact question resolved

DOG4.6 is the physically correct platform (current-controlled servo envelope; see
`actuator_model_correction.md`). After a reviewer-prompted check, the shared gait was
**retuned for this robot**: a single shared parameter, `TrotGait` period 0.34 → **0.42 s**
(`srb_mpc/gait.py`), applied identically to all four stacks — high- and low-level
algorithms byte-unchanged. Reason below. Stock (DOG3.0-inherited) results are preserved in
`results/stock_tuning_backup/`; the head-to-head is `paper_figs/fig_tuning_compare.png`.

## Why retune: the "only ETH survives" picture was largely a tuning artifact

The stock tuning came from DOG3.0 (5.57 kg, 18 N·m) and was never adapted to the new
7.334 kg / 7 N·m robot. Diagnosis at a *held* 0.2 m/s showed the failure is lateral
**rock-over**, not torque starvation: X2 survives 6 s held at 0.2 — its staircase fall was
triggered only by the 0.1→0.2 command jump. Slowing the gait (longer stance) is the single
knob that addresses rock-over on a heavy robot. Effect of 0.34 → 0.42 s:

| stack | stock | retuned |
|---|---|---|
| ETH | survives staircase but **station-keeps** (mean≈0.08 at cmd 0.5, RMSE 0.41, att 12.9°) | **tracks perfectly** (mean = cmd 0.1–0.5, att < 0.8°, CoT 0.43) |
| X1  | **falls at 0.2** | **survives the whole staircase to 0.5** |
| MIT | falls at 0.2 | falls at 0.3 |
| X2  | falls at 0.2 | falls at 0.3 |

So the figure the reader flagged — "only ETH has data" — was substantially a stock-tuning
artifact. At platform tuning, ETH **and** X1 produce full curves and ETH's tracking goes
from station-keeping to riding the ideal line (`fig_tuning_compare.png`, bottom-left).

## Measured outcomes at platform tuning (the result of record)

| stack | high/low | S2 v_max | tracks to | push /20 | best CoT | solve |
|---|---|---|---|---|---|---|
| MIT | convex/leg | 0.00 (RO) | 0.2 | 8 | 3.20 | 3.8 ms |
| **ETH** | **SLQ/WBC** | **0.60 (HC)** | **0.5 (mean=cmd)** | **18** | **0.43** | 23.3 ms |
| X1 | convex/WBC | 0.20 (RO) | 0.2 | **18** | 0.99 | 4.2 ms |
| X2 | SLQ/leg | 0.10 (HC) | 0.2 | 11 | 1.72 | 23.7 ms |

Retuning improved tracking, push and CoT across the board (ETH push 15→18, CoT 1.70→0.43;
X1 fell@0.2 → full staircase). The headline therefore **strengthens**: the canonical ETH
stack (SLQ + WBC) is now the unambiguous best — only stack that genuinely tracks to 0.5,
highest push count, 4× better economy — not merely the last one standing.

## What is robust vs what is a tuning/command-profile artifact

The platform is near the stability boundary for every stack, so absolute fall speeds are
**not robust** — they swing with both gait and command profile:
- X2: staircase improves (0.1→0.2-tracking) but the fine S2 ramp **drops 0.40→0.10**.
- the WBC stacks track beautifully under the benchmark's gentle ramp yet tip over under a
  hard stand→0.3 step (smoke test) at the slow gait.
- the sweep found **conflicting per-stack optima** (MIT prefers the fast 0.34 gait, X2 the
  slow 0.42) — no single shared tuning is simultaneously optimal.

Treat as **tuning-sensitive (report with caveat)**: the exact v_max numbers and any
"cross ties canonical" claim. Treat as **robust (tuning-insensitive)**:
1. **ETH (SLQ+WBC) is best** — best at both stock and retuned, the only stack that tracks.
2. **MIT (convex+leg) is worst** — S2 = 0.00 under every tuning tried.
3. **Robustness is owned by the WBC low level**: WBC stacks (ETH 18, X1 18) reject pushes
   and survive far better than leg stacks (MIT 8, X2 11), at fixed high level.
4. **The leg-controller stacks are command-profile-fragile** (their numbers move most with
   gait/profile); the WBC's acceleration-bounded swing tasks are what stabilise them.
5. **Compute is owned by the high level** (convex ≈ 4 ms vs SLQ ≈ 23 ms).

## Answer to "is the early fall a tuning artifact?"

**Yes, predominantly.** The early falls and the "only ETH" figure are artifacts of
DOG3.0-inherited tuning plus the staircase's command jumps — one shared gait change rescues
X1 entirely and turns ETH's survival into genuine tracking. What survives retuning is the
*ranking structure*, not the absolute speeds: ETH best, MIT worst, WBC = robustness. The
absolute v_max figures should be reported as boundary-sensitive.

## Implications for the dissertation
- Report at **platform-appropriate tuning**, not DOG3.0-inherited; document the one shared
  change (gait 0.42) in the fairness appendix — high/low-level internals untouched, so the
  comparison stays fair.
- State the **command-profile / tuning sensitivity** as an explicit threat to validity; the
  robust claims are the ranking structure and the layer attribution, not the v_max numbers.
- Highest-value next steps: (i) a small per-platform shared-tuning pass logged in the
  fairness receipt; (ii) hardware identification of real joint/gearbox damping (the 4.5 arm
  showed the ranking is hypersensitive to it).

## Files
- retuned raw: `s1_s2_metrics.csv`, `s5_push.csv`, `solve_times.csv`, `comparison_summary.md`
- stock (DOG3.0-tuning) twin: `stock_tuning_backup/`
- figures: `paper_figs/fig_tuning_compare.{png,pdf}` (stock vs retuned, × = fell),
  `fig_tracking`, `fig_robustness`, `fig_compute`, `table.tex`
- tooling: `scene/diagnose_fall.py`, `scene/sweep_retune.py`
