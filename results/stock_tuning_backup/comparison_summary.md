# Stack comparison summary (S1/S2/S5)

Same robot, shared modules, ground-truth state; MuJoCo 3.9.0, dt=0.002. Cross-combinations X1/X2 answer RQ4 (attribution).

## RQ1 — tracking (S1 staircase): vel RMSE [m/s] / att RMSE [deg] / CoT

| vx cmd | MIT (convexMPC + leg ctrl) | ETH (SLQ-MPC + QP-WBC) | X1 (convexMPC + QP-WBC) | X2 (SLQ-MPC + leg ctrl) |
|---|---|---|---|---|
| 0.1 | 0.089 / 2.7 / 4.80 | 0.066 / 2.4 / 2.77 | 0.064 / 4.0 / 2.33 | 0.075 / 5.4 / 5.12 |
| 0.2 | fell (0.08) | 0.086 / 12.9 / 2.27 | fell (0.15) | fell (0.13) |
| 0.3 | — | 0.235 / 3.5 / 2.05 | — | — |
| 0.4 | — | 0.326 / 4.4 / 1.84 | — | — |
| 0.5 | — | 0.409 / 5.9 / 1.70 | — | — |

## RQ1 — max stable speed (S2, ramp to failure)

- **MIT (convexMPC + leg ctrl)**: 0.00 m/s (height collapse)
- **ETH (SLQ-MPC + QP-WBC)**: 0.65 m/s (roll-over)
- **X1 (convexMPC + QP-WBC)**: 0.15 m/s (roll-over)
- **X2 (SLQ-MPC + leg ctrl)**: 0.40 m/s (height collapse)

## RQ2 — push recovery (S5; Y per magnitude [10, 20, 30, 40, 50] N for 0.1 s)

| dir | MIT (convexMPC + leg ctrl) | ETH (SLQ-MPC + QP-WBC) | X1 (convexMPC + QP-WBC) | X2 (SLQ-MPC + leg ctrl) |
|---|---|---|---|---|
| +x | Y Y · · · | Y Y Y Y Y | Y · Y Y Y | Y Y · Y Y |
| -x | · Y Y · · | Y · · Y Y | Y Y Y Y Y | Y Y Y · · |
| +y | · · Y · · | Y Y · · · | Y Y · Y · | Y Y Y · Y |
| -y | · · · · · | Y Y Y Y Y | Y Y Y Y Y | · Y Y Y · |

## RQ3 — computation (caveat: OSQP is compiled C; SLQ is numpy)

- **MIT (convexMPC + leg ctrl)**: solve mean 4.20 ms, p95 7.83 ms
- **ETH (SLQ-MPC + QP-WBC)**: solve mean 24.33 ms, p95 25.33 ms
- **X1 (convexMPC + QP-WBC)**: solve mean 4.38 ms, p95 9.07 ms
- **X2 (SLQ-MPC + leg ctrl)**: solve mean 24.67 ms, p95 25.94 ms
