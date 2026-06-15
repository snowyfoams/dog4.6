# Stack comparison summary (S1/S2/S5)

Same robot, shared modules, ground-truth state; MuJoCo 3.9.0, dt=0.002. Cross-combinations X1/X2 answer RQ4 (attribution).

## RQ1 — tracking (S1 staircase): vel RMSE [m/s] / att RMSE [deg] / CoT

| vx cmd | MIT (convexMPC + leg ctrl) | ETH (SLQ-MPC + QP-WBC) | X1 (convexMPC + QP-WBC) | X2 (SLQ-MPC + leg ctrl) |
|---|---|---|---|---|
| 0.1 | 0.116 / 4.3 / 5.94 | 0.034 / 0.8 / 1.57 | 0.065 / 2.2 / 2.86 | 0.037 / 1.7 / 2.52 |
| 0.2 | 0.126 / 4.7 / 3.20 | 0.029 / 0.5 / 0.69 | 0.072 / 4.5 / 0.99 | 0.089 / 3.3 / 1.72 |
| 0.3 | fell (0.44) | 0.034 / 0.5 / 0.50 | 0.125 / 6.9 / 1.16 | fell (0.23) |
| 0.4 | — | 0.042 / 0.6 / 0.46 | 0.193 / 7.6 / 1.11 | — |
| 0.5 | — | 0.045 / 0.6 / 0.43 | 0.243 / 7.1 / 1.13 | — |

## RQ1 — max stable speed (S2, ramp to failure)

- **MIT (convexMPC + leg ctrl)**: 0.00 m/s (roll-over)
- **ETH (SLQ-MPC + QP-WBC)**: 0.60 m/s (height collapse)
- **X1 (convexMPC + QP-WBC)**: 0.20 m/s (roll-over)
- **X2 (SLQ-MPC + leg ctrl)**: 0.10 m/s (height collapse)

## RQ2 — push recovery (S5; Y per magnitude [10, 20, 30, 40, 50] N for 0.1 s)

| dir | MIT (convexMPC + leg ctrl) | ETH (SLQ-MPC + QP-WBC) | X1 (convexMPC + QP-WBC) | X2 (SLQ-MPC + leg ctrl) |
|---|---|---|---|---|
| +x | · · · Y Y | Y Y Y Y Y | Y Y Y Y Y | · · · · Y |
| -x | Y · · · · | Y Y Y Y Y | Y Y Y Y Y | Y · Y · Y |
| +y | · · Y · Y | Y Y Y · · | Y Y Y Y Y | · Y Y Y Y |
| -y | Y Y · Y · | Y Y Y Y Y | Y Y Y · · | Y Y Y · · |

## RQ3 — computation (caveat: OSQP is compiled C; SLQ is numpy)

- **MIT (convexMPC + leg ctrl)**: solve mean 3.75 ms, p95 5.43 ms
- **ETH (SLQ-MPC + QP-WBC)**: solve mean 23.28 ms, p95 24.12 ms
- **X1 (convexMPC + QP-WBC)**: solve mean 4.20 ms, p95 9.89 ms
- **X2 (SLQ-MPC + leg ctrl)**: solve mean 23.70 ms, p95 24.86 ms
