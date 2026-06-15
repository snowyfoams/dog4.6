# DOG3.0 — two locomotion stacks for the dissertation comparison (MuJoCo)

Keyboard-drivable quadruped locomotion for the DOG3.0 model, with BOTH
model-based stacks the dissertation compares, behind one interface and one set
of shared modules:

- **MIT lineage** (`srb_mpc/controller.py`): convex MPC — one-shot QP on the
  yaw-linearised SRB (Di Carlo et al., 2018) — + J^T leg controller.
- **ETH lineage** (`srb_mpc/eth_controller.py`): SLQ-MPC — iterated dynamic
  programming (iLQR/DDP) on the NONLINEAR SRB, affine time-varying policy whose
  feedback gains survive between replans — + QP whole-body controller
  (full-dynamics, friction-cone and torque limits inside the QP).

Shared across both (the fairness backbone): trot gait clock, contact-aware
early-touchdown layer, Raibert/Bezier swing trajectories, reference generation
and station keeping, ground-truth state, same horizon/dt/weights at the MPC
level, same torque limits.

## Run it

```
D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG3.0_description\run_dog.py        # MIT stack
D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG3.0_description\run_dog.py --eth  # ETH stack
```

Controls (focus the viewer window):

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| W / S | forward / backward | A / D | turn left / right |
| Q / E | strafe left / right | SPACE | stop & stand |
| R | reset to standing pose | | |

The robot stands still on a 4-foot balancing MPC until you give a command, then trots.
It auto-resets if it tips over. Commands step in small increments (tap to accelerate).

## How it's built

1. **`build_scene.py`** turns the converted `DOG3.0.xml` into a simulation scene
   `scene/DOG3.0_scene.xml`: floating `trunk` (freejoint), base inertia re-attached,
   foot contact spheres + sites at each toe, ground/lights, 12 torque motors, light
   joint damping/armature, and a symmetric standing pose solved by IK and **baked in
   as the model default** — the scene loads standing (qpos0), joint zeros are the
   stand angles, and the `stand` keyframe equals the default configuration.
   With `MIRROR_RIGHT_LEGS = True` the right legs (FR/RR) are **re-mounted as true
   mirror copies of the left legs** (mirrored body frames + STLs reflected via
   negative y mesh scale), because the CAD assembly mounts the leg modules as
   rotational "propeller" copies — remove the switch once the Fusion assembly is
   fixed and re-exported.
2. **`srb_mpc/`** is the controller:
   - `robot.py` — MuJoCo state interface (trunk-CoM state, foot Jacobians via
     `mj_jacSite`, GRF→torque, gravity feed-forward `G(q)`).
   - `gait.py` — trot scheduler (diagonal pairs, duty-0.6 double-support); the
     schedule itself is fixed/clock-driven and stays unchanged.
   - `contact_gait.py` — sim2real layer over the fixed schedule: measured-contact
     estimator (`MujocoFootContact`; swap for foot sensors on hardware),
     early-touchdown promotion (`ContactAwareGait`), and the low-pass placement
     velocity (`LowPassVelocity`, time-constant parametrised).
   - `convex_mpc.py` — 13-state SRB QP over a 10-step / 0.3 s horizon, friction
     pyramid + swing zero-force, solved with OSQP.
   - `swing.py` — Raibert footstep target + Bezier swing arc; `targets()` is the
     SHARED trajectory generator both stacks consume; `torque()` is the MIT
     Cartesian-PD `J^T F` tracking of it.
   - `controller.py` — MIT stack: `tau = G(q) - J^T f_stance + J^T F_swing`; holds
     heading (`yaw_des`) and station (`p_des` + bounded velocity correction fed to
     the MPC reference and the Raibert targets), so a zero command trots in place
     without drifting.
   - `slq_mpc.py` — ETH high level: discrete iLQR/DDP on the nonlinear SRB
     (true R(rpy), Euler-rate map, gyroscopic term), same N=10/dt=0.03/weights as
     the convex MPC, friction cone via smooth penalty (hard clamps inside the
     rollout stall the iteration — the linearisation can't see them), 2
     warm-started real-time iterations per tick.
   - `qp_wbc.py` — ETH low level: stance/base whole-body QP over (qdd, f) with
     floating-base dynamics, tangential-Baumgarte contact task, friction cone and
     torque limits; swing legs get operational-space PD torque overlays
     (WBIC style — swing tasks inside the QP trade against base acceleration).
     dof order ≠ actuator order in this model: torques are permuted explicitly.
   - `eth_controller.py` — ETH stack glue, same interface as `controller.py`.
3. **`run_dog.py`** — the interactive viewer + keyboard interface.

## Verify (headless)

```
python build_scene.py                # rebuild scene + standing render
python tests/test_balance.py         # MPC balances on 4 feet (RESULT: BALANCE OK)
python tests/test_trot_in_place.py   # trot with zero command holds station (no drift)
python tests/test_eth_stack.py       # ETH stack: stand / in-place / forward smoke test
python tests/demo_drive.py           # scripted drive, saves scene/DOG3.0_drive.gif
python -u tests/compare_stacks.py run <stack> <scenario>
                                     # dissertation Ch.6 benchmark slices, resumable:
                                     # stack in {mit, eth, cvxwbc, slqjt} (cvxwbc/slqjt
                                     # are the RQ4 cross-combinations), scenario in
                                     # {s1s2, s5}; rows appended to results/*.csv
python tests/compare_stacks.py report
                                     # -> results/comparison_summary.md + PNGs
python tests/test_jump.py [stack]    # fixed-place jump (pronk) sweep, all stacks:
                                     # crouch->thrust->flight->land; metrics ->
                                     # results/jump_*.csv|md, gifs -> scene/DOG3.0_jump_*.gif
```

The jump is a scripted maneuver (`srb_mpc/jump.py`) that duck-types the gait
interface — the fixed trot schedule is untouched; the controller swaps it in for
one jump (interactive: press `J` in run_dog.py) and both MPC high levels plan
through the all-stance -> flight -> all-stance schedule via the shared
`z_ref_fn` height-reference hook.

`results/fairness_protocol.md` is the comparison's tuning receipt: shared-module
inventory, every final gain/weight of all four stacks, and the openly-stated
asymmetries (Appendix E material for the dissertation).

## Notes on this model (non-obvious)

- The CAD export frame is rotated ~120° about X and offset ~1.4 m from the origin; a
  corrective trunk quaternion (roll fix + 180° yaw so the F legs face +x) levels it,
  and MuJoCo's numeric Jacobians sidestep the tilted-axis kinematics. The roll sign
  matters: the wrong sign mounts the dog **belly-up** and the standing IK then winds
  the unlimited joints ~180° into a crossed-leg crab pose (`build_scene.py` now
  asserts feet-below-hips and bounded stand angles, and bakes per-joint ranges).
- The CAD mounts the four leg modules as ROTATIONAL copies ("propeller"), not
  mirror copies: posed with symmetric feet, the original right legs tilt their leg
  planes ~38° with the thigh attach swung inboard, and the composite CoM ends up
  ~3 cm off the centerline. The scene therefore re-mounts the right legs mirrored
  (see above); with that, the lateral CoM offset vanishes (~0.0 mm).
- The legs are ~70% of total mass (3.86 kg legs vs 1.71 kg trunk). Two consequences:
  the SRB point is a **trunk-fixed point at the standing composite CoM** (a biased
  SRB point skews every MPC moment and rolls the trot over; the live composite CoM
  jerks when legs swing — the body-fixed standing CoM is both unbiased and smooth),
  and **gravity compensation `G(q)` is essential** (without it the heavy legs sag
  and the MPC forces are never realised). The trot's nominal footholds are also
  centred laterally on the SRB point (`foot_nominal` in `robot.py`) — a no-op for
  the mirrored robot, but it keeps the gait honest if mass asymmetry returns.
- A purely clock-driven trot is unstable over time: body oscillation makes swing
  feet touch down EARLY, the swing controller keeps pushing them along the arc into
  the floor while the MPC models them as force-free, and the impulse train pumps a
  gait-frequency rock (~31° within 12 s in-place). `contact_gait.py` fixes both
  halves: measured early touchdowns are promoted to stance (with an immediate MPC
  replan), and the Raibert placement uses a ~100 ms low-passed CoM velocity instead
  of the gait-cyclic instantaneous one. Result: 30 s in-place trot at 0.3° tilt /
  2 mm drift; forward trot tracks 0.4 m/s with ~4° tilt.
- The stance is narrow and the robot small, so it is balance-sensitive. It is tuned
  for smooth/ramped commands (as from key taps); slamming full-speed step commands can
  still tip it. Keep speeds moderate (defaults cap ~0.35 m/s).
