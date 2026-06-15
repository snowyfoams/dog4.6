r"""Fixed-place jump (pronk) test: crouch -> thrust -> flight -> land, for all
four stacks, sweeping the scheduled flight time. Metrics per trial: liftoff,
measured flight time, apex CoM gain, landing drift, peak tilt, success
(upright and back near standing height 1.5 s after landing). Saves a gif per
stack at the middle intensity — the simulation evidence.

  D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG3.0_description\tests\test_jump.py [stack]
"""
import os
import sys
import csv
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srb_mpc.controller import LocomotionController
from srb_mpc.eth_controller import LocomotionControllerETH
from srb_mpc.contact_gait import MujocoFootContact
from srb_mpc import jump

SCENE = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_scene.xml"
RESULTS = r"D:\mujoco\DOG3.0_description\results"
GIFDIR = r"D:\mujoco\DOG3.0_description\scene"
os.makedirs(RESULTS, exist_ok=True)

M = mujoco.MjModel.from_xml_path(SCENE)

STACKS = {
    "mit": ("MIT (convexMPC + leg ctrl)", lambda: LocomotionController(M, stand=True)),
    "eth": ("ETH (SLQ-MPC + QP-WBC)", lambda: LocomotionControllerETH(M, stand=True)),
    "cvxwbc": ("X1 (convexMPC + QP-WBC)",
               lambda: LocomotionControllerETH(M, stand=True,
                                               high_level="convex", low_level="wbc")),
    "slqjt": ("X2 (SLQ-MPC + leg ctrl)",
              lambda: LocomotionControllerETH(M, stand=True,
                                              high_level="slq", low_level="jt")),
}
if len(sys.argv) > 1:
    STACKS = {k: v for k, v in STACKS.items() if k == sys.argv[1]}

FLIGHTS = [0.20, 0.30, 0.40]
GIF_FLIGHT = 0.30


def trial(make, flight_t, record=False):
    d = mujoco.MjData(M)
    ctrl = make()
    ctrl.reset(d)
    rob = ctrl.rob
    touch = MujocoFootContact(M)
    dt = M.opt.timestep
    frames = []
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(M, cam)
    cam.azimuth, cam.elevation, cam.distance = 105, -12, 0.95
    r = mujoco.Renderer(M, 480, 640) if record else None

    jm = None
    p_trig = None
    apex = -np.inf
    air_time = 0.0
    max_tilt = 0.0
    landed_at = None
    fail = None
    T = 1.0 + 1.5 + 2.5            # stand, headroom for jump, settle window
    for k in range(int(T / dt)):
        t = k * dt
        if jm is None and t >= 1.0:
            jm = jump.install(ctrl, flight_t=flight_t)
            p_trig = rob.com(d).copy()
        if jm is not None and ctrl.z_ref_fn is not None and jm.done(ctrl.t):
            jump.restore(ctrl, stand=True)
        d.ctrl[:] = ctrl(d)
        mujoco.mj_step(M, d)
        com = rob.com(d)
        tilt = np.degrees(np.abs(rob.rpy(d)[:2])).max()
        if jm is not None:
            apex = max(apex, com[2])
            max_tilt = max(max_tilt, tilt)
            if not touch.measure(d).any():
                air_time += dt
                landed_at = None
            elif air_time > 0 and landed_at is None:
                landed_at = t
            if not np.isfinite(d.qpos).all() or tilt > 45 or com[2] < 0.055:
                fail = f"fell at t={t:.2f} (tilt={tilt:.0f}, z={com[2]:.3f})"
                break
        if record and r is not None and k % 10 == 0:
            cam.lookat[:] = [p_trig[0] if p_trig is not None else com[0],
                             com[1], 0.15]
            r.update_scene(d, camera=cam)
            frames.append(r.render())
    if r is not None:
        r.close()
    com = rob.com(d)
    drift = float(np.linalg.norm(com[:2] - p_trig[:2])) if p_trig is not None else np.nan
    # "recovered upright": stable and level near standing height (landing foot
    # placement can leave the stand a few cm high; renormalises on the next step)
    settled = (fail is None and abs(com[2] - rob.nominal_h) < 0.05
               and np.degrees(np.abs(rob.rpy(d)[:2])).max() < 8)
    return dict(flight_cmd=flight_t,
                liftoff=int(air_time > 0.04),
                air_time=round(air_time, 3),
                apex_gain=round(float(apex - rob.nominal_h), 3),
                drift=round(drift, 3),
                max_tilt=round(max_tilt, 1),
                success=int(settled),
                note=fail or ""), frames


rows = []
for key, (name, make) in STACKS.items():
    print(f"\n=== {name} ===")
    for ft in FLIGHTS:
        res, frames = trial(make, ft, record=(abs(ft - GIF_FLIGHT) < 1e-9))
        res["stack"] = name
        rows.append(res)
        print(f"  flight={ft:.2f}s: liftoff={bool(res['liftoff'])} "
              f"air={res['air_time']:.2f}s apex=+{res['apex_gain']*100:.0f}cm "
              f"drift={res['drift']*100:.0f}cm tilt={res['max_tilt']:.0f}deg "
              f"success={bool(res['success'])} {res['note']}")
        if frames:
            out = os.path.join(GIFDIR, f"DOG3.0_jump_{key}.gif")
            imageio.mimsave(out, frames, fps=50)
            print(f"  gif -> {out}")

keys = ["stack", "flight_cmd", "liftoff", "air_time", "apex_gain", "drift",
        "max_tilt", "success", "note"]
with open(os.path.join(RESULTS, "jump_metrics.csv"), "w", newline="") as fp:
    wr = csv.DictWriter(fp, fieldnames=keys)
    wr.writeheader()
    for r_ in rows:
        wr.writerow({k: r_.get(k, "") for k in keys})

with open(os.path.join(RESULTS, "jump_summary.md"), "w", encoding="utf-8") as fp:
    fp.write("# Fixed-place jump (pronk): crouch 0.30 s / thrust 0.12 s / "
             "scheduled flight sweep / land 0.40 s\n\n")
    fp.write("| stack | flight cmd [s] | liftoff | air time [s] | apex gain [cm] "
             "| drift [cm] | peak tilt [deg] | success |\n")
    fp.write("|---|---|---|---|---|---|---|---|\n")
    for r_ in rows:
        fp.write(f"| {r_['stack']} | {r_['flight_cmd']:.2f} | "
                 f"{'Y' if r_['liftoff'] else 'n'} | {r_['air_time']:.2f} | "
                 f"{r_['apex_gain']*100:+.0f} | {r_['drift']*100:.0f} | "
                 f"{r_['max_tilt']:.0f} | {'Y' if r_['success'] else 'n'} "
                 f"{r_['note']} |\n")
print("\nresults -> jump_metrics.csv, jump_summary.md")
