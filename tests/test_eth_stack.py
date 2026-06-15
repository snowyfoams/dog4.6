r"""Smoke test for the ETH stack (SLQ-MPC + QP-WBC): stand hold, trot in place,
forward trot — same pass criteria spirit as the MIT-stack tests. Gates the
comparison benchmark.

  D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG3.0_description\tests\test_eth_stack.py
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srb_mpc.eth_controller import LocomotionControllerETH

SCENE = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_scene.xml"
OUT = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_eth_trot.gif"

M = mujoco.MjModel.from_xml_path(SCENE)


def run(tag, plan, T, record=False):
    d = mujoco.MjData(M)
    ctrl = LocomotionControllerETH(M, stand=True)
    ctrl.reset(d)
    rob = ctrl.rob
    dt = M.opt.timestep
    frames = []
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(M, cam)
    cam.azimuth, cam.elevation, cam.distance = 120, -15, cam.distance * 0.55
    p0 = rob.com(d).copy()
    maxtilt, maxdrift, minz = 0.0, 0.0, 1.0
    r = mujoco.Renderer(M, 480, 640) if record else None
    for k in range(int(T / dt)):
        t = k * dt
        for tend, vx, vy, wz, stand in plan:
            if t < tend:
                ctrl.cmd.vx, ctrl.cmd.vy, ctrl.cmd.wz = vx, vy, wz
                ctrl.gait.stand = stand
                break
        d.ctrl[:] = ctrl(d)
        mujoco.mj_step(M, d)
        tilt = np.degrees(np.abs(rob.rpy(d)[:2])).max()
        maxtilt = max(maxtilt, tilt)
        maxdrift = max(maxdrift, float(np.linalg.norm(rob.com(d)[:2] - p0[:2])))
        minz = min(minz, rob.com(d)[2])
        if not np.isfinite(d.qpos).all() or tilt > 50 or rob.com(d)[2] < 0.07:
            print(f"  {tag}: FELL at t={t:.2f}s (tilt={tilt:.1f} z={rob.com(d)[2]:.3f})")
            if r: r.close()
            return None
        if record and k % 25 == 0:
            cam.lookat[:] = rob.com(d)
            r.update_scene(d, camera=cam)
            frames.append(r.render())
    if r:
        r.close()
    st = np.array(ctrl.slq.solve_times) * 1e3
    print(f"  {tag}: OK  max tilt={maxtilt:.1f} deg  max drift={maxdrift:.3f} m  "
          f"min z={minz:.3f}  SLQ {st.mean():.1f}ms mean / {np.percentile(st,95):.1f}ms p95  "
          f"WBC fallbacks={ctrl.wbc_fallbacks}/{ctrl.wbc.solve_count}")
    return dict(tilt=maxtilt, drift=maxdrift, frames=frames, com=rob.com(d).copy(), p0=p0)


print("=== ETH stack (SLQ-MPC + QP-WBC) smoke test ===")
ok = True

r1 = run("stand 3s          ", [(3.0, 0, 0, 0, True)], 3.0)
ok &= r1 is not None and r1["tilt"] < 2.0 and r1["drift"] < 0.03

r2 = run("trot in place 8s  ", [(1.0, 0, 0, 0, True), (9.0, 0, 0, 0, False)], 9.0)
ok &= r2 is not None and r2["tilt"] < 15.0 and r2["drift"] < 0.10

r3 = run("forward 0.4 m/s 4s", [(1.0, 0, 0, 0, True), (5.0, 0.4, 0, 0, False)], 5.0,
         record=True)
ok &= r3 is not None
if r3 is not None:
    dx = r3["com"][0] - r3["p0"][0]
    print(f"  forward distance: {dx:+.2f} m  (mean {dx/4.0:+.2f} m/s of 0.40 cmd)")
    ok &= dx > 0.8
    imageio.mimsave(OUT, r3["frames"], fps=20)
    print("  gif ->", OUT)

print("RESULT:", "ETH STACK OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
