r"""Scripted 'keyboard drive' demo: stand -> trot in place -> forward -> turn -> stop.
Exercises the same command transitions as run_dog.py and renders a follow-cam gif.

  D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG3.0_description\tests\demo_drive.py
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srb_mpc.controller import LocomotionController

SCENE = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_scene.xml"
OUT = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_drive.gif"

M = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(M)
ctrl = LocomotionController(M, stand=True)
ctrl.reset(d)
rob = ctrl.rob
dt = M.opt.timestep


# (t_start, stand, vx, vy, wz, label)
SCHEDULE = [
    (0.0, True,  0.0, 0.0, 0.0, "STAND"),
    (1.0, False, 0.0, 0.0, 0.0, "TROT in place"),
    (2.5, False, 0.25, 0.0, 0.0, "FORWARD"),
    (5.5, False, 0.0, 0.0, 0.5, "TURN left"),
    (7.5, False, 0.20, 0.0, 0.0, "FORWARD"),
    (9.5, True,  0.0, 0.0, 0.0, "STAND"),
]
T_END = 10.5


def cmd_at(t):
    cur = SCHEDULE[0]
    for s in SCHEDULE:
        if t >= s[0]:
            cur = s
    return cur


frames, label_now = [], ""
cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(M, cam)
cam.distance = cam.distance * 0.55; cam.elevation = -18; cam.azimuth = 120
prev_vx = 0.0
resets, max_tilt, min_h = 0, 0.0, 1.0
with mujoco.Renderer(M, 480, 720) as r:
    for k in range(int(T_END / dt)):
        t = k * dt
        _ts, stand, vx, vy, wz, label = cmd_at(t)
        if label != label_now:
            print(f"t={t:4.1f}s  -> {label}"); label_now = label
        # gentle command ramp so transitions are smooth (like tapping keys)
        prev_vx += np.clip(vx - prev_vx, -0.5 * dt, 0.5 * dt)
        ctrl.gait.stand = stand
        ctrl.cmd.vx, ctrl.cmd.vy, ctrl.cmd.wz = prev_vx, vy, wz
        d.ctrl[:] = ctrl(d)
        mujoco.mj_step(M, d)
        max_tilt = max(max_tilt, np.degrees(np.abs(rob.rpy(d)[:2])).max())
        min_h = min(min_h, rob.com(d)[2])
        if not np.isfinite(d.qpos).all() or rob.com(d)[2] < 0.07:
            print(f"  fell at t={t:.2f}; resetting"); ctrl.reset(d); prev_vx = 0.0; resets += 1
        if k % 10 == 0:
            cam.lookat[:] = rob.com(d)
            r.update_scene(d, camera=cam); frames.append(r.render())

print(f"\nstability over {T_END}s drive: resets={resets}  max tilt={max_tilt:.1f} deg  min height={min_h:.3f}")
print("RESULT:", "DROVE CLEAN (no falls)" if resets == 0 else f"{resets} reset(s)")
imageio.mimsave(OUT, frames, fps=50)
print("gif ->", OUT, f"({len(frames)} frames)")
