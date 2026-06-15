"""Milestone 2: SRB convex-MPC balances the floating robot on all four feet.

  D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG3.0_description\tests\test_balance.py
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srb_mpc.controller import LocomotionController

SCENE = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_scene.xml"
OUT = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_balance.jpg"

M = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(M)
ctrl = LocomotionController(M, stand=True)
ctrl.reset(d)

z0 = ctrl.rob.com(d)[2]
print(f"nominal height={ctrl.rob.nominal_h:.4f}  start CoM z={z0:.4f}")

T = 3.0
nstep = int(T / M.opt.timestep)
hz, tilt = [], []
for k in range(nstep):
    d.ctrl[:] = ctrl(d)
    mujoco.mj_step(M, d)
    hz.append(ctrl.rob.com(d)[2])
    tilt.append(np.degrees(np.abs(ctrl.rob.rpy(d)[:2])).max())
    if not np.isfinite(d.qpos).all():
        print(f"DIVERGED at step {k} (t={k*M.opt.timestep:.3f})"); break

hz = np.array(hz); tilt = np.array(tilt)
print(f"CoM z: start={z0:.4f} end={hz[-1]:.4f} min={hz.min():.4f} max={hz.max():.4f}")
print(f"max tilt over run = {tilt.max():.2f} deg ; final tilt = {tilt[-1]:.2f} deg")
print(f"mean MPC solve time = {1e3*np.mean(ctrl.solve_times):.2f} ms")
ok = np.isfinite(hz[-1]) and abs(hz[-1] - ctrl.rob.nominal_h) < 0.05 and tilt[-1] < 12
print("RESULT:", "BALANCE OK" if ok else "FAILED")

cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(M, cam)
cam.azimuth, cam.elevation, cam.distance = 130, -20, cam.distance * 0.55
with mujoco.Renderer(M, 720, 960) as r:
    r.update_scene(d, camera=cam)
    imageio.imwrite(OUT, r.render(), quality=95)
print("render ->", OUT)
