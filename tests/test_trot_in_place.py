r"""Trot in place at a fixed point: stand 1 s, then run the trot gait with a ZERO
velocity command for 30 s and check the robot stays where it is — bounded position
drift, bounded heading drift, no fall. The long horizon matters: a purely
clock-driven gait pumps itself over after ~10 s (early-touchdown impulses); this
test guards the contact-aware layer in srb_mpc/contact_gait.py. Saves a gif.

  D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG3.0_description\tests\test_trot_in_place.py
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srb_mpc.controller import LocomotionController

SCENE = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_scene.xml"
OUT = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_trot_in_place.gif"

T_STAND, T_TROT = 1.0, 30.0
DRIFT_MAX = 0.05          # m   allowed CoM excursion from the start point
YAW_MAX = 15.0            # deg allowed heading drift
TILT_MAX = 40.0           # deg fall threshold
FRAME_EVERY = 25          # sim steps per gif frame (keeps the 30 s gif small)

M = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(M)
ctrl = LocomotionController(M, stand=True)
ctrl.reset(d)
rob = ctrl.rob
dt = M.opt.timestep

frames = []
cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(M, cam)
cam.azimuth, cam.elevation, cam.distance = 120, -15, cam.distance * 0.55
cam.lookat[:] = rob.com(d)

p0 = None                                 # anchor point, latched at trot start
drift, tilt, hz, yawerr = [], [], [], []
fell = False
with mujoco.Renderer(M, 480, 640) as r:
    for k in range(int((T_STAND + T_TROT) / dt)):
        t = k * dt
        if t >= T_STAND and ctrl.gait.stand:
            ctrl.gait.stand = False       # start trotting, command stays zero
            p0 = rob.com(d).copy()
            yaw0 = rob.yaw(d)
        d.ctrl[:] = ctrl(d)
        mujoco.mj_step(M, d)
        if p0 is not None:
            drift.append(float(np.linalg.norm(rob.com(d)[:2] - p0[:2])))
            yawerr.append(float(np.degrees(rob.yaw(d) - yaw0)))
        tilt.append(np.degrees(np.abs(rob.rpy(d)[:2])).max())
        hz.append(rob.com(d)[2])
        if not np.isfinite(d.qpos).all() or tilt[-1] > TILT_MAX or hz[-1] < 0.07:
            print(f"FELL at t={t:.2f}s (tilt={tilt[-1]:.1f} deg, z={hz[-1]:.3f})")
            fell = True
            break
        if k % FRAME_EVERY == 0:
            r.update_scene(d, camera=cam)
            frames.append(r.render())

drift = np.array(drift); tilt = np.array(tilt); hz = np.array(hz); yawerr = np.array(yawerr)
print(f"trot in place for {T_TROT}s (zero command):")
print(f"  CoM drift   : final={drift[-1]:.3f} m  max={drift.max():.3f} m  (limit {DRIFT_MAX})")
print(f"  yaw drift   : final={yawerr[-1]:+.1f} deg  max|.|={np.abs(yawerr).max():.1f} deg  (limit {YAW_MAX})")
print(f"  tilt        : max={tilt.max():.1f} deg   height: mean={hz.mean():.3f} min={hz.min():.3f}")
ok = (not fell) and drift.max() < DRIFT_MAX and np.abs(yawerr).max() < YAW_MAX
print("RESULT:", "IN-PLACE OK" if ok else "DRIFTED / FELL")

imageio.mimsave(OUT, frames, fps=20)
print("gif ->", OUT, f"({len(frames)} frames)")
sys.exit(0 if ok else 1)
