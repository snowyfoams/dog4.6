"""Milestone 3: trot in place, then trot forward; save a gif as evidence.

  D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG3.0_description\tests\test_trot.py [vx]
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srb_mpc.controller import LocomotionController

SCENE = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_scene.xml"
OUT = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_trot.gif"
VX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4

M = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(M)
ctrl = LocomotionController(M, stand=False)
ctrl.reset(d)
rob = ctrl.rob

p0 = rob.com(d).copy()
frames, hz, tilt = [], [], []
cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(M, cam)
cam.azimuth, cam.elevation, cam.distance = 120, -15, cam.distance * 0.6

T_warm, T_run = 1.0, 4.0
dt = M.opt.timestep
diverged = False
with mujoco.Renderer(M, 480, 640) as r:
    for k in range(int((T_warm + T_run) / dt)):
        t = k * dt
        # ramp the command in after a short warm-up trotting in place
        ctrl.cmd.vx = VX if t > T_warm else 0.0
        d.ctrl[:] = ctrl(d)
        mujoco.mj_step(M, d)
        hz.append(rob.com(d)[2]); tilt.append(np.degrees(np.abs(rob.rpy(d)[:2])).max())
        if not np.isfinite(d.qpos).all():
            print("DIVERGED at t=", round(t, 3)); diverged = True; break
        if k % 10 == 0:
            cam.lookat[:] = rob.com(d)
            r.update_scene(d, camera=cam); frames.append(r.render())

p1 = rob.com(d).copy()
hz = np.array(hz); tilt = np.array(tilt)
print(f"vx_cmd={VX}  CoM moved dx={p1[0]-p0[0]:+.3f} dy={p1[1]-p0[1]:+.3f} m over {T_run}s")
print(f"mean fwd speed ~ {(p1[0]-p0[0])/T_run:+.3f} m/s")
print(f"CoM z mean={hz.mean():.3f} min={hz.min():.3f} max={hz.max():.3f}  max tilt={tilt.max():.1f} deg")
print(f"mean MPC solve = {1e3*np.mean(ctrl.solve_times):.2f} ms")
print("RESULT:", "FELL" if (diverged or tilt[-1] > 40 or hz[-1] < 0.07) else "UPRIGHT")
imageio.mimsave(OUT, frames, fps=50)
print("gif ->", OUT, f"({len(frames)} frames)")
