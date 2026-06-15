"""Validate the scene: hold the standing keyframe with joint-space PD and check
the robot stays upright on its feet. Also sanity-prints the SRB constants.

  D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG3.0_description\tests\test_stand.py
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srb_mpc.robot import RobotModel

SCENE = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_scene.xml"
OUT = r"D:\mujoco\DOG3.0_description\scene\DOG3.0_pdhold.jpg"

M = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(M)
rob = RobotModel(M)
print(f"mass={rob.mass:.3f} kg  g={rob.g:.3f}")
print("composite I_body=\n", np.round(rob.I_body, 5))
print("hip offsets (body frame):\n", np.round(rob.hip_body, 3))

mujoco.mj_resetDataKeyframe(M, d, 0)
qpos_des = d.qpos.copy()                         # keyframe pose
mujoco.mj_forward(M, d)

# index maps (flattened, leg-major) so torque routes to the correct joint
qadr = rob.qadr.flatten(); vadr = rob.vadr.flatten(); cadr = rob.ctrl.flatten()
des = qpos_des[qadr]
tmax = M.actuator_ctrlrange[:, 1]

z0 = rob.com(d)[2]
Kp, Kd = 150.0, 4.0
T = 2.0
nstep = int(T / M.opt.timestep)
heights, rolls = [], []
for k in range(nstep):
    tau = Kp * (des - d.qpos[qadr]) - Kd * d.qvel[vadr]
    d.ctrl[cadr] = np.clip(tau, -tmax[cadr], tmax[cadr])
    mujoco.mj_step(M, d)
    heights.append(rob.com(d)[2])
    rolls.append(rob.rpy(d)[:2])

heights = np.array(heights); rolls = np.array(rolls)
print(f"\nCoM height: start={z0:.4f}  end={heights[-1]:.4f}  min={heights.min():.4f}")
print(f"final roll/pitch (deg)={np.round(np.degrees(rolls[-1]),2)}  "
      f"max|tilt|={np.degrees(np.abs(rolls).max()):.2f} deg")
upright = heights[-1] > 0.6 * z0 and np.degrees(np.abs(rolls[-1])).max() < 20
print("RESULT:", "STAND OK" if upright else "FELL / unstable")

cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(M, cam)
cam.azimuth, cam.elevation, cam.distance = 130, -20, cam.distance * 0.55
with mujoco.Renderer(M, 720, 960) as r:
    r.update_scene(d, camera=cam)
    imageio.imwrite(OUT, r.render(), quality=95)
print("render ->", OUT)
