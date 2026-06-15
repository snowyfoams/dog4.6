import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srb_mpc.controller import LocomotionController

M = mujoco.MjModel.from_xml_path(r"D:\mujoco\DOG3.0_description\scene\DOG3.0_scene.xml")
d = mujoco.MjData(M)
VX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
ctrl = LocomotionController(M, stand=False); ctrl.reset(d)
rob = ctrl.rob
dt = M.opt.timestep
p0 = rob.com(d).copy()
for k in range(int(6.0 / dt)):
    ctrl.cmd.vx = VX
    d.ctrl[:] = ctrl(d)
    mujoco.mj_step(M, d)
    if k % 50 == 0:
        com = rob.com(d); rpy = np.degrees(rob.rpy(d))
        ct = ctrl.gait.contact(ctrl.t).astype(int)
        v = rob.com_vel(d)
        fz = rob.foot_pos(d)[:, 2]
        # swing-foot landing error: for scheduled-stance feet, how high is the foot?
        print(f"t={k*dt:.2f} z={com[2]:.3f} dxy=({com[0]-p0[0]:+.2f},{com[1]-p0[1]:+.2f}) "
              f"rpy={np.round(rpy,0)} v=({v[0]:+.2f},{v[1]:+.2f}) contact={ct} "
              f"footz={np.round(fz,3)}")
    if not np.isfinite(d.qpos).all():
        print("  diverged t=", round(k*dt, 3)); break
