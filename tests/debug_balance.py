import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srb_mpc.controller import LocomotionController

M = mujoco.MjModel.from_xml_path(r"D:\mujoco\DOG3.0_description\scene\DOG3.0_scene.xml")
d = mujoco.MjData(M)
REPLAN = int(sys.argv[1]) if len(sys.argv) > 1 else 15
ctrl = LocomotionController(M, stand=True, replan_every=REPLAN); ctrl.reset(d)
rob = ctrl.rob


def foot_normal():
    f = np.zeros(4)
    for c in range(d.ncon):
        con = d.contact[c]
        ff = np.zeros(6); mujoco.mj_contactForce(M, d, c, ff)
        # find which foot geom is involved
        for leg in range(4):
            gid = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_GEOM, f"{['FR','FL','RR','RL'][leg]}_foot_col")
            if con.geom1 == gid or con.geom2 == gid:
                f[leg] += abs(ff[0])
    return f


print(f"replan_every={REPLAN}")
for k in range(int(1.0 / M.opt.timestep)):
    d.ctrl[:] = ctrl(d)
    mujoco.mj_step(M, d)
    if k % 50 == 0:
        com = rob.com(d); rpy = np.degrees(rob.rpy(d))
        fz = rob.foot_pos(d)[:, 2]
        print(f"t={k*M.opt.timestep:.2f} z={com[2]:.3f} rpy={np.round(rpy,1)} "
              f"footz={np.round(fz,3)} Nrm={np.round(foot_normal(),1)}")
    if not np.isfinite(d.qpos).all():
        print("  diverged at t=", round(k * M.opt.timestep, 3)); break
