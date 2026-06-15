# Smoke test for the NEW dog scene with the four unchanged control stacks.
# 1) MIT stand 5 s   2) each stack trot 0.3 m/s for 3 s
import os, sys
import numpy as np
import mujoco
import actuator_envelope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from srb_mpc.controller import LocomotionController
from srb_mpc.eth_controller import LocomotionControllerETH

M = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "scene", "DOG4.6_scene.xml"))
actuator_envelope.install()   # must come AFTER model load (compile-time callback crash)
STACKS = {
    "mit": lambda: LocomotionController(M, stand=True),
    "eth": lambda: LocomotionControllerETH(M, stand=True),
    "cvxwbc": lambda: LocomotionControllerETH(M, stand=True, high_level="convex", low_level="wbc"),
    "slqjt": lambda: LocomotionControllerETH(M, stand=True, high_level="slq", low_level="jt"),
}
dt = M.opt.timestep


def run(name, make, trot_T=3.0):
    d = mujoco.MjData(M)
    ctrl = make()
    ctrl.reset(d)
    rob = ctrl.rob
    # stand 2 s
    for _ in range(int(2.0 / dt)):
        ctrl.gait.stand = True
        d.ctrl[:] = ctrl(d)
        mujoco.mj_step(M, d)
    rp = np.degrees(np.abs(rob.rpy(d)[:2])).max()
    h = rob.com(d)[2]
    stand_ok = np.isfinite(d.qpos).all() and rp < 10 and h > 0.12
    # trot 0.3
    ctrl.gait.stand = False
    ctrl.cmd.vx = 0.3
    sat = []
    for _ in range(int(trot_T / dt)):
        tau = ctrl(d)
        d.ctrl[:] = tau
        sat.append(np.mean(np.abs(tau) >= 0.98 * 7.0))
        mujoco.mj_step(M, d)
        if not np.isfinite(d.qpos).all():
            break
    rp2 = np.degrees(np.abs(rob.rpy(d)[:2])).max()
    h2 = rob.com(d)[2]
    vx = rob.com_vel(d)[0]
    trot_ok = np.isfinite(d.qpos).all() and rp2 < 45 and h2 > 0.07
    print("%-7s stand: %s (rp=%4.1fdeg h=%.3f) | trot: %s (rp=%4.1f h=%.3f vx=%.2f sat=%4.1f%%)" % (
        name, "OK " if stand_ok else "FAIL", rp, h,
        "OK " if trot_ok else "FAIL", rp2, h2, vx, 100 * float(np.mean(sat))))
    return stand_ok and trot_ok


if __name__ == "__main__":
    keys = sys.argv[1:] or list(STACKS)
    allok = all([run(k, STACKS[k]) for k in keys])
    print("SMOKE", "PASS" if allok else "FAIL")
