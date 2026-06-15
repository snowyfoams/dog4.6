# Diagnose WHY MIT/X1/X2 fall at vx=0.2 on DOG4.6: hold 0.2 m/s and time-trace
# the body attitude, height, torque saturation, joint speed and foot slip until fall.
import os, sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import actuator_envelope as env
from srb_mpc.controller import LocomotionController
from srb_mpc.eth_controller import LocomotionControllerETH

M = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "scene", "DOG4.6_scene.xml"))
env.install()
dt = M.opt.timestep
TAU = 7.0
STACKS = {
    "mit": lambda: LocomotionController(M, stand=True),
    "eth": lambda: LocomotionControllerETH(M, stand=True),
    "x2":  lambda: LocomotionControllerETH(M, stand=True, high_level="slq", low_level="jt"),
}


def diagnose(name, make, vx=0.2, T=6.0):
    d = mujoco.MjData(M)
    ctrl = make()
    ctrl.reset(d)
    rob = ctrl.rob
    for _ in range(int(1.0 / dt)):
        ctrl.gait.stand = True
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    ctrl.gait.stand = False
    ctrl.cmd.vx = vx
    print("\n=== %s @ vx=%.2f ===" % (name.upper(), vx))
    print(" t[s]  roll  pitch   h[m]  |w|max  sat%  vx_act")
    t_fall = None
    last_print = -1
    for k in range(int(T / dt)):
        tau = ctrl(d)
        d.ctrl[:] = tau
        mujoco.mj_step(M, d)
        t = k * dt
        rp = np.degrees(rob.rpy(d)[:2])
        h = rob.com(d)[2]
        wmax = np.abs(d.qvel[6:]).max()
        sat = 100 * np.mean(np.abs(tau) >= 0.98 * TAU)
        vxa = rob.com_vel(d)[0]
        if t - last_print >= 0.25:
            print("%5.2f %6.1f %6.1f %6.3f %6.1f %5.1f %6.2f" % (
                t, rp[0], rp[1], h, wmax, sat, vxa))
            last_print = t
        if not np.isfinite(d.qpos).all() or np.abs(rp).max() > 45 or h < 0.07:
            t_fall = t
            mode = "height collapse" if h < 0.07 else ("roll-over" if abs(rp[0]) > abs(rp[1]) else "pitch-over")
            print(">>> FELL at t=%.2f s, mode=%s (roll=%.1f pitch=%.1f h=%.3f)" % (
                t, mode, rp[0], rp[1], h))
            break
    if t_fall is None:
        print(">>> survived %.1f s at vx=%.2f (no fall)" % (T, vx))
    return t_fall


if __name__ == "__main__":
    keys = sys.argv[1:] or ["mit", "x2", "eth"]
    for k in keys:
        diagnose(k, STACKS[k])
