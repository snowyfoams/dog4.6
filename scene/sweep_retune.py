# Sweep shared gait/swing tuning (applied identically to ALL stacks via default
# overrides) and report how far each fragile stack survives an S1-style staircase.
# Fairness preserved: only the SHARED TrotGait / SwingController defaults change.
import os, sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import actuator_envelope as env
from srb_mpc.gait import TrotGait
from srb_mpc.swing import SwingController
from srb_mpc.controller import LocomotionController
from srb_mpc.eth_controller import LocomotionControllerETH

M = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "scene", "DOG4.6_scene.xml"))
env.install()
dt = M.opt.timestep
SPEEDS = [0.1, 0.2, 0.3, 0.4, 0.5]

STACKS = {
    "mit": lambda: LocomotionController(M, stand=True),
    "x1":  lambda: LocomotionControllerETH(M, stand=True, high_level="convex", low_level="wbc"),
    "x2":  lambda: LocomotionControllerETH(M, stand=True, high_level="slq", low_level="jt"),
    "eth": lambda: LocomotionControllerETH(M, stand=True),
}

# (label, period, duty, step_h, kp, kd, raibert_k)
COMBOS = [
    ("stock           ", 0.34, 0.6, 0.045, 500, 14, 0.10),
    ("slow42           ", 0.42, 0.6, 0.045, 500, 14, 0.10),
    ("slow50           ", 0.50, 0.6, 0.045, 500, 14, 0.10),
    ("slow46+raibert16 ", 0.46, 0.6, 0.045, 500, 14, 0.16),
    ("slow46+soft+rai16", 0.46, 0.6, 0.045, 400, 12, 0.16),
    ("slow50+rai18+lowh", 0.50, 0.6, 0.035, 400, 12, 0.18),
]


def set_tuning(period, duty, step_h, kp, kd, rai):
    TrotGait.__init__.__defaults__ = (period, duty)
    # SwingController defaults: step_height, kp, kd, raibert_k, foot_ground
    SwingController.__init__.__defaults__ = (step_h, kp, kd, rai, 0.016)


def survive_to(make, hold=2.0):
    d = mujoco.MjData(M)
    ctrl = make()
    ctrl.reset(d)
    rob = ctrl.rob
    for _ in range(int(1.0 / dt)):
        ctrl.gait.stand = True
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    ctrl.gait.stand = False
    for _ in range(int(1.0 / dt)):
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    best = 0.0
    for v in SPEEDS:
        ctrl.cmd.vx = v
        ok = True
        for _ in range(int(hold / dt)):
            d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
            if not np.isfinite(d.qpos).all() or np.degrees(np.abs(rob.rpy(d)[:2])).max() > 45 \
                    or rob.com(d)[2] < 0.07:
                ok = False; break
        if not ok:
            break
        best = v
    return best


if __name__ == "__main__":
    probes = sys.argv[1:] or ["mit", "x1", "x2", "eth"]
    print("combo              | " + "  ".join("%5s" % p for p in probes) + " | sum")
    print("-" * 60)
    for combo in COMBOS:
        label = combo[0]
        set_tuning(*combo[1:])
        res = []
        for p in probes:
            res.append(survive_to(STACKS[p]))
        print("%s | %s | %.1f" % (label, "  ".join("%5.2f" % r for r in res), sum(res)))
