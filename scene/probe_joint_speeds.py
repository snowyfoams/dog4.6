# Probe: do benchmark gaits ever reach the actuator envelope?
# Records per-joint |qvel| peaks during trot at 0.3 / 0.5 m/s and an S2-style ramp,
# against the envelope corners: 22.9 rad/s (power), 31.1 (cliff start), 33.5 (zero).
# Usage: python probe_joint_speeds.py [ideal|envelope]   (default: envelope)
import os, sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import actuator_envelope as env

MODE = sys.argv[1] if len(sys.argv) > 1 else "envelope"
_INSTALL_PENDING = (MODE == "envelope")
print("[actuator mode: %s]" % MODE)

from srb_mpc.controller import LocomotionController
from srb_mpc.eth_controller import LocomotionControllerETH

M = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "scene", "DOG4.6_scene.xml"))
if _INSTALL_PENDING:
    env.install()
dt = M.opt.timestep
STACKS = {
    "mit": lambda: LocomotionController(M, stand=True),
    "eth": lambda: LocomotionControllerETH(M, stand=True),
    "slqjt": lambda: LocomotionControllerETH(M, stand=True, high_level="slq", low_level="jt"),
}
CORNERS = (22.9, 31.1, 33.5)


def run(name, make, vxs, T=3.0):
    d = mujoco.MjData(M)
    ctrl = make()
    ctrl.reset(d)
    wmax = 0.0
    frac_above = [0, 0, 0]
    n = 0
    for _ in range(int(1.0 / dt)):
        ctrl.gait.stand = True
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    ctrl.gait.stand = False
    for vx in vxs:
        ctrl.cmd.vx = vx
        for _ in range(int(T / dt)):
            d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
            if not np.isfinite(d.qpos).all():
                print("  %s: diverged at vx=%.2f" % (name, vx))
                return
            w = np.abs(d.qvel[6:]).max()
            wmax = max(wmax, w)
            for i, c in enumerate(CORNERS):
                frac_above[i] += int(w > c)
            n += 1
            if np.degrees(np.abs(ctrl.rob.rpy(d)[:2])).max() > 45 or ctrl.rob.com(d)[2] < 0.07:
                print("  %s: fell at vx=%.2f (wmax so far %.1f)" % (name, vx, wmax))
                return wmax, frac_above, n
    return wmax, frac_above, n


for name, make in STACKS.items():
    r = run(name, make, [0.3, 0.5, 0.7])
    if r:
        wmax, fa, n = r
        print("%-6s peak |qvel| = %5.1f rad/s (%4.0f rpm out) | time above corners "
              "22.9/31.1/33.5: %.2f%% / %.2f%% / %.2f%%" % (
            name, wmax, wmax * 60 / (2 * np.pi), 100 * fa[0] / n, 100 * fa[1] / n, 100 * fa[2] / n))
print("corners: power cap starts 22.9 rad/s, voltage cliff 31.1, zero torque 33.5")
