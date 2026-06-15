r"""Interactive MuJoCo viewer: drive the DOG4.6 quadruped (7 Nm) with the keyboard.

    D:\mujoco\.venv\Scripts\python.exe D:\mujoco\DOG4.6_description\run_dog.py [--stack NAME]

Stacks (benchmark ranking on this 7 Nm platform — see results/):
    slqjt  SLQ-MPC + leg controller   [DEFAULT — best: tracks to 0.4, ceiling 0.65 m/s]
    eth    SLQ-MPC + QP-WBC           (ceiling 0.65 but barely tracks; best pushes)
    cvxwbc convex MPC + QP-WBC        (falls above ~0.15 m/s)
    mit    convex MPC + leg controller (falls above ~0.1 m/s — worst here)
    --eth is kept as an alias for --stack eth.

Controls (focus the viewer window):
    W / S : forward / backward      A / D : turn left / right
    Q / E : strafe left / right     SPACE : stop & stand
    R     : reset to standing pose  J     : jump in place (pronk)

The robot stands still until you give a motion command, then trots. It
auto-resets if it tips over. Close the window to quit.
"""
import os
import sys
import time
import numpy as np
import mujoco
import sys as _s, os as _o
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), 'scene'))
import actuator_envelope
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srb_mpc import jump
from srb_mpc.controller import LocomotionController as _MIT
from srb_mpc.eth_controller import LocomotionControllerETH as _ETH

STACK = "slqjt"
if "--eth" in sys.argv:
    STACK = "eth"
if "--stack" in sys.argv:
    STACK = sys.argv[sys.argv.index("--stack") + 1]

_FACTORY = {
    "mit":    (lambda m: _MIT(m, stand=True),
               "MIT — convex MPC + leg controller"),
    "eth":    (lambda m: _ETH(m, stand=True),
               "ETH — SLQ-MPC + QP-WBC"),
    "cvxwbc": (lambda m: _ETH(m, stand=True, high_level="convex", low_level="wbc"),
               "X1 — convex MPC + QP-WBC"),
    "slqjt":  (lambda m: _ETH(m, stand=True, high_level="slq", low_level="jt"),
               "X2 — SLQ-MPC + leg controller (best on this platform)"),
}
_make, _label = _FACTORY[STACK]
print("[stack: %s]" % _label)

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene", "DOG4.6_scene.xml")

VX_STEP, VY_STEP, WZ_STEP = 0.05, 0.04, 0.2
# keyboard limits matched to measured platform capability per stack
_VXMAX = {"mit": 0.10, "cvxwbc": 0.15, "eth": 0.50, "slqjt": 0.40}
VX_MAX, VY_MAX, WZ_MAX = _VXMAX[STACK], 0.12, 0.6

M = mujoco.MjModel.from_xml_path(SCENE)
actuator_envelope.install()   # must come AFTER model load
d = mujoco.MjData(M)
ctrl = _make(M)
ctrl.reset(d)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def on_key(keycode):
    c = ctrl.cmd
    moved = True
    if keycode == ord('W'):
        c.vx = clamp(c.vx + VX_STEP, -VX_MAX, VX_MAX)
    elif keycode == ord('S'):
        c.vx = clamp(c.vx - VX_STEP, -VX_MAX, VX_MAX)
    elif keycode == ord('Q'):
        c.vy = clamp(c.vy + VY_STEP, -VY_MAX, VY_MAX)
    elif keycode == ord('E'):
        c.vy = clamp(c.vy - VY_STEP, -VY_MAX, VY_MAX)
    elif keycode == ord('A'):
        c.wz = clamp(c.wz + WZ_STEP, -WZ_MAX, WZ_MAX)
    elif keycode == ord('D'):
        c.wz = clamp(c.wz - WZ_STEP, -WZ_MAX, WZ_MAX)
    elif keycode == ord(' '):
        c.clear(); ctrl.gait.stand = True; moved = False
    elif keycode == ord('R'):
        c.clear(); ctrl.reset(d); ctrl.gait.stand = True; moved = False
        print("  [reset]")
    elif keycode == ord('J'):
        global jump_jm
        if jump_jm is None and ctrl.z_ref_fn is None:
            c.clear()
            jump_jm = jump.install(ctrl, flight_t=0.30)
            print("  [JUMP]")
        moved = False
    else:
        return
    if moved:
        ctrl.gait.stand = False
    print(f"  cmd: vx={c.vx:+.2f}  vy={c.vy:+.2f}  yaw={c.wz:+.2f}  "
          f"{'TROT' if not ctrl.gait.stand else 'STAND'}")


def fallen():
    return ctrl.rob.com(d)[2] < 0.07 or np.degrees(np.abs(ctrl.rob.rpy(d)[:2])).max() > 50


jump_jm = None

print(__doc__)
with mujoco.viewer.launch_passive(M, d, key_callback=on_key) as v:
    v.cam.distance = 1.4
    v.cam.elevation = -20
    v.cam.azimuth = 120
    while v.is_running():
        t0 = time.time()
        if jump_jm is not None and ctrl.z_ref_fn is not None and jump_jm.done(ctrl.t):
            jump.restore(ctrl, stand=True)
            jump_jm = None
            print("  [landed]")
        d.ctrl[:] = ctrl(d)
        mujoco.mj_step(M, d)
        if not np.isfinite(d.qpos).all() or fallen():
            print("  [fell - auto reset]")
            ctrl.cmd.clear(); ctrl.reset(d); ctrl.gait.stand = True
            jump_jm = None
        v.cam.lookat[:] = ctrl.rob.com(d)
        v.sync()
        lag = M.opt.timestep - (time.time() - t0)
        if lag > 0:
            time.sleep(lag)
