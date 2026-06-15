# Generate DOG4.6 media set (parity with DOG3.0's scene/*.jpg + *.gif):
#   DOG4.6_stand.jpg                      keyframe pose
#   DOG4.6_trot.gif                       X2 (SLQ+leg) 0.3 m/s  (best stack on this platform)
#   DOG4.6_eth_trot.gif                   ETH (SLQ+WBC) 0.3 m/s
#   DOG4.6_trot_in_place.gif              X2, vx=0
#   DOG4.6_drive.gif                      ETH scripted drive: fwd -> turn -> strafe
#   DOG4.6_jump_{mit,eth,cvxwbc,slqjt}.gif  pronk on each stack (7 Nm — may be marginal)
import os, sys
import numpy as np
import mujoco
import actuator_envelope
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

try:
    _FONT = ImageFont.truetype("arial.ttf", 22)
except Exception:
    _FONT = ImageFont.load_default()

LABELS = {
    "mit":    "MIT: convex MPC + leg ctrl",
    "eth":    "ETH: SLQ-MPC + QP-WBC",
    "cvxwbc": "X1: convex MPC + QP-WBC",
    "slqjt":  "X2: SLQ-MPC + leg ctrl",
}


def stamp(frame, text):
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    dr.rectangle([8, 8, 16 + dr.textlength(text, font=_FONT), 40], fill=(0, 0, 0, 180))
    dr.text((12, 12), text, fill=(255, 255, 255), font=_FONT)
    return np.asarray(im)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from srb_mpc.controller import LocomotionController
from srb_mpc.eth_controller import LocomotionControllerETH
from srb_mpc import jump

SCENE = os.path.join(ROOT, "scene", "DOG4.6_scene.xml")
M = mujoco.MjModel.from_xml_path(SCENE)
actuator_envelope.install()   # must come AFTER model load
DT = M.opt.timestep
FPS = 25
W, H = 640, 480

STACKS = {
    "mit": lambda: LocomotionController(M, stand=True),
    "eth": lambda: LocomotionControllerETH(M, stand=True),
    "cvxwbc": lambda: LocomotionControllerETH(M, stand=True, high_level="convex", low_level="wbc"),
    "slqjt": lambda: LocomotionControllerETH(M, stand=True, high_level="slq", low_level="jt"),
}


class Cam:
    def __init__(self):
        self.c = mujoco.MjvCamera()
        self.c.distance, self.c.azimuth, self.c.elevation = 1.3, 135, -16


def record(make, script, path, T, label=None):
    """script(ctrl, t) mutates commands; records T seconds after 1 s stand."""
    d = mujoco.MjData(M)
    ctrl = make()
    ctrl.reset(d)
    r = mujoco.Renderer(M, H, W)
    cam = Cam()
    frames = []
    state = {"jm": None}
    n_pre, n_rec = int(1.0 / DT), int(T / DT)
    every = max(1, int(round(1.0 / (FPS * DT))))
    for k in range(n_pre + n_rec):
        t = k * DT
        if k < n_pre:
            ctrl.gait.stand = True
        else:
            script(ctrl, t - 1.0, d, state)
        if state["jm"] is not None and ctrl.z_ref_fn is not None and state["jm"].done(ctrl.t):
            jump.restore(ctrl, stand=True)
            state["jm"] = None
        d.ctrl[:] = ctrl(d)
        mujoco.mj_step(M, d)
        if not np.isfinite(d.qpos).all():
            break
        if k >= n_pre and (k - n_pre) % every == 0:
            cam.c.lookat[:] = ctrl.rob.com(d)
            r.update_scene(d, cam.c)
            f = r.render()
            if label:
                f = stamp(f, label)
            frames.append(f)
    imageio.mimsave(path, frames, fps=FPS, loop=0)
    print("wrote %s (%d frames)" % (os.path.basename(path), len(frames)))


def trot_at(vx):
    def s(ctrl, t, d, st):
        ctrl.gait.stand = False
        ctrl.cmd.vx = vx
    return s


def drive(ctrl, t, d, st):
    ctrl.gait.stand = False
    c = ctrl.cmd
    if t < 2.0:
        c.vx, c.wz, c.vy = 0.25, 0.0, 0.0
    elif t < 4.0:
        c.vx, c.wz, c.vy = 0.15, 0.5, 0.0
    else:
        c.vx, c.wz, c.vy = 0.0, 0.0, 0.12


def jumper(ctrl, t, d, st):
    ctrl.gait.stand = True
    if st["jm"] is None and ctrl.z_ref_fn is None and 0.2 < t < 0.3:
        ctrl.cmd.clear()
        st["jm"] = jump.install(ctrl, flight_t=0.30)


if __name__ == "__main__":
    out = os.path.join(ROOT, "scene")
    only = sys.argv[1:] or ["stand", "trot", "eth_trot", "in_place", "drive", "jump"]

    if "stand" in only:
        d = mujoco.MjData(M)
        mujoco.mj_resetDataKeyframe(M, d, 0)
        mujoco.mj_forward(M, d)
        r = mujoco.Renderer(M, 720, 1080)
        cam = Cam()
        cam.c.lookat[:] = [0, 0, 0.12]
        r.update_scene(d, cam.c)
        imageio.imwrite(os.path.join(out, "DOG4.6_stand.jpg"), r.render())
        print("wrote DOG4.6_stand.jpg")
    if "trot" in only:
        # all four stacks, identical command (0.3 m/s), labelled — benchmark conditions
        for k in ("mit", "eth", "cvxwbc", "slqjt"):
            record(STACKS[k], trot_at(0.3), os.path.join(out, "DOG4.6_trot_%s.gif" % k),
                   4.0, label=LABELS[k] + "  |  cmd 0.3 m/s")
    if "in_place" in only:
        record(STACKS["slqjt"], trot_at(0.0), os.path.join(out, "DOG4.6_trot_in_place.gif"),
               4.0, label=LABELS["slqjt"] + "  |  in place")
    if "drive" in only:
        record(STACKS["eth"], drive, os.path.join(out, "DOG4.6_drive.gif"), 6.0,
               label=LABELS["eth"] + "  |  drive")
    if "jump" in only:
        for k in ("mit", "eth", "cvxwbc", "slqjt"):
            record(STACKS[k], jumper, os.path.join(out, "DOG4.6_jump_%s.gif" % k), 3.0,
                   label=LABELS[k] + "  |  pronk")
    print("MEDIA DONE")
