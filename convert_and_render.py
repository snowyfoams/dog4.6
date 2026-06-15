"""Render the DOG4.6 scene to JPG (offscreen). Parity with DOG3.0's convert_and_render;
the convert step lives in urdf\\build_urdf.py / build_mjcf.py (Fusion pipeline) and
scene\\build_dog_scene.py (scene authoring)."""
import os
import mujoco
import sys as _s, os as _o
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), 'scene'))
import actuator_envelope
import imageio.v2 as imageio

ROOT = os.path.dirname(os.path.abspath(__file__))
M = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "scene", "DOG4.6_scene.xml"))
actuator_envelope.install()   # must come AFTER model load
d = mujoco.MjData(M)
mujoco.mj_resetDataKeyframe(M, d, 0)
mujoco.mj_forward(M, d)
r = mujoco.Renderer(M, 720, 1080)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.12]
cam.distance, cam.azimuth, cam.elevation = 1.3, 135, -18
r.update_scene(d, cam)
out = os.path.join(ROOT, "DOG4.6_render.jpg")
imageio.imwrite(out, r.render())
print("wrote", out)
