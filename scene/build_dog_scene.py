# Generate scene/DOG3.0_scene.xml for the NEW Fusion-exported dog (tau_max = 7 Nm),
# with naming 100% compatible with srb_mpc/robot.py (joint{1..12}, m{1..12},
# {leg}_foot sites, body "trunk", keyframe 0 = "stand"). Control code untouched.
#
# Conventions copied from the original DOG3.0 scene (DOG3.0_scene.orig.xml):
#   foot sphere r=0.016 priority=1 friction "0.9 0.02 0.001", floor plane,
#   integrator implicitfast, dt=0.002, absolute meshdir.
# Robot data from the Fusion export session (verified by FK + yourdfpy + mujoco).
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
MESHDIR = "D:/mujoco/DOG4.6_description/meshes"
OUT = os.path.join(HERE, "DOG4.6_scene.xml")
MT = json.load(open(os.path.join(os.path.dirname(HERE), "urdf", "mesh_transforms.json")))

LINKS = json.loads(r'''
{"trunk": {"mass": 3.288409, "com_cm": [0.071, -0.0009, -0.8279], "I_com_kgcm2": [159.0483, 937.76955, 1064.66566, -0.01667, -0.00471, -1.10111]},
"FR_hip": {"mass": 0.495739, "com_cm": [26.5394, -12.1997, -0.0001], "I_com_kgcm2": [3.40872, 3.25733, 4.09086, 1.0822, -0.00014, -0.00013]},
"FR_thigh": {"mass": 0.468206, "com_cm": [21.3052, -13.1295, -9.8341], "I_com_kgcm2": [3.93064, 5.01256, 2.48721, 0.32328, 0.58957, -1.17418]},
"FR_calf": {"mass": 0.047362, "com_cm": [24.7612, -10.6759, -17.098], "I_com_kgcm2": [0.89083, 1.1814, 0.33338, 0.0, 0.0, 0.45671]},
"FL_hip": {"mass": 0.495739, "com_cm": [26.5394, 12.1997, 0.0001], "I_com_kgcm2": [3.40872, 3.25733, 4.09086, -1.0822, -0.00013, 0.00012]},
"FL_thigh": {"mass": 0.468206, "com_cm": [21.3286, 13.1295, -9.8465], "I_com_kgcm2": [3.86513, 5.01195, 2.55212, -0.34957, -0.57546, -1.21157]},
"FL_calf": {"mass": 0.047362, "com_cm": [24.7612, 10.6759, -17.098], "I_com_kgcm2": [0.89083, 1.1814, 0.33338, 0.0, 0.0, 0.45671]},
"RL_hip": {"mass": 0.495739, "com_cm": [-26.538, 12.1997, 0.0002], "I_com_kgcm2": [3.40872, 3.25672, 4.09025, 1.08305, -0.00013, 0.00018]},
"RL_thigh": {"mass": 0.468206, "com_cm": [-32.6681, 13.1295, -9.847], "I_com_kgcm2": [3.86494, 5.01256, 2.55291, -0.34894, -0.57475, -1.21212]},
"RL_calf": {"mass": 0.047362, "com_cm": [-29.2344, 10.6759, -17.098], "I_com_kgcm2": [0.89083, 1.1814, 0.33338, 0.0, 0.0, 0.45671]},
"RR_hip": {"mass": 0.495739, "com_cm": [-26.538, -12.1997, -0.0002], "I_com_kgcm2": [3.40872, 3.25672, 4.09025, -1.08305, -0.00013, -0.00018]},
"RR_thigh": {"mass": 0.468206, "com_cm": [-32.6904, -13.1295, -9.8341], "I_com_kgcm2": [3.93064, 5.01256, 2.48721, 0.32328, 0.58957, -1.17418]},
"RR_calf": {"mass": 0.047362, "com_cm": [-29.2344, -10.6759, -17.098], "I_com_kgcm2": [0.89083, 1.1814, 0.33338, 0.0, 0.0, 0.45671]}}
''')

# child-frame positions (m) in parent frame; axes; from Fusion joint extraction
LEG_DATA = {  # leg: (side y sign, fwd x sign)
    "FR": (-1, +1), "FL": (+1, +1), "RR": (-1, -1), "RL": (+1, -1),
}
# robot.py LEG_JOINT_NUM: joint numbers per leg [hip(abd), thigh, calf]
JNUM = {"FR": (12, 11, 10), "FL": (9, 8, 7), "RR": (1, 3, 2), "RL": (6, 5, 4)}

HIP_X, HIP_Y = 0.269978, 0.06
THIGH_Y = 0.056109
CALF_DX, CALF_DZ = -0.06, -0.103923
# foot sphere center in calf frame (= foot mesh COM): x .06, y ±.00935, z -.106907
FOOT_LOCAL = (0.06, 0.00935, -0.106907)
FOOT_R = 0.016
STAND_Z = 0.210830 + FOOT_R   # foot center depth below trunk + sphere radius

LIM = {"hip": (-0.80, 0.80), "thigh": (-2.60, 2.60), "calf": (-2.60, 2.60)}
TAU = 7.0
# Outward stance splay baked into the stand keyframe (lateral support margin),
# same trick as DOG3.0's STAND_SPREAD: feet directly under the hips roll over in trot.
SPLAY = 0.10  # rad; right legs -, left legs +

LINK_MESHES = {"trunk": ["base_link", "FR_abd_motor", "FL_abd_motor", "RL_abd_motor", "RR_abd_motor"]}
for leg in LEG_DATA:
    LINK_MESHES[leg + "_hip"] = [leg + "_hip", leg + "_hip_motor"]
    LINK_MESHES[leg + "_thigh"] = [leg + "_thigh", leg + "_knee_motor"]
    LINK_MESHES[leg + "_calf"] = [leg + "_calf", leg + "_foot"]

FRAME = {"trunk": (0.0, 0.0, 0.0)}
for leg, (s, fx) in LEG_DATA.items():
    hip = (fx * HIP_X, s * HIP_Y, 0.0)
    thigh = (hip[0], hip[1] + s * THIGH_Y, 0.0)
    calf = (thigh[0] + CALF_DX, thigh[1], thigh[2] + CALF_DZ)
    FRAME[leg + "_hip"], FRAME[leg + "_thigh"], FRAME[leg + "_calf"] = hip, thigh, calf


def fmt(v):
    s = "%.8f" % v
    s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def quat_from_R(R):
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x = 0.25 * s, (R[2][1] - R[1][2]) / s
        y, z = (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        w, x = (R[2][1] - R[1][2]) / s, 0.25 * s
        y, z = (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        w, x = (R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s
        y, z = 0.25 * s, (R[1][2] + R[2][1]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        w, x = (R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s
        y, z = (R[1][2] + R[2][1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return w / n, x / n, y / n, z / n


L = []
def emit(i, s): L.append("  " * i + s)


def inertial(name, ind):
    lk, f = LINKS[name], FRAME[name]
    com = [(lk["com_cm"][i] * 0.01 - f[i]) for i in range(3)]
    I = [v * 1e-4 for v in lk["I_com_kgcm2"]]  # ixx iyy izz ixy iyz ixz
    emit(ind, '<inertial pos="%s %s %s" mass="%s" fullinertia="%s %s %s %s %s %s"/>' % (
        fmt(com[0]), fmt(com[1]), fmt(com[2]), fmt(lk["mass"]),
        fmt(I[0]), fmt(I[1]), fmt(I[2]), fmt(I[3]), fmt(I[5]), fmt(I[4])))  # MJCF: ixy ixz iyz


def geoms(name, ind):
    f = FRAME[name]
    for mesh in LINK_MESHES[name]:
        M = MT[mesh]
        t = [(M["t_cm"][i] * 0.01 - f[i]) for i in range(3)]
        q = quat_from_R(M["R"])
        emit(ind, '<geom type="mesh" mesh="dog_%s" pos="%s %s %s" quat="%s %s %s %s" '
                  'contype="0" conaffinity="0" group="1"/>' % (
            mesh, fmt(t[0]), fmt(t[1]), fmt(t[2]), fmt(q[0]), fmt(q[1]), fmt(q[2]), fmt(q[3])))


emit(0, '<mujoco model="dog_new">')
emit(1, '<!-- NEW Fusion dog (7.334 kg, MG5010-i10, tau_max 7 Nm). Naming matches srb_mpc/robot.py. -->')
emit(1, '<compiler angle="radian" meshdir="%s" autolimits="true"/>' % MESHDIR)
emit(1, '<option timestep="0.002" integrator="implicitfast"/>')
emit(1, '<visual><global offwidth="1440" offheight="1080"/></visual>')
emit(1, '<default>')
emit(2, '<joint type="hinge" damping="0.01" armature="0.0085"/>')
emit(1, '</default>')
emit(1, '<asset>')
emit(2, '<texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.9 0.9 0.9" width="64" height="64"/>')
emit(2, '<texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.35 0.4" rgb2="0.4 0.45 0.5" width="256" height="256"/>')
emit(2, '<material name="grid" texture="grid" texrepeat="4 4" reflectance="0.1"/>')
for ms in LINK_MESHES.values():
    for m in ms:
        emit(2, '<mesh name="dog_%s" file="dog_%s.stl" scale="0.001 0.001 0.001"/>' % (m, m))
emit(1, '</asset>')
emit(1, '<worldbody>')
emit(2, '<light pos="0 0 2" dir="0 0 -1" directional="true"/>')
emit(2, '<geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1" friction="0.9 0.02 0.001"/>')
emit(2, '<body name="trunk" pos="0 0 %s">' % fmt(STAND_Z))
emit(3, '<freejoint name="root"/>')
inertial("trunk", 3)
geoms("trunk", 3)
for leg, (s, fx) in LEG_DATA.items():
    nh, nt, nc = JNUM[leg]
    hip, thigh, calf = FRAME[leg + "_hip"], FRAME[leg + "_thigh"], FRAME[leg + "_calf"]
    emit(3, '<body name="%s_hip" pos="%s %s %s">' % (leg, fmt(hip[0]), fmt(hip[1]), fmt(hip[2])))
    emit(4, '<joint name="joint%d" axis="1 0 0" range="%s %s"/>' % (nh, fmt(LIM["hip"][0]), fmt(LIM["hip"][1])))
    inertial(leg + "_hip", 4)
    geoms(leg + "_hip", 4)
    emit(4, '<body name="%s_thigh" pos="%s %s %s">' % (
        leg, fmt(thigh[0] - hip[0]), fmt(thigh[1] - hip[1]), fmt(thigh[2] - hip[2])))
    emit(5, '<joint name="joint%d" axis="0 1 0" range="%s %s"/>' % (nt, fmt(LIM["thigh"][0]), fmt(LIM["thigh"][1])))
    inertial(leg + "_thigh", 5)
    geoms(leg + "_thigh", 5)
    emit(5, '<body name="%s_calf" pos="%s %s %s">' % (
        leg, fmt(calf[0] - thigh[0]), fmt(calf[1] - thigh[1]), fmt(calf[2] - thigh[2])))
    emit(6, '<joint name="joint%d" axis="0 1 0" range="%s %s"/>' % (nc, fmt(LIM["calf"][0]), fmt(LIM["calf"][1])))
    inertial(leg + "_calf", 6)
    geoms(leg + "_calf", 6)
    fy = s * FOOT_LOCAL[1] * -1 if False else (FOOT_LOCAL[1] if s < 0 else -FOOT_LOCAL[1])
    # right legs (s=-1): +y local offset; left legs (s=+1): -y  (foot COM vs calf frame)
    emit(6, '<site name="%s_foot" pos="%s %s %s" size="0.008" rgba="1 0 0 0.5"/>' % (
        leg, fmt(FOOT_LOCAL[0]), fmt(fy), fmt(FOOT_LOCAL[2])))
    emit(6, '<geom name="%s_foot_col" type="sphere" pos="%s %s %s" size="%s" contype="1" '
            'conaffinity="1" friction="0.9 0.02 0.001" rgba="0.2 0.2 0.2 1" priority="1"/>' % (
        leg, fmt(FOOT_LOCAL[0]), fmt(fy), fmt(FOOT_LOCAL[2]), fmt(FOOT_R)))
    emit(5, '</body>')
    emit(4, '</body>')
    emit(3, '</body>')
emit(2, '</body>')
emit(1, '</worldbody>')
emit(1, '<actuator>')
J2LEG = {}
for leg, nums in JNUM.items():
    for n, kind in zip(nums, ("hip", "thigh", "calf")):
        J2LEG[n] = (leg, kind)
# DOG4.6: current-controlled servo envelope via user bias callback (see
# actuator_envelope.py: flat 7 N*m -> 160 W power cap -> kt^2/R voltage cliff @320rpm).
# Without the callback installed, bias=0 and this is the ideal +/-7 source (= DOG4.0).
for n in range(1, 13):
    emit(2, '<general name="m%d" joint="joint%d" gear="1" gaintype="fixed" gainprm="1" '
            'biastype="user" ctrlrange="-%s %s" forcerange="-%s %s"/>' % (
        n, n, fmt(TAU), fmt(TAU), fmt(TAU), fmt(TAU)))
emit(1, '</actuator>')
# joint qpos order follows XML tree: FR(hip,thigh,calf), FL, RR, RL
SPLAY_Q = [-SPLAY, 0, 0,  SPLAY, 0, 0,  -SPLAY, 0, 0,  SPLAY, 0, 0]
emit(1, '<keyframe>')
emit(2, '<key name="stand" qpos="0 0 %s 1 0 0 0  %s"/>' % (
    fmt(STAND_Z), " ".join(fmt(q) for q in SPLAY_Q)))
emit(1, '</keyframe>')
emit(0, '</mujoco>')

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", OUT)

# ---- numeric keyframe correction: drop trunk so splayed feet rest on floor ----
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path(OUT)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "%s_foot" % l) for l in ("FR", "FL", "RR", "RL")]
zmin = min(d.site_xpos[s][2] for s in sids)
z_fix = STAND_Z - (zmin - FOOT_R)
txt = open(OUT, encoding="utf-8").read()
txt = txt.replace('<key name="stand" qpos="0 0 %s' % fmt(STAND_Z),
                  '<key name="stand" qpos="0 0 %s' % fmt(z_fix))
open(OUT, "w", encoding="utf-8").write(txt)
print("keyframe trunk z corrected: %s -> %s (splay %.2f rad)" % (fmt(STAND_Z), fmt(z_fix), SPLAY))

# ------------------------------- self checks -------------------------------
m = mujoco.MjModel.from_xml_path(OUT)
d = mujoco.MjData(m)
print("nbody=%d nq=%d nv=%d nu=%d | mass=%.3f kg" % (m.nbody, m.nq, m.nv, m.nu, m.body_mass.sum()))
assert m.nu == 12 and m.nq == 19
# actuator m{N} must sit at index N-1 and drive joint{N}
for n in range(1, 13):
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "m%d" % n)
    jid = m.actuator_trnid[aid, 0]
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    assert aid == n - 1 and jname == "joint%d" % n, (n, aid, jname)
print("actuator order m1..m12 == index 0..11, joints matched")
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for leg in ("FR", "FL", "RR", "RL"):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "%s_foot" % leg)
    p = d.site_xpos[sid]
    print("%s_foot site: (%7.4f, %7.4f, %7.4f)  sphere bottom z=%.4f" % (leg, p[0], p[1], p[2], p[2] - FOOT_R))
# static stance torque estimate: tau = J^T (mg/4) per leg
g = 9.81 * m.body_mass.sum() / 4
worst = 0.0
for leg in ("FR", "FL", "RR", "RL"):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "%s_foot" % leg)
    jp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, None, sid)
    tau = jp[:, 6:].T @ np.array([0, 0, g])
    worst = max(worst, np.abs(tau).max())
print("static stance torque (J^T mg/4): worst joint = %.2f Nm  (limit %.1f)" % (worst, TAU))
# ---- envelope callback checks (flat / power / cliff / beyond) ----
import actuator_envelope as env

def act_force(w, cmd):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    jid1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    d.qvel[m.jnt_dofadr[jid1]] = w
    d.ctrl[0] = cmd
    mujoco.mj_forward(m, d)
    return float(d.actuator_force[0])

# without callback: ideal +/-7 fallback
assert abs(act_force(25.0, 7.0) - 7.0) < 1e-9, "fallback broken"
env.install()
cases = [(10.0, 7.0, 7.0),                 # flat (current limit)
         (25.0, 7.0, 160.0 / 25.0),        # power cap 6.4
         (32.0, 7.0, env.B_CLIFF * (env.W0 - 32.0)),  # voltage cliff 3.24
         (34.0, 7.0, 0.0),                 # beyond no-load speed
         (-25.0, 7.0, 7.0),                # braking quadrant: current limit only
         (25.0, -7.0, -7.0)]               # negative cmd at +w: regen, full torque
for w, cmd, want in cases:
    got = act_force(w, cmd)
    assert abs(got - want) < 1e-9, (w, cmd, got, want)
    print("envelope: w=%6.1f cmd=%5.1f -> tau=%6.3f (expect %6.3f) OK" % (w, cmd, got, want))
env.uninstall()

# 0.3 s zero-ctrl integration must stay finite
mujoco.mj_resetDataKeyframe(m, d, 0)
d.ctrl[:] = 0
for _ in range(150):
    mujoco.mj_step(m, d)
assert np.isfinite(d.qpos).all()
print("0.3 s zero-ctrl integration finite OK  trunk z=%.3f" % d.qpos[2])
