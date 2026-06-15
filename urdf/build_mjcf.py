# Build dog.xml (MJCF) from the same Fusion-extracted data as build_urdf.py.
# Units: m, kg, kg*m^2. Meshes are mm STLs -> scale 0.001.
import json, math, os

HERE = os.path.dirname(__file__)
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

JOINTS = json.loads(r'''
{"FR_hip_joint": {"parent": "trunk", "child": "FR_hip", "origin_cm": [26.9978, -6.0, 0.0], "axis": [1, 0, 0]},
"FR_thigh_joint": {"parent": "FR_hip", "child": "FR_thigh", "origin_cm": [26.9978, -11.6109, 0.0], "axis": [0, 1, 0]},
"FR_calf_joint": {"parent": "FR_thigh", "child": "FR_calf", "origin_cm": [20.9978, -11.6109, -10.3923], "axis": [0, 1, 0]},
"FL_hip_joint": {"parent": "trunk", "child": "FL_hip", "origin_cm": [26.9978, 6.0, 0.0], "axis": [1, 0, 0]},
"FL_thigh_joint": {"parent": "FL_hip", "child": "FL_thigh", "origin_cm": [26.9978, 11.6109, 0.0], "axis": [0, 1, 0]},
"FL_calf_joint": {"parent": "FL_thigh", "child": "FL_calf", "origin_cm": [20.9978, 11.6109, -10.3923], "axis": [0, 1, 0]},
"RL_hip_joint": {"parent": "trunk", "child": "RL_hip", "origin_cm": [-26.9978, 6.0, 0.0], "axis": [1, 0, 0]},
"RL_thigh_joint": {"parent": "RL_hip", "child": "RL_thigh", "origin_cm": [-26.9978, 11.6109, 0.0], "axis": [0, 1, 0]},
"RL_calf_joint": {"parent": "RL_thigh", "child": "RL_calf", "origin_cm": [-32.9978, 11.6109, -10.3923], "axis": [0, 1, 0]},
"RR_hip_joint": {"parent": "trunk", "child": "RR_hip", "origin_cm": [-26.9978, -6.0, 0.0], "axis": [1, 0, 0]},
"RR_thigh_joint": {"parent": "RR_hip", "child": "RR_thigh", "origin_cm": [-26.9978, -11.6109, 0.0], "axis": [0, 1, 0]},
"RR_calf_joint": {"parent": "RR_thigh", "child": "RR_calf", "origin_cm": [-32.9978, -11.6109, -10.3923], "axis": [0, 1, 0]}}
''')

MESHES = json.load(open(os.path.join(HERE, "mesh_transforms.json")))

LINK_MESHES = {"trunk": ["base_link", "FR_abd_motor", "FL_abd_motor", "RL_abd_motor", "RR_abd_motor"]}
for leg in ("FR", "FL", "RL", "RR"):
    LINK_MESHES[leg+"_hip"]   = [leg+"_hip", leg+"_hip_motor"]
    LINK_MESHES[leg+"_thigh"] = [leg+"_thigh", leg+"_knee_motor"]
    LINK_MESHES[leg+"_calf"]  = [leg+"_calf", leg+"_foot"]

FRAME = {"trunk": [0.0, 0.0, 0.0]}
CHILDREN = {}
for jn, j in JOINTS.items():
    FRAME[j["child"]] = j["origin_cm"]
    CHILDREN.setdefault(j["parent"], []).append((jn, j["child"]))

# Joint params. PLACEHOLDERS for range/ctrl — fill from MG5010-i10 datasheet.
LIM = {"hip": (-0.80, 0.80), "thigh": (-2.60, 2.60), "calf": (-2.60, 2.60)}
CTRL = 10.0
# rotor inertia 850 g*cm^2 = 8.5e-5 kg*m^2; reflected at joint assuming i10 = 10:1 gearbox:
ARMATURE = 8.5e-5 * 10**2   # = 8.5e-3 kg*m^2  (adjust if gear ratio differs)

def fmt(v):
    s = "%.8f" % v
    s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"

def quat_from_R(R):
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2][1] - R[1][2]) / s
        y = (R[0][2] - R[2][0]) / s
        z = (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        w = (R[2][1] - R[1][2]) / s; x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s; z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        w = (R[0][2] - R[2][0]) / s; x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s; z = (R[1][2] + R[2][1]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        w = (R[1][0] - R[0][1]) / s; x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s; z = 0.25 * s
    n = math.sqrt(w*w + x*x + y*y + z*z)
    return w/n, x/n, y/n, z/n

L = []
def emit(ind, s): L.append("  " * ind + s)

def body_xml(name, ind):
    f = FRAME[name]
    lk = LINKS[name]
    com = [(lk["com_cm"][i] - f[i]) * 0.01 for i in range(3)]
    ixx, iyy, izz, ixy, iyz, ixz = (v * 1e-4 for v in lk["I_com_kgcm2"])
    if name == "trunk":
        emit(ind, '<body name="trunk" pos="0 0 0.2243">')
        emit(ind + 1, '<freejoint name="root"/>')
    else:
        pf = FRAME[[j["parent"] for j in JOINTS.values() if j["child"] == name][0]]
        pos = [(f[i] - pf[i]) * 0.01 for i in range(3)]
        emit(ind, '<body name="%s" pos="%s %s %s">' % (name, *[fmt(v) for v in pos]))
        jn = [n for n, j in JOINTS.items() if j["child"] == name][0]
        kind = "hip" if jn.endswith("_hip_joint") else ("thigh" if "thigh" in jn else "calf")
        lo, hi = LIM[kind]
        ax = JOINTS[jn]["axis"]
        emit(ind + 1, '<joint name="%s" axis="%d %d %d" range="%s %s"/>' % (jn, ax[0], ax[1], ax[2], fmt(lo), fmt(hi)))
    emit(ind + 1, '<inertial pos="%s %s %s" mass="%s" fullinertia="%s %s %s %s %s %s"/>' % (
        fmt(com[0]), fmt(com[1]), fmt(com[2]), fmt(lk["mass"]),
        fmt(ixx), fmt(iyy), fmt(izz), fmt(ixy), fmt(ixz), fmt(iyz)))  # MJCF order: ixx iyy izz ixy ixz iyz
    for mesh in LINK_MESHES[name]:
        M = MESHES[mesh]
        t = [(M["t_cm"][i] - f[i]) * 0.01 for i in range(3)]
        q = quat_from_R(M["R"])
        emit(ind + 1, '<geom mesh="%s" pos="%s %s %s" quat="%s %s %s %s"/>' % (
            mesh, fmt(t[0]), fmt(t[1]), fmt(t[2]), fmt(q[0]), fmt(q[1]), fmt(q[2]), fmt(q[3])))
    for jn, child in CHILDREN.get(name, []):
        body_xml(child, ind + 1)
    emit(ind, '</body>')

emit(0, '<mujoco model="dog">')
emit(1, '<!-- Generated from Fusion 360 design "DOG" (2026-06-12). Zero pose = standing.')
emit(1, '     armature = rotor inertia 8.5e-5 kg*m^2 x gear^2 (assumed 10:1 for MG5010-i10 -> 8.5e-3).')
emit(1, '     Joint ranges and ctrlrange are PLACEHOLDERS - fill from the MG5010-i10 datasheet. -->')
emit(1, '<compiler angle="radian" meshdir="meshes" autolimits="true"/>')
emit(1, '<option timestep="0.002"/>')
emit(1, '<default>')
emit(2, '<joint type="hinge" damping="0.01" frictionloss="0.1" armature="%s"/>' % fmt(ARMATURE))
emit(2, '<geom type="mesh" friction="0.6 0.01 0.01"/>')
emit(2, '<motor ctrlrange="-%s %s"/>' % (fmt(CTRL), fmt(CTRL)))
emit(1, '</default>')
emit(1, '<asset>')
emit(2, '<texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.9 0.9 0.9" width="64" height="64"/>')
emit(2, '<texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.35 0.4" rgb2="0.4 0.45 0.5" width="256" height="256"/>')
emit(2, '<material name="grid" texture="grid" texrepeat="4 4" reflectance="0.1"/>')
for ms in LINK_MESHES.values():
    for m in ms:
        emit(2, '<mesh name="%s" file="%s.stl" scale="0.001 0.001 0.001"/>' % (m, m))
emit(1, '</asset>')
emit(1, '<worldbody>')
emit(2, '<light pos="0 0 2" dir="0 0 -1" directional="true"/>')
emit(2, '<geom name="floor" type="plane" size="3 3 0.05" material="grid"/>')
body_xml("trunk", 2)
emit(1, '</worldbody>')
emit(1, '<actuator>')
for jn in JOINTS:
    emit(2, '<motor name="%s" joint="%s" gear="1"/>' % (jn.replace("_joint", ""), jn))
emit(1, '</actuator>')
emit(1, '<keyframe>')
emit(2, '<key name="home" qpos="0 0 0.2243 1 0 0 0  0 0 0  0 0 0  0 0 0  0 0 0"/>')
emit(1, '</keyframe>')
emit(0, '</mujoco>')

path = os.path.join(HERE, "dog.xml")
open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", path)
