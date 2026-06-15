# Build dog.urdf from Fusion-extracted data.
# Units in: cm (positions), kg, kg*cm^2 (inertia, tensor convention), mm (STL meshes)
# Units out: m, kg, kg*m^2; mesh scale 0.001
import json, math, os

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

MESHES = json.load(open(os.path.join(os.path.dirname(__file__), "mesh_transforms.json")))

LINK_MESHES = {"trunk": ["base_link", "FR_abd_motor", "FL_abd_motor", "RL_abd_motor", "RR_abd_motor"]}
for leg in ("FR", "FL", "RL", "RR"):
    LINK_MESHES[leg+"_hip"]   = [leg+"_hip", leg+"_hip_motor"]
    LINK_MESHES[leg+"_thigh"] = [leg+"_thigh", leg+"_knee_motor"]
    LINK_MESHES[leg+"_calf"]  = [leg+"_calf", leg+"_foot"]

# world position of each link frame (cm)
FRAME = {"trunk": [0.0, 0.0, 0.0]}
for jn, j in JOINTS.items():
    FRAME[j["child"]] = j["origin_cm"]

def rpy_from_R(R):
    # URDF rpy: R = Rz(y)*Ry(p)*Rx(r)
    p = -math.asin(max(-1.0, min(1.0, R[2][0])))
    r = math.atan2(R[2][1], R[2][2])
    y = math.atan2(R[1][0], R[0][0])
    # verify
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rc = [[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
          [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
          [-sp,   cp*sr,            cp*cr]]
    err = max(abs(Rc[i][k]-R[i][k]) for i in range(3) for k in range(3))
    assert err < 1e-6, "rpy extraction err %g" % err
    return r, p, y

def fmt(v): return ("%.8f" % v).rstrip("0").rstrip(".") or "0"

# Placeholder limits — replace with MG5010-i10 datasheet values!
LIM = {"hip": (-0.80, 0.80), "thigh": (-2.60, 2.60), "calf": (-2.60, 2.60)}
EFFORT, VEL = 10.0, 25.0

out = []
out.append('<?xml version="1.0"?>')
out.append('<!-- Generated from Fusion 360 design "DOG" (2026-06-12).')
out.append('     Zero pose = standing: thigh 30 deg behind vertical, foot under hip, stance height ~211 mm.')
out.append('     Frames: X forward, Y left, Z up. All link frames axis-aligned with trunk at zero pose.')
out.append('     Motors: MG5010-i10, 420 g each, rotor inertia 850 g*cm^2 = 8.5e-5 kg*m^2 (rotor side;')
out.append('     reflected at joint = 8.5e-5 * gear_ratio^2 - set as armature in MuJoCo / <implicitSpringDamper> equiv).')
out.append('     Joint limits/effort/velocity are PLACEHOLDERS - fill from the MG5010-i10 datasheet. -->')
out.append('<robot name="dog">')

def link_xml(name):
    L = LINKS[name]
    f = FRAME[name]
    com = [(L["com_cm"][i] - f[i]) * 0.01 for i in range(3)]
    I = [v * 1e-4 for v in L["I_com_kgcm2"]]
    s = ['  <link name="%s">' % name]
    s.append('    <inertial>')
    s.append('      <origin xyz="%s %s %s" rpy="0 0 0"/>' % tuple(fmt(c) for c in com))
    s.append('      <mass value="%s"/>' % fmt(L["mass"]))
    s.append('      <inertia ixx="%s" iyy="%s" izz="%s" ixy="%s" iyz="%s" ixz="%s"/>' % tuple(fmt(v) for v in I))
    s.append('    </inertial>')
    for mesh in LINK_MESHES[name]:
        M = MESHES[mesh]
        t = [(M["t_cm"][i] - f[i]) * 0.01 for i in range(3)]
        r, p, y = rpy_from_R(M["R"])
        org = '      <origin xyz="%s %s %s" rpy="%s %s %s"/>' % tuple(fmt(v) for v in (t[0], t[1], t[2], r, p, y))
        geo = '      <geometry><mesh filename="meshes/%s.stl" scale="0.001 0.001 0.001"/></geometry>' % mesh
        s.append('    <visual>'); s.append(org); s.append(geo); s.append('    </visual>')
        s.append('    <collision>'); s.append(org); s.append(geo); s.append('    </collision>')
    s.append('  </link>')
    return "\n".join(s)

def joint_xml(name):
    J = JOINTS[name]
    pf, cf = FRAME[J["parent"]], FRAME[J["child"]]
    o = [(cf[i] - pf[i]) * 0.01 for i in range(3)]
    kind = "hip" if name.endswith("_hip_joint") else ("thigh" if "thigh" in name else "calf")
    lo, hi = LIM[kind]
    s = ['  <joint name="%s" type="revolute">' % name]
    s.append('    <origin xyz="%s %s %s" rpy="0 0 0"/>' % tuple(fmt(v) for v in o))
    s.append('    <parent link="%s"/>' % J["parent"])
    s.append('    <child link="%s"/>' % J["child"])
    s.append('    <axis xyz="%d %d %d"/>' % tuple(J["axis"]))
    s.append('    <limit lower="%s" upper="%s" effort="%s" velocity="%s"/>' % (fmt(lo), fmt(hi), fmt(EFFORT), fmt(VEL)))
    s.append('    <dynamics damping="0.01" friction="0.1"/>')
    s.append('  </joint>')
    return "\n".join(s)

out.append(link_xml("trunk"))
for leg in ("FR", "FL", "RL", "RR"):
    out.append(joint_xml(leg+"_hip_joint"));   out.append(link_xml(leg+"_hip"))
    out.append(joint_xml(leg+"_thigh_joint")); out.append(link_xml(leg+"_thigh"))
    out.append(joint_xml(leg+"_calf_joint"));  out.append(link_xml(leg+"_calf"))
out.append('</robot>')

path = os.path.join(os.path.dirname(__file__), "dog.urdf")
open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")

# ---- validation ----
import xml.etree.ElementTree as ET
tree = ET.parse(path)
robot = tree.getroot()
links = robot.findall("link"); joints = robot.findall("joint")
total = sum(float(l.find("inertial/mass").get("value")) for l in links)
print("links=%d joints=%d total_mass=%.3f kg" % (len(links), len(joints), total))
# tree check: every link except trunk is a child exactly once
children = [j.find("child").get("link") for j in joints]
parents = set(j.find("parent").get("link") for j in joints)
names = set(l.get("name") for l in links)
assert len(set(children)) == len(children) == len(names) - 1, "tree malformed"
assert "trunk" not in children and parents <= names, "root/parent error"
# inertia positive-definite-ish checks
for l in links:
    i = l.find("inertial/inertia")
    xx, yy, zz = (float(i.get(k)) for k in ("ixx", "iyy", "izz"))
    assert xx > 0 and yy > 0 and zz > 0, l.get("name")
    assert xx + yy >= zz * 0.999 and yy + zz >= xx * 0.999 and xx + zz >= yy * 0.999, "triangle ineq " + l.get("name")
# mesh files exist
md = os.path.join(os.path.dirname(__file__), "meshes")
for ln, ms in LINK_MESHES.items():
    for m in ms:
        assert os.path.exists(os.path.join(md, m + ".stl")), m
print("URDF validation passed: tree OK, inertias OK, all 29 meshes present")
print("wrote", path)
