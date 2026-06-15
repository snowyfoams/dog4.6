# Log per-control-step trot data for the extra-results figures (DOG4.6, retuned gait,
# actuator envelope active). Stepped command 0.1->0.2->0.3; the 0.2 hold is the steady
# window. Reuses RobotModel accessors and controller hooks -- no controller changes.
import os, sys, time
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
LOGDIR = os.path.join(ROOT, "results", "logs")
os.makedirs(LOGDIR, exist_ok=True)

STACKS = {
    "mit":    lambda: LocomotionController(M, stand=True),
    "eth":    lambda: LocomotionControllerETH(M, stand=True),
    "cvxwbc": lambda: LocomotionControllerETH(M, stand=True, high_level="convex", low_level="wbc"),
    "slqjt":  lambda: LocomotionControllerETH(M, stand=True, high_level="slq", low_level="jt"),
}
PROFILE = [(0.0, 1.0), (0.1, 2.0), (0.2, 3.0), (0.3, 2.0)]   # (vx, hold s) after stand

LEG_ORDER = ["FR", "FL", "RR", "RL"]
FOOT_GEOM = {l: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_GEOM, "%s_foot_col" % l) for l in LEG_ORDER}


def measured_grf(d):
    """World-frame ground reaction force per foot (4,3) from actual contacts."""
    g = np.zeros((4, 3))
    cf = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        for li, leg in enumerate(LEG_ORDER):
            fg = FOOT_GEOM[leg]
            if c.geom1 == fg or c.geom2 == fg:
                mujoco.mj_contactForce(M, d, i, cf)
                fw = c.frame.reshape(3, 3).T @ cf[:3]    # contact-frame -> world
                if c.geom1 == fg:                         # force on geom2 from geom1 -> flip
                    fw = -fw
                g[li] += fw
    return g


def log_stack(name, make):
    d = mujoco.MjData(M)
    ctrl = make()
    ctrl.reset(d)
    rob = ctrl.rob
    qa, va, ca = rob.qadr.flatten(), rob.vadr.flatten(), rob.ctrl.flatten()
    for _ in range(int(1.0 / dt)):              # stand
        ctrl.gait.stand = True
        d.ctrl[:] = ctrl(d); mujoco.mj_step(M, d)
    ctrl.gait.stand = False
    rec = {k: [] for k in ("t", "vx_cmd", "q", "qd", "tau_cmd", "tau_app", "foot_pos",
                            "foot_des", "grf", "contact", "com_vel", "rpy", "h",
                            "mpc_ms", "loop_ms")}
    n_solve_prev = len(ctrl.solve_times)
    for vx, hold in PROFILE:
        ctrl.cmd.vx = vx
        for _ in range(int(hold / dt)):
            t0 = time.perf_counter()
            tau = ctrl(d)
            loop_ms = (time.perf_counter() - t0) * 1e3
            contact = ctrl.touch.measure(d).copy()
            foot_des = ctrl.swing.p_des.copy()
            d.ctrl[:] = tau
            mujoco.mj_step(M, d)
            grf = measured_grf(d)
            # MPC solve time only on replan ticks; carry last value otherwise
            mpc_ms = (ctrl.solve_times[-1] * 1e3) if len(ctrl.solve_times) > n_solve_prev else np.nan
            n_solve_prev = len(ctrl.solve_times)
            rec["t"].append(ctrl.t)
            rec["vx_cmd"].append(vx)
            rec["q"].append(d.qpos[qa].copy())
            rec["qd"].append(d.qvel[va].copy())
            rec["tau_cmd"].append(tau[ca].copy())
            rec["tau_app"].append(d.actuator_force[ca].copy())
            rec["foot_pos"].append(rob.foot_pos(d).copy())
            rec["foot_des"].append(foot_des)
            rec["grf"].append(grf)
            rec["contact"].append(contact)
            rec["com_vel"].append(rob.com_vel(d).copy())
            rec["rpy"].append(rob.rpy(d).copy())
            rec["h"].append(float(rob.com(d)[2]))
            rec["mpc_ms"].append(mpc_ms)
            rec["loop_ms"].append(loop_ms)
            if not np.isfinite(d.qpos).all() or np.degrees(np.abs(rob.rpy(d)[:2])).max() > 45 \
                    or rob.com(d)[2] < 0.07:
                break
        else:
            continue
        break
    out = {k: np.asarray(v) for k, v in rec.items()}
    # ctrl/joint index order is [FR,FL,RR,RL] x [abd,thigh,knee] flattened
    out["leg_names"] = np.array(["FR", "FL", "RR", "RL"])
    out["joint_names"] = np.array(["abd", "thigh", "knee"])
    path = os.path.join(LOGDIR, "%s.npz" % name)
    np.savez_compressed(path, **out)
    nseg = int(np.sum((out["vx_cmd"] > 0.19) & (out["vx_cmd"] < 0.21)))
    print("%-7s logged %4d steps (%.1f s), 0.2-window %d steps, mpc mean %.1f ms loop mean %.1f ms" % (
        name, len(out["t"]), len(out["t"]) * dt, nseg,
        np.nanmean(out["mpc_ms"]), np.mean(out["loop_ms"])))


if __name__ == "__main__":
    keys = sys.argv[1:] or list(STACKS)
    for k in keys:
        log_stack(k, STACKS[k])
    print("LOG DONE")
