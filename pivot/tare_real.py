#!/usr/bin/env python3
"""Plan safe RB5 paths and record empty Robotiq wrench in three orientations."""

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

from aft_tare import Aft200Sensor, TareTable, stable_wrench

PIVOT_WORKDIR = Path(os.environ.get("PIVOT_WORKDIR", "")).expanduser()
if not (PIVOT_WORKDIR / "density_id_drake.py").is_file():
    raise RuntimeError("set PIVOT_WORKDIR to the PIVOT my_work directory")
sys.path.insert(0, str(PIVOT_WORKDIR.resolve()))

import density_id_drake as alg
import density_id_objects as obj
import hardware as hw
from path_planning import ArmPathPlanner
import robot_scene as scene


def empty_tool_spec():
    part = obj.Part("tool", (2.0, 78.0, 2.0), 0.01, 1000.0,
                    (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (0.3, 0.3, 0.3, 1.0), grasp_width_mm=78.0)
    return obj.ObjectSpec("tare_tool", "empty Robotiq", [part], [],
                          (0.0, 0.0, 0.0))


def nearest_equivalent(q, reference, lower, upper):
    result = []
    for value, current, lo, hi in zip(q, reference, lower, upper):
        choices = [value + 2 * np.pi * k for k in range(-2, 3)
                   if lo <= value + 2 * np.pi * k <= hi]
        if not choices:
            raise RuntimeError("no equivalent joint angle inside the limits")
        result.append(min(choices, key=lambda item: abs(item - current)))
    return np.asarray(result)


def require_ready(data, require_real=False):
    state = data.request_data(2.0)
    if state is None:
        raise TimeoutError("no RB5 state response")
    status = state.sdata
    checks = {
        "activation": status.init_state_info == 6 and status.init_error == 0,
        "arm_power": ((status.information_chunk_1 >> 6) & 1) == 1,
        "idle": status.robot_state == 1 and status.task_state == 1,
        "collision_detection": status.collision_detect_onoff == 1,
        "freedrive_off": status.is_freedrive_mode == 0,
        "no_fault": not any((status.op_stat_collision_occur, status.op_stat_sos_flag,
                              status.op_stat_soft_estop_occur, status.op_stat_ems_flag)),
        "real_mode": not require_real or status.real_vs_simulation_mode == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("RB5 safety check failed: " + ", ".join(failed))
    return np.deg2rad(np.asarray(status.jnt_ang[:6], dtype=float))


def plan_paths(current, clearance_m, max_iters):
    checker = scene.PoseChecker(
        empty_tool_spec(), densities=[1000.0], joint_limits_rad=[],
        min_distance_m=clearance_m, gripper="robotiq2f85",
        ik_restarts=30, seed_q=current)
    indices = [joint.position_start() for joint in checker.arm_joints]
    fixed = checker.plant.GetPositions(checker.context).copy()
    planner = ArmPathPlanner(checker.plant, checker.context,
                             checker.arm_joints, clearance_m, fixed, seed=11)
    start = nearest_equivalent(current, np.clip(current, planner.lower, planner.upper),
                               planner.lower, planner.upper)
    for joint, value in zip(checker.arm_joints, start):
        fixed[joint.position_start()] = value
    planner.set_fixed(fixed)

    targets = []
    for g_hat in alg.G_DIRS:
        checker._last_solution = None
        full = checker.solve_robust(np.array([]), g_hat)
        if full is None:
            raise RuntimeError(f"IK failed for gravity direction {g_hat}")
        targets.append(np.asarray(full)[indices])

    paths, clearances, state = [], [], start
    for target in targets + [start]:
        target = nearest_equivalent(target, state, planner.lower, planner.upper)
        path = planner.plan(state, target, max_iters=max_iters)
        if path is None:
            raise RuntimeError("no collision-free path from the current pose")
        if not np.allclose(path[0], state) or not np.allclose(path[-1], target):
            raise RuntimeError("planned path endpoints are invalid")
        measured = planner.path_clearance(path, samples_per_edge=60)
        if measured < clearance_m * 0.9 - 1e-6:
            raise RuntimeError(f"path clearance is too small: {measured*1000:.2f} mm")
        paths.append(path)
        clearances.append(measured)
        state = target
    return start, paths, clearances


def path_duration(path, speed_deg_s):
    max_delta = max(float(np.max(np.abs(np.degrees(b - a))))
                    for a, b in zip(path[:-1], path[1:]))
    return len(path) * max(10.0, 2.0 * max_delta / speed_deg_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.50.51")
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/aft_tare_current.json"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--aft-hz", type=float, default=50.0)
    parser.add_argument("--speed-deg-s", type=float, default=3.0)
    parser.add_argument("--clearance-mm", type=float, default=10.0)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite and not args.plan_only:
        parser.error(f"refusing to overwrite {args.output}")

    rbpodo = __import__("rbpodo")
    data = rbpodo.CobotData(args.robot_ip)
    current = require_ready(data)
    start, paths, clearances = plan_paths(
        current, args.clearance_mm / 1000.0, args.max_iters)
    print("start joints [deg]:", np.round(np.degrees(start), 2).tolist())
    for label, path, clearance in zip(("g-down", "g-x", "g-y", "return"),
                                      paths, clearances):
        print(f"{label}: {len(path)} waypoints, clearance {clearance*1000:.2f} mm, "
              f"goal {np.round(np.degrees(path[-1]), 1).tolist()} deg")
    if args.plan_only:
        print("PLAN ONLY: no robot command")
        return

    unchanged = require_ready(data)
    error = np.degrees(np.abs(np.arctan2(np.sin(unchanged - current),
                                         np.cos(unchanged - current))))
    if np.max(error) > 0.05:
        raise RuntimeError("robot pose changed while planning")

    robot = hw.Rb5Driver(args.robot_ip, enable_motion=True,
                         max_speed_deg_s=args.speed_deg_s,
                         max_accel_deg_s2=2 * args.speed_deg_s)
    sensor = Aft200Sensor(args.robot_ip, args.aft_hz)
    tare = TareTable()
    try:
        deadline = time.monotonic() + 3.0
        while data.request_data(2.0).sdata.real_vs_simulation_mode != 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("RB5 did not enter Real mode within 3 seconds")
            time.sleep(0.1)
        require_ready(data, require_real=True)
        for g_hat, path in zip(alg.G_DIRS, paths[:3]):
            robot.follow(path, path_duration(path, args.speed_deg_s))
            robot.stop()
            time.sleep(1.0)
            raw = stable_wrench(sensor, args.samples)
            tare.record(g_hat, raw)
            print(f"tare g={g_hat.tolist()}: {np.round(raw, 4).tolist()}")
        robot.follow(paths[3], path_duration(paths[3], args.speed_deg_s))
        robot.stop()
    finally:
        robot.stop()

    tare.save(args.output)
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
