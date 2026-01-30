#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data as pd

POS_SIZE = 3
ROT_SIZE = 4
JOINT_POS_SIZE = 12
TAR_TOE_POS_LOCAL_SIZE = 12
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
JOINT_VEL_SIZE = 12
TAR_TOE_VEL_LOCAL_SIZE = 12

ROOT_POS_START = 0
ROOT_ROT_START = ROOT_POS_START + POS_SIZE
JOINT_POS_START = ROOT_ROT_START + ROT_SIZE
TAR_TOE_POS_START = JOINT_POS_START + JOINT_POS_SIZE
LINEAR_VEL_START = TAR_TOE_POS_START + TAR_TOE_POS_LOCAL_SIZE
ANGULAR_VEL_START = LINEAR_VEL_START + LINEAR_VEL_SIZE
JOINT_VEL_START = ANGULAR_VEL_START + ANGULAR_VEL_SIZE
TAR_TOE_VEL_START = JOINT_VEL_START + JOINT_VEL_SIZE

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_motion(path: Path):
    motion = json.loads(path.read_text())
    frames = np.asarray(motion["Frames"], dtype=np.float32)
    frame_duration = float(motion.get("FrameDuration", 0.02))
    weight = float(motion.get("MotionWeight", 1.0))
    return frames, frame_duration, weight


def reorder_legs(data: np.ndarray, order: str) -> np.ndarray:
    if order == "fl_fr_rl_rr":
        return data
    if order == "fr_fl_rr_rl":
        legs = np.split(data, 4)
        return np.concatenate([legs[1], legs[0], legs[3], legs[2]])
    raise ValueError(f"Unsupported order: {order}")


def load_joint_limits(urdf_path: Path):
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    robot = p.loadURDF(str(urdf_path), useFixedBase=True)
    limits = {}
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name = info[1].decode("utf-8")
        jtype = info[2]
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            limits[name] = (info[8], info[9])
    p.disconnect(cid)
    return limits


def summarize_stats(name, values):
    return {
        "name": name,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Quantitative checks for Go2 AMP motions.")
    parser.add_argument("paths", nargs="+", type=Path, help="Motion files or directories")
    parser.add_argument("--urdf", type=Path, default=repo_root() / "legged_gym/resources/robots/go2/urdf/go2.urdf")
    parser.add_argument("--order", choices=["fl_fr_rl_rr", "fr_fl_rr_rl"], default="fl_fr_rl_rr",
                        help="Leg ordering in the dataset")
    parser.add_argument("--tol", type=float, default=1e-3, help="Joint limit tolerance")
    args = parser.parse_args()

    files = []
    for pth in args.paths:
        if pth.is_dir():
            files.extend(sorted(p for p in pth.iterdir() if p.is_file()))
        else:
            files.append(pth)

    limits = load_joint_limits(args.urdf)
    joint_limits = [limits.get(name, (-np.inf, np.inf)) for name in JOINT_NAMES]

    for path in files:
        frames, frame_duration, weight = load_motion(path)
        if frames.ndim != 2:
            print(f"{path}: invalid frame shape {frames.shape}")
            continue

        if frames.shape[1] != TAR_TOE_VEL_START + TAR_TOE_VEL_LOCAL_SIZE:
            print(f"{path}: unexpected frame width {frames.shape[1]} (expected 61)")

        finite_mask = np.isfinite(frames)
        finite_ratio = float(np.mean(finite_mask))

        root_pos = frames[:, ROOT_POS_START:ROOT_ROT_START]
        root_rot = frames[:, ROOT_ROT_START:JOINT_POS_START]
        joint_pos = frames[:, JOINT_POS_START:TAR_TOE_POS_START]
        toe_pos = frames[:, TAR_TOE_POS_START:LINEAR_VEL_START]
        lin_vel = frames[:, LINEAR_VEL_START:ANGULAR_VEL_START]
        ang_vel = frames[:, ANGULAR_VEL_START:JOINT_VEL_START]
        joint_vel = frames[:, JOINT_VEL_START:TAR_TOE_VEL_START]

        joint_pos = reorder_legs(joint_pos, args.order)
        toe_pos = reorder_legs(toe_pos, args.order)
        joint_vel = reorder_legs(joint_vel, args.order)

        quat_norm = np.linalg.norm(root_rot, axis=1)
        quat_dev = np.abs(quat_norm - 1.0)

        joint_low = np.array([jl for jl, _ in joint_limits], dtype=np.float32)
        joint_high = np.array([jh for _, jh in joint_limits], dtype=np.float32)
        below = joint_pos < (joint_low - args.tol)
        above = joint_pos > (joint_high + args.tol)
        limit_viol = np.logical_or(below, above)
        limit_viol_ratio = float(np.mean(limit_viol))
        max_viol = 0.0
        if np.any(limit_viol):
            max_viol = float(np.max(np.maximum(joint_low - joint_pos, joint_pos - joint_high)))

        root_speed = np.linalg.norm(lin_vel, axis=1)
        ang_speed = np.linalg.norm(ang_vel, axis=1)
        joint_speed = np.linalg.norm(joint_vel.reshape(joint_vel.shape[0], 4, 3), axis=2).reshape(-1)

        toe_pos = toe_pos.reshape(-1, 4, 3)
        toe_z = toe_pos[:, :, 2].reshape(-1)

        duration = (frames.shape[0] - 1) * frame_duration

        print(f"\n{path}")
        print(f"  frames: {frames.shape[0]} | duration: {duration:.3f}s | dt: {frame_duration:.4f}s | weight: {weight}")
        print(f"  finite_ratio: {finite_ratio:.4f} | quat_norm_dev max: {float(np.max(quat_dev)):.4e}")
        print(f"  base_z: min={root_pos[:,2].min():.3f} mean={root_pos[:,2].mean():.3f} max={root_pos[:,2].max():.3f}")
        print(f"  joint_limit_viol: {limit_viol_ratio*100:.2f}% | max_violation(rad): {max_viol:.4f}")

        rs = summarize_stats("root_speed", root_speed)
        av = summarize_stats("ang_speed", ang_speed)
        js = summarize_stats("joint_speed", joint_speed)
        tz = summarize_stats("toe_z_local", toe_z)
        print(f"  root_speed(m/s): min={rs['min']:.3f} mean={rs['mean']:.3f} p95={rs['p95']:.3f} max={rs['max']:.3f}")
        print(f"  ang_speed(rad/s): min={av['min']:.3f} mean={av['mean']:.3f} p95={av['p95']:.3f} max={av['max']:.3f}")
        print(f"  joint_speed(rad/s): min={js['min']:.3f} mean={js['mean']:.3f} p95={js['p95']:.3f} max={js['max']:.3f}")
        print(f"  toe_z_local(m): min={tz['min']:.3f} mean={tz['mean']:.3f} p95={tz['p95']:.3f} max={tz['max']:.3f}")


if __name__ == "__main__":
    main()
