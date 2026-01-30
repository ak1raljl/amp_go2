#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

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


def load_motion(path: Path):
    data = json.loads(path.read_text())
    frames = np.asarray(data["Frames"], dtype=np.float32)
    return data, frames


def build_bad_mask(frames, max_lin, max_ang, max_joint, quat_tol):
    if frames.ndim != 2 or frames.shape[1] < TAR_TOE_VEL_START + TAR_TOE_VEL_LOCAL_SIZE:
        return np.ones((frames.shape[0],), dtype=bool), "bad_shape"

    finite_mask = np.isfinite(frames).all(axis=1)

    lin = frames[:, LINEAR_VEL_START:ANGULAR_VEL_START]
    ang = frames[:, ANGULAR_VEL_START:JOINT_VEL_START]
    jv = frames[:, JOINT_VEL_START:TAR_TOE_VEL_START]

    lin_bad = np.max(np.abs(lin), axis=1) > max_lin
    ang_bad = np.max(np.abs(ang), axis=1) > max_ang
    jv_bad = np.max(np.abs(jv), axis=1) > max_joint

    quat = frames[:, ROOT_ROT_START:JOINT_POS_START]
    quat_norm = np.linalg.norm(quat, axis=1)
    quat_bad = np.abs(quat_norm - 1.0) > quat_tol

    bad = ~(finite_mask) | lin_bad | ang_bad | jv_bad | quat_bad
    return bad, None


def split_segments(mask):
    segments = []
    start = None
    for i, good in enumerate(mask):
        if good and start is None:
            start = i
        elif not good and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def save_motion(dst_path, template, frames):
    out = dict(template)
    out["Frames"] = frames.tolist()
    dst_path.write_text(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Clean Go2 AMP motion dataset.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-lin", type=float, default=1.5, help="Max abs base linear velocity (m/s)")
    parser.add_argument("--max-ang", type=float, default=2.0, help="Max abs base angular velocity (rad/s)")
    parser.add_argument("--max-joint-vel", type=float, default=10.0, help="Max abs joint velocity (rad/s)")
    parser.add_argument("--quat-tol", type=float, default=1e-3, help="Quaternion norm tolerance")
    parser.add_argument("--min-frames", type=int, default=60, help="Minimum frames per segment")
    parser.add_argument("--drop-bad-ratio", type=float, default=0.2, help="Drop file if bad ratio exceeds this")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.input_dir.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"No files found in {args.input_dir}")

    for path in files:
        template, frames = load_motion(path)
        bad_mask, err = build_bad_mask(
            frames, args.max_lin, args.max_ang, args.max_joint_vel, args.quat_tol
        )
        if err:
            print(f"{path.name}: skipped ({err})")
            continue
        good_mask = ~bad_mask
        bad_ratio = 1.0 - float(np.mean(good_mask))

        if bad_ratio > args.drop_bad_ratio:
            print(f"{path.name}: dropped (bad_ratio={bad_ratio:.3f})")
            continue

        segments = split_segments(good_mask)
        segments = [seg for seg in segments if (seg[1] - seg[0]) >= args.min_frames]

        if not segments:
            print(f"{path.name}: no valid segments")
            continue

        stem = path.stem
        if len(segments) == 1:
            out_path = args.output_dir / f"{stem}.txt"
            s, e = segments[0]
            save_motion(out_path, template, frames[s:e])
            print(f"{path.name}: kept {e - s} frames -> {out_path.name}")
        else:
            for i, (s, e) in enumerate(segments):
                out_path = args.output_dir / f"{stem}_c{i+1}.txt"
                save_motion(out_path, template, frames[s:e])
                print(f"{path.name}: kept {e - s} frames -> {out_path.name}")


if __name__ == "__main__":
    main()
