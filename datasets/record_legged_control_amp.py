#!/usr/bin/env python3
# recording legged_control
import argparse
import json
import os
import sys
from typing import List

import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState


def parse_list(arg: str) -> List[str]:
    return [item.strip() for item in arg.split(',') if item.strip()]


def normalize_quat(qx, qy, qz, qw):
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm <= 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


class AMPRecorder:
    def __init__(self, args):
        self.args = args
        self.frames = []
        self.odom = None
        self.joint_state = None
        self.joint_index = None
        self.last_foot_pos = None
        self.last_stamp = None
        self.root_pos0 = None

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.odom_sub = rospy.Subscriber(args.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.joint_sub = rospy.Subscriber(args.joint_topic, JointState, self.joint_cb, queue_size=1)

        self.cmd_pub = None
        self.cmd_timer = None
        if args.cmd_vel is not None:
            self.cmd_pub = rospy.Publisher(args.cmd_topic, Twist, queue_size=1)
            self.cmd_timer = rospy.Timer(rospy.Duration(1.0 / args.cmd_rate), self.publish_cmd)
            if args.cmd_duration > 0.0:
                rospy.Timer(rospy.Duration(args.cmd_duration), self.stop_cmd, oneshot=True)

        self.sample_timer = rospy.Timer(rospy.Duration(1.0 / args.rate), self.sample)
        if args.duration > 0.0:
            rospy.Timer(rospy.Duration(args.duration), self.stop_recording, oneshot=True)

    def odom_cb(self, msg: Odometry):
        self.odom = msg

    def joint_cb(self, msg: JointState):
        self.joint_state = msg
        if self.joint_index is None:
            name_to_idx = {name: i for i, name in enumerate(msg.name)}
            missing = [name for name in self.args.joint_names if name not in name_to_idx]
            if missing:
                rospy.logwarn_throttle(2.0, f"Waiting for joints: {missing}")
                return
            self.joint_index = [name_to_idx[name] for name in self.args.joint_names]

    def publish_cmd(self, _evt):
        if self.cmd_pub is None:
            return
        twist = Twist()
        twist.linear.x = self.args.cmd_vel[0]
        twist.linear.y = self.args.cmd_vel[1]
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = self.args.cmd_vel[2]
        self.cmd_pub.publish(twist)

    def stop_cmd(self, _evt):
        if self.cmd_timer is not None:
            self.cmd_timer.shutdown()
            self.cmd_timer = None
        if self.cmd_pub is not None:
            zero = Twist()
            self.cmd_pub.publish(zero)

    def sample(self, _evt):
        if self.odom is None or self.joint_state is None or self.joint_index is None:
            return

        stamp = self.odom.header.stamp
        if stamp.to_sec() == 0.0:
            stamp = rospy.Time.now()

        # Lookup foot positions in base frame.
        foot_positions = []
        for frame in self.args.foot_frames:
            try:
                tf_msg = self.tf_buffer.lookup_transform(self.args.base_frame, frame, rospy.Time(0), rospy.Duration(0.05))
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
                return
            t = tf_msg.transform.translation
            foot_positions.extend([t.x, t.y, t.z])

        # Joint positions/velocities in requested order.
        joint_pos = []
        joint_vel = []
        for idx in self.joint_index:
            joint_pos.append(float(self.joint_state.position[idx]))
            if idx < len(self.joint_state.velocity):
                joint_vel.append(float(self.joint_state.velocity[idx]))
            else:
                joint_vel.append(0.0)

        # Root pose.
        pose = self.odom.pose.pose
        root_pos = [pose.position.x, pose.position.y, pose.position.z]
        if self.args.zero_root or self.args.zero_root_xy:
            if self.root_pos0 is None:
                self.root_pos0 = list(root_pos)
            if self.args.zero_root:
                root_pos = [root_pos[0] - self.root_pos0[0], root_pos[1] - self.root_pos0[1], root_pos[2] - self.root_pos0[2]]
            else:
                root_pos = [root_pos[0] - self.root_pos0[0], root_pos[1] - self.root_pos0[1], root_pos[2]]
        if self.args.root_z_offset != 0.0:
            root_pos[2] += float(self.args.root_z_offset)

        qx, qy, qz, qw = normalize_quat(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        root_rot = [qx, qy, qz, qw]

        # Base velocities in base frame.
        twist = self.odom.twist.twist
        lin_vel = [twist.linear.x, twist.linear.y, twist.linear.z]
        ang_vel = [twist.angular.x, twist.angular.y, twist.angular.z]

        # Foot velocities (finite difference in base frame).
        foot_vel = [0.0] * 12
        if self.last_foot_pos is not None and self.last_stamp is not None:
            dt = (stamp - self.last_stamp).to_sec()
            if dt > 1e-6:
                foot_vel = [(foot_positions[i] - self.last_foot_pos[i]) / dt for i in range(12)]

        self.last_foot_pos = list(foot_positions)
        self.last_stamp = stamp

        frame = []
        frame.extend(root_pos)
        frame.extend(root_rot)
        frame.extend(joint_pos)
        frame.extend(foot_positions)
        frame.extend(lin_vel)
        frame.extend(ang_vel)
        frame.extend(joint_vel)
        frame.extend(foot_vel)

        if len(frame) != 61:
            rospy.logerr(f"Invalid frame length {len(frame)} (expected 61)")
            return

        self.frames.append(frame)
        if self.args.max_frames > 0 and len(self.frames) >= self.args.max_frames:
            self.stop_recording(None)

    def stop_recording(self, _evt):
        if self.sample_timer is not None:
            self.sample_timer.shutdown()
            self.sample_timer = None
        self.write_dataset()
        rospy.signal_shutdown("recording complete")

    def write_dataset(self):
        os.makedirs(os.path.dirname(self.args.out), exist_ok=True)
        data = {
            "LoopMode": self.args.loop_mode,
            "FrameDuration": float(1.0 / self.args.rate),
            "EnableCycleOffsetPosition": bool(self.args.enable_cycle_offset_position),
            "EnableCycleOffsetRotation": bool(self.args.enable_cycle_offset_rotation),
            "MotionWeight": float(self.args.motion_weight),
            "Frames": self.frames,
        }
        with open(self.args.out, "w") as f:
            json.dump(data, f, indent=2)
        rospy.loginfo(f"Saved {len(self.frames)} frames to {self.args.out}")


def main():
    parser = argparse.ArgumentParser(description="Record legged_control motion for AMP dataset.")
    parser.add_argument("--out", type=str, default="/home/ak1ra/legged_control/rl_amp/datasets/my_motions_go2/go2_motion.json")
    parser.add_argument("--rate", type=float, default=50.0, help="Recording rate in Hz.")
    parser.add_argument("--duration", type=float, default=0.0, help="Recording duration in seconds (0 = manual stop).")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited).")
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--zero-root", action="store_true", help="Zero root position at start (x,y,z).")
    parser.add_argument("--zero-root-xy", action="store_true", help="Zero root x/y only; keep z.")
    parser.add_argument("--root-z-offset", type=float, default=0.0, help="Add constant offset to root z.")
    parser.add_argument("--loop-mode", type=str, default="Wrap", help="LoopMode field in AMP file.")
    parser.add_argument("--enable-cycle-offset-position", action="store_true", help="EnableCycleOffsetPosition field.")
    parser.add_argument("--enable-cycle-offset-rotation", action="store_true", help="EnableCycleOffsetRotation field.")

    parser.add_argument("--odom-topic", type=str, default="/odom")
    parser.add_argument("--joint-topic", type=str, default="/joint_states")
    parser.add_argument("--base-frame", type=str, default="base")
    parser.add_argument(
        "--foot-frames",
        type=parse_list,
        default=parse_list("LF_FOOT,RF_FOOT,LH_FOOT,RH_FOOT"),
        help="Comma-separated foot frames in order: FL,FR,RL,RR",
    )
    parser.add_argument(
        "--joint-names",
        type=parse_list,
        default=parse_list("LF_HAA,LF_HFE,LF_KFE,RF_HAA,RF_HFE,RF_KFE,LH_HAA,LH_HFE,LH_KFE,RH_HAA,RH_HFE,RH_KFE"),
        help="Comma-separated joint names in order: FL,FR,RL,RR (hip,thigh,calf).",
    )

    parser.add_argument("--cmd-topic", type=str, default="/cmd_vel")
    parser.add_argument(
        "--cmd-vel",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "WZ"),
        help="Optional cmd_vel to publish while recording.",
    )
    parser.add_argument("--cmd-rate", type=float, default=20.0)
    parser.add_argument("--cmd-duration", type=float, default=0.0)

    args = parser.parse_args()

    rospy.init_node("legged_control_amp_recorder", anonymous=True)
    recorder = AMPRecorder(args)
    rospy.loginfo("AMP recording started.")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
