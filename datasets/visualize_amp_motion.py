"""Visualize AMP motion datasets in IsaacGym (no trained model needed)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import quat_rotate

import glob
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry
from rsl_rl.datasets.motion_loader import AMPLoader


def main():
    args = get_args()
    if not hasattr(args, 'task') or args.task is None:
        args.task = 'go2_amp'

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    for k in env_cfg.control.stiffness:
        env_cfg.control.stiffness[k] = 0.0
    for k in env_cfg.control.damping:
        env_cfg.control.damping[k] = 0.0

    args.headless = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    motion_dir = os.path.join(os.path.dirname(__file__), 'go2_motion')
    motion_files = sorted(glob.glob(os.path.join(motion_dir, '*')))
    if not motion_files:
        print(f"No motion files found in {motion_dir}")
        return

    loader = AMPLoader(
        device=env.device,
        time_between_frames=env.dt,
        motion_files=motion_files,
    )

    zero_actions = torch.zeros(1, env.num_actions, device=env.device)
    env_ids = torch.tensor([0], device=env.device)
    env_ids_int32 = env_ids.to(dtype=torch.int32)

    for traj_idx in range(loader.num_motions):
        traj_name = loader.trajectory_names[traj_idx]
        traj_len = loader.trajectory_lens[traj_idx]
        print(f"[{traj_idx+1}/{loader.num_motions}] Playing: {traj_name} ({traj_len:.2f}s)")

        t = 0.0
        while t + loader.time_between_frames + env.dt < traj_len:
            frame = loader.get_full_frame_at_time_batch(
                np.array([traj_idx]), np.array([t]))

            root_pos = AMPLoader.get_root_pos_batch(frame)
            root_rot = AMPLoader.get_root_rot_batch(frame)
            env.root_states[env_ids, :3] = root_pos
            env.root_states[env_ids, 3:7] = root_rot
            env.root_states[env_ids, 7:10] = quat_rotate(
                root_rot, AMPLoader.get_linear_vel_batch(frame))
            env.root_states[env_ids, 10:13] = quat_rotate(
                root_rot, AMPLoader.get_angular_vel_batch(frame))
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim, gymtorch.unwrap_tensor(env.root_states),
                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

            env.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frame)
            env.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frame)
            env.gym.set_dof_state_tensor_indexed(
                env.sim, gymtorch.unwrap_tensor(env.dof_state),
                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

            env.step(zero_actions)

            look_at = env.root_states[0, :3].cpu().numpy().astype(np.float64)
            cam_offset = np.array([-2.0, 0.0, 0.9])
            env.set_camera(look_at + cam_offset, look_at)

            t += env.dt

        print(f"  Done: {traj_name}")

    print("All trajectories played.")


if __name__ == '__main__':
    main()
