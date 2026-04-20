import sys
from pathlib import Path
PATH_PARENT = Path(__file__).parent
sys.path.append(str(PATH_PARENT))
from utils import MujocoRenderUtils

import os
import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import os
import imageio
from argparse import ArgumentParser
import pygame
from matplotlib import pyplot as plt

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_rotate_inverse(q, v):
    q = np.array(q, np.float32)
    v = np.array(v, np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_keyboard_command(keys, cmd_limits):
    cmd_x = 0.0
    cmd_y = 0.0
    cmd_yaw = 0.0
    if keys[pygame.K_w]:    cmd_x += cmd_limits[0, 1]
    if keys[pygame.K_s]:    cmd_x += cmd_limits[0, 0]
    if keys[pygame.K_a]:    cmd_y += cmd_limits[1, 1]
    if keys[pygame.K_d]:    cmd_y += cmd_limits[1, 0]
    if keys[pygame.K_q]:    cmd_yaw += cmd_limits[2, 1]
    if keys[pygame.K_e]:    cmd_yaw += cmd_limits[2, 0]
    return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    args = parser.parse_args()
    save_video = args.save_video
    config_file = "go2.yaml"

    pygame.init()
    screen = pygame.display.set_mode((200, 100))
    pygame.display.set_caption("Keyboard Control")
    print("Keyboard control: W/S=vx, A/D=vy, Q/E=yaw")

    with open(f"deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        if "cmd_range" in config:
            cmd_limits = np.array(config["cmd_range"], dtype=np.float32)
        else:
            max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            cmd_limits = np.stack((-max_cmd, max_cmd), axis=1)

        idx_model2mj = idx_mj2model = list(range(num_actions))
        if 'mujoco_joint_names' in config and 'model_joint_names' in config:
            mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]
            idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
            idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    model_name = os.path.basename(policy_path).split('.')[0]
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    renderer = mujoco.Renderer(m, height=360, width=640)
    
    # load policy
    policy = torch.jit.load(policy_path)

    video_fps = 50
    if save_video:
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        print(f"Video recording will be saved to: {video_path}")
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = int(sim_fps / video_fps)
        if frame_skip < 1:
            frame_skip = 1
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Sim FPS: {sim_fps:.2f}, Video FPS: {video_fps}, Frame Skip: {frame_skip}, Save at: {video_path}")
    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:

        # set viewer.camera to follow robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 60.0

        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            vel = d.qvel[:3]
            ang_vel = d.qvel[3:6]
            local_vel = quat_rotate_inverse(d.qpos[3:7], vel)
            local_ang_vel = quat_rotate_inverse(d.qpos[3:7], ang_vel)
            show_str = f"Speed: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:.2f}, "
            step_start = time.time()

            if counter % control_decimation == 0:
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                cmd = get_keyboard_command(keys, cmd_limits)
                show_str += f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}"
                print(show_str, end='\r')

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            mujoco_render_utils.update(cmd, d)

            if save_video and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                    frame = renderer.render()
                    writer.append_data(frame)
                except Exception as e:
                    print(f"Error rendering frame: {e}")

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                lin_vel = d.qvel[:3]
                ang_vel = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale

                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                lin_vel = lin_vel * lin_vel_scale
                ang_vel = ang_vel * ang_vel_scale

                obs[:3] = ang_vel
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj[idx_mj2model]
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj[idx_mj2model]
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action[idx_mj2model]
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                last_action = action
                result = policy(obs_tensor)
                if isinstance(result, tuple):
                    action, (weights, latent) = result
                    action = action.detach().numpy().squeeze()[idx_model2mj]
                    weights = weights.detach().numpy().squeeze()
                    latent = latent.detach().numpy().squeeze()
                else:
                    action = result.detach().cpu().numpy().squeeze()[idx_model2mj]
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            # time_until_next_step = m.opt.timestep - (time.time() - step_start) - 0.1
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

    # writer.close()
    if save_video:
        print(f"Video saved successfully to {video_path}")
        writer.close()
