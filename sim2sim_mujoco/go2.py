from legged_gym.envs.go2.go2_amp_config import Go2AMPCfg
import math
import numpy as np
import mujoco, mujoco.viewer
from legged_gym import LEGGED_GYM_ROOT_DIR
from scipy.spatial.transform import Rotation as R
import torch
import keyword
from pynput import keyboard

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 3.0

joystick_use = True
joystick_opened = False
def on_press(key):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    try:
        if key == keyboard.Key.up:
            x_vel_cmd += 0.3
        elif key == keyboard.Key.down:
            x_vel_cmd -= 0.3
        elif key == keyboard.Key.left:
            y_vel_cmd += 0.3
        elif key == keyboard.Key.right:
            y_vel_cmd -= 0.3
        elif key.char == ',':
            yaw_vel_cmd += 0.3
        elif key.char == '.':
            yaw_vel_cmd -= 0.3
        elif key.char == 'm':
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
        x_vel_cmd = np.clip(x_vel_cmd, -x_vel_max, x_vel_max)
        y_vel_cmd = np.clip(y_vel_cmd, -y_vel_max, y_vel_max)
        yaw_vel_cmd = np.clip(yaw_vel_cmd, -yaw_vel_max, yaw_vel_max)
        print(f"Command: {x_vel_cmd:.2f}, {y_vel_cmd:.2f}, {yaw_vel_cmd:.2f}")
    except AttributeError:
        pass

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    base_pos = data.qpos[0:3].astype(np.double)
    dof_pos = data.qpos[7:19].astype(np.double)
    dof_vel = data.qvel[6:].astype(np.double)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    base_lin_vel = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    base_ang_vel = data.qvel[3:6].astype(np.double)
    
    return base_pos, dof_pos, dof_vel, quat, base_lin_vel, base_ang_vel

def pd_control(target_dof_pos, dof_pos, kp, target_dof_vel, dof_vel, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_dof_pos + cfg.robot_config.default_dof_pos - dof_pos) * kp \
                 + (target_dof_vel - dof_vel) * kd
    
    return torque_out

def run_mujoco(policy, cfg):
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    data.qpos[2] = 0.42  # initial height
    default_dof_pos = cfg.robot_config.default_dof_pos
    data.qpos[7:19] = default_dof_pos
    mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data)

    target_dof_pos = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_dof_vel = np.zeros_like(target_dof_pos)
    actions = np.zeros((cfg.env.num_actions), dtype=np.double)

    count_lowlevel = 1

    print(f"观测维度: {cfg.env.num_observations}")

    for step in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
        base_pos, dof_pos, dof_vel, quat, base_lin_vel, base_ang_vel = get_obs(data)
        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)

            obs[0, 0:3] = base_ang_vel * cfg.normalization.obs_scales.ang_vel  # 3 angle_vel
            obs[0, 3:6] = np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd]) * np.array([
                cfg.normalization.obs_scales.lin_vel,
                cfg.normalization.obs_scales.lin_vel,
                cfg.normalization.obs_scales.ang_vel
            ])  # 3 command
            obs[0, 6:18] = (dof_pos - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos  # 12 dof_pos
            obs[0, 18:30] = dof_vel * cfg.normalization.obs_scales.dof_vel  # 12 dof_vel
            obs[0, 30:42] = actions # 12 actions

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            policy_input = obs
            actions[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            actions = np.clip(actions, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            
            target_dof_pos[:] = actions * cfg.control.action_scale
        
        if step < 100:
            tau = pd_control(np.zeros(cfg.env.num_actions), dof_pos, cfg.robot_config.kps,
                            np.zeros(cfg.env.num_actions), dof_vel, cfg.robot_config.kds, cfg)
        else:
            tau = pd_control(target_dof_pos, dof_pos, cfg.robot_config.kps,
                            target_dof_vel, dof_vel, cfg.robot_config.kds, cfg)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.sync()
        count_lowlevel += 1    
    viewer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script')
    parser.add_argument('--load_model', type=str, default='/home/ak1ra/rl_amp/legged_gym/logs/amp_go2/exported/policies/policy_1.pt')
    parser.add_argument('--mujoco_model', type=str, default='/home/ak1ra/Quadruped/go2_sim2sim/go2_description/scene_terrain.xml')
    parser.add_argument('--terrain', action='store_true', help='enable terrain')
    args = parser.parse_args()

    class Sim2MujocoTrot( Go2AMPCfg ):
        class sim_config:
            mujoco_model_path = args.mujoco_model
            sim_duration = 120.
            dt = 0.001
            decimation = 20

        class robot_config:
            kps = np.array([20.0] * 12, dtype=np.double)  # 与Isaac Gym保持一致：25.0
            kds = np.array([0.5] * 12, dtype=np.double)   # 与Isaac Gym保持一致：0.6
            tau_limit = 45 * np.ones(12, dtype=np.double)
            default_dof_pos = np.array([
                0.0, 0.8, -1.5,   # FL
                0.0, 0.8, -1.5,   # RL
                -0.0, 0.8, -1.5,  # FR
                -0.0, 0.8, -1.5,  # RR
            ], dtype=np.double)
        
    policy = torch.jit.load(args.load_model)
    print("Loaded policy from ", args.load_model)
    
    run_mujoco(policy, Sim2MujocoTrot)