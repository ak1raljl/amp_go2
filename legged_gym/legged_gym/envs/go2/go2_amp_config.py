import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# MOTION_FILES = glob.glob('datasets/mocap_motions_go2/*')
MOTION_FILES = glob.glob('datasets/go2_motion/*')
class Go2AMPCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 235
        reference_state_initialization = True
        reference_state_initialization_prob = 999 # 1. >1 所有要reset的环境都用AMP frame初始化 2. <0 所有环境都走普通reset。
        amp_motion_files = MOTION_FILES
        ee_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( LeggedRobotCfg.terrain ):
        # mesh_type = 'trimesh'
        mesh_type = 'plane'
        border_size = 25 # [m]
        measure_heights = True
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, flat]
        curriculum = True
        max_init_terrain_level = 5
        terrain_proportions = [0.4, 0.4, 0.1, 0.1, 0.0]
        # terrain_proportions = [0.5, 0.5, 0.0, 0.0, 0.0]
        terrain_length = 8.0 # [m]
        terrain_width = 8.0 # [m]
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        ### Robot properties ###
        randomize_friction = True
        friction_range = [0.0, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]
        randomize_restitution = True # restitution to robot links (Robot init)
        restitution_range = [0.0, 0.5]

        ### Environment reset ###
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        ### Environment step ###
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        
    class noise( LeggedRobotCfg.noise ):
        add_noise = True

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.38
        only_positive_rewards = False
        tracking_sigma = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            torques = -1e-4
            base_height = -0.3
            action_rate = -0.01
            collision = -1.0
            dof_pos_limits = -2.0
            feet_air_time =  1.0
    
    class normalization ( LeggedRobotCfg.normalization ):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 2.5
        clip_observations = 100.
        clip_actions = 100.

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.2, 1.5] # min max [m/s]
            lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-1.57, 1.57]

class Go2AMPCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'flat'
        experiment_name = 'go2_amp'
        max_iterations = 500000 # number of policy updates
        save_interval = 2000 # number of policy updates

        amp_reward_coef = 0.2
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.8
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.01, 0.01, 0.01] * 4
