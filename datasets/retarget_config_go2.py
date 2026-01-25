import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR

VISUALIZE_RETARGETING = True

URDF_FILENAME = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

OUTPUT_DIR = "{LEGGED_GYM_ROOT_DIR}/../datasets/mocap_motions_go2".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

REF_POS_SCALE = 0.825

INIT_POS = np.array([0, 0, 0.35])
INIT_ROT = np.array([0, 0, 0, 1.0])

# Following the same pattern as A1: [FR, FL, RR, RL]
SIM_TOE_JOINT_IDS = [13, 7, 25, 19]  # FR_foot_joint, FL_foot_joint, RR_foot_joint, RL_foot_joint
SIM_HIP_JOINT_IDS = [8, 2, 20, 14]   # FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint
SIM_ROOT_OFFSET = np.array([0, 0, -0.04])

SIM_TOE_OFFSET_LOCAL = [
    np.array([0.0, -0.01, 0.0]),   # FR
    np.array([0.0, 0.01, 0.0]),  # FL
    np.array([0.0, -0.01, 0.0]),   # RR
    np.array([0.0, 0.01, 0.0])   # RL
]
TOE_HEIGHT_OFFSET = 0.02

# [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
DEFAULT_JOINT_POSE = np.array([
    0.0, 0.8, -1.5,   # FL: hip, thigh, calf
    0.0, 0.8, -1.5,   # FR: hip, thigh, calf
    -0.0, 1.0, -1.5,  # RL: hip, thigh, calf
    -0.0, 1.0, -1.5   # RR: hip, thigh, calf
])

JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])

FR_FOOT_NAME = "FR_foot"
FL_FOOT_NAME = "FL_foot"
HR_FOOT_NAME = "RR_foot"
HL_FOOT_NAME = "RL_foot"

MOCAP_MOTIONS = [
    ["pace0", "datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 162, 201, 1],
    ["pace1", "datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 201, 400, 1],
    ["pace2", "datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 400, 600, 1],
    ["trot0", "datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 448, 481, 1],
    ["trot1", "datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 400, 600, 1],
    ["trot2", "datasets/keypoint_datasets/ai4animation/dog_run04_joint_pos.txt", 480, 663, 1],
    ["canter0", "datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 430, 480, 1],
    ["canter1", "datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 380, 430, 1],
    ["canter2", "datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 480, 566, 1],
    ["right_turn0", "datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 1085, 1124, 1.5],
    ["right_turn1", "datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 560, 670, 1.5],
    ["left_turn0", "datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 2404, 2450, 1.5],
    ["left_turn1", "datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 120, 220, 1.5]
]