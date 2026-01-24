import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR

VISUALIZE_RETARGETING = True
URDF_FILENAME = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf".format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
OUTPUT_DIR = "{LEGGED_GYM_ROOT_DIR}/../datasets/mocap_motions_go2".format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
REF_POS_SCALE = 0.825 # 缩放系数,如果遇到关节限位异常，尝试将此数变小
INIT_POS = np.array([0, 0, 0.42]) # go2
INIT_ROT = np.array([0, 0, 0, 1.0])