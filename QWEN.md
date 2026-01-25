# RL-AMP Project Documentation

## Project Overview

This is a reinforcement learning project focused on Adversarial Motion Priors (AMP) for quadrupedal robot locomotion. The project combines three main components:

1. **legged_gym**: Environments for training quadrupedal robots (ANYmal, A1, Cassie) to walk on rough terrain using NVIDIA's Isaac Gym
2. **rsl_rl**: Fast and simple RL algorithms implementation using PyTorch (PPO and AMP)
3. **datasets**: Motion capture data processing and retargeting tools for creating reference motions

The project uses adversarial learning techniques to leverage motion capture data for training realistic locomotion behaviors.

## Architecture

### Core Components

- **legged_gym**: Contains robot environments (A1, ANYmal B/C, Cassie) configured for standard RL and AMP training
- **rsl_rl**: Implements PPO algorithm and AMP discriminator for adversarial motion priors
- **datasets**: Tools for processing and retargeting motion capture data to robot kinematics

### Key Features

- **AMP (Adversarial Motion Priors)**: Uses adversarial learning to learn from motion capture data
- **Motion Retargeting**: Converts human/animal motion capture data to robot joint angles
- **Rough Terrain Navigation**: Environments specifically designed for walking on challenging terrain
- **Sim-to-Real Transfer**: Includes domain randomization to transfer policies to real robots

## Building and Running

### Prerequisites

1. Python 3.6-3.8 (recommended: 3.8)
2. PyTorch 1.10 with CUDA 11.3
3. Isaac Gym Preview 3 (Preview 2 is incompatible!)
4. rsl_rl library

### Installation

```bash
# 1. Create Python virtual environment (recommended Python 3.8)
# 2. Install PyTorch 1.10 with CUDA 11.3
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 3. Install Isaac Gym
# Download and install Isaac Gym Preview 3 from https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e .

# 4. Install rsl_rl
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .

# 5. Install legged_gym
cd legged_gym && pip install -e .
```

### Training Commands

```bash
# Train A1 robot with AMP
python legged_gym/scripts/train.py --task=a1_amp

# Train with custom parameters
python legged_gym/scripts/train.py --task=a1_amp --num_envs=4096 --max_iterations=4000
```

### Playing Trained Policies

```bash
# Play trained policy
python legged_gym/scripts/play.py --task=a1_amp
```

### Processing Motion Capture Data

```bash
# Retarget motion capture data to A1 robot
python datasets/retarget_kp_motions.py
```

### Replaying AMP Data

```bash
# Replay AMP trajectories
python legged_gym/scripts/replay_amp_data.py --task=a1_amp
```

## Key Files and Directories

### legged_gym/
- `envs/`: Robot environments (A1, ANYmal, Cassie) and base implementations
- `scripts/`: Training and evaluation scripts
- `utils/`: Helper utilities for training and visualization

### rsl_rl/
- `algorithms/`: PPO and AMP algorithm implementations
- `modules/`: Neural network modules
- `storage/`: Experience storage for RL algorithms

### datasets/
- `retarget_config_a1.py`: Motion retargeting configuration for A1 robot
- `retarget_config_go2.py`: Motion retargeting configuration for Go2 robot
- `retarget_kp_motions.py`: Motion retargeting pipeline
- `retarget_utils.py`: Motion processing utilities
- `mocap_motions_a1/`: Processed motion capture data for A1 robot

## Development Conventions

### AMP (Adversarial Motion Priors)
- Combines task rewards with motion imitation using a discriminator
- The discriminator learns to distinguish between expert motion capture data and policy-generated motions
- The policy tries to fool the discriminator while achieving task objectives

### Motion Retargeting Process
1. Load motion capture data (keypoints from animals/humans)
2. Apply coordinate transforms and scaling
3. Map to robot joint angles using inverse kinematics
4. Generate smooth transition velocities
5. Save processed motions for training

### Environment Structure
- Each robot has a base environment class and configuration
- Configurations separate environment parameters from training parameters
- Reward functions are modular and configurable
- Domain randomization is used for sim-to-real transfer

## Key Technologies

- **Isaac Gym**: Physics simulation and graphics rendering
- **PyTorch**: Deep learning framework
- **rsl_rl**: Custom RL algorithms library
- **PyBullet**: Used for motion retargeting IK computations
- **CUDA**: GPU acceleration for simulation and training

## Important Notes

- This project is based on Isaac Gym (not Isaac Sim), which has been deprecated by NVIDIA
- Maintainers suggest migrating to Orbit framework for new projects
- Motion capture data is processed using inverse kinematics to match robot kinematics
- AMP combines traditional RL rewards with motion imitation to achieve more natural movement