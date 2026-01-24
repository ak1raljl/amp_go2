# RL-AMP项目文档

## 项目概述

这是一个专注于对抗运动先验（AMP）的强化学习项目，用于四足机器人运动。该项目结合了三个主要组件：

1. **legged_gym**: 使用Isaac Gym训练四足机器人（ANYmal, A1, Cassie）在复杂地形上行走的环境
2. **rsl_rl**: 使用PyTorch实现的快速简单RL算法（PPO和AMP）
3. **datasets**: 运动捕捉数据处理和重定向工具，用于创建参考动作

该项目通过对抗学习技术使用运动捕捉数据来训练真实的运动行为。

## 架构

### 核心组件

- **legged_gym**: 包含机器人环境（A1, ANYmal B/C, Cassie），配置用于标准RL和AMP训练
- **rsl_rl**: 实现PPO算法和AMP判别器用于对抗运动先验
- **datasets**: 用于处理和重定向运动捕捉数据到机器人运动学的工具

### 主要特性

- **AMP（对抗运动先验）**: 使用对抗学习从运动捕捉数据中学习
- **运动重定向**: 将人类/动物运动捕捉数据转换为机器人关节角度
- **复杂地形导航**: 专为在挑战性地形上行走而设计的环境
- **仿真到现实迁移**: 包括域随机化以将策略迁移到真实机器人

## 构建和运行

### 先决条件

1. Python 3.6-3.8（推荐：3.8）
2. PyTorch 1.10 with CUDA 11.3
3. Isaac Gym Preview 3（Preview 2不兼容！）
4. rsl_rl库

### 安装

```bash
# 1. 创建Python虚拟环境（推荐Python 3.8）
# 2. 安装PyTorch 1.10 with CUDA 11.3
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 3. 安装Isaac Gym
# 从 https://developer.nvidia.com/isaac-gym 下载并安装Isaac Gym Preview 3
cd isaacgym/python && pip install -e .

# 4. 安装rsl_rl
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .

# 5. 安装legged_gym
cd legged_gym && pip install -e .
```

### 训练命令

```bash
# 使用AMP训练A1机器人
python legged_gym/scripts/train.py --task=a1_amp

# 使用自定义参数训练
python legged_gym/scripts/train.py --task=a1_amp --num_envs=4096 --max_iterations=4000
```

### 播放训练策略

```bash
# 播放训练好的策略
python legged_gym/scripts/play.py --task=a1_amp
```

### 处理运动捕捉数据

```bash
# 将运动捕捉数据重定向到A1机器人
python datasets/retarget_kp_motions.py
```

### 回放AMP数据

```bash
# 回放AMP轨迹
python legged_gym/scripts/replay_amp_data.py --task=a1_amp
```

## 关键文件和目录

### legged_gym/
- `envs/`: 机器人环境（A1, ANYmal, Cassie）及基础实现
- `scripts/`: 训练和评估脚本
- `utils/`: 用于训练和可视化的辅助工具

### rsl_rl/
- `algorithms/`: PPO和AMP算法实现
- `modules/`: 神经网络模块
- `storage/`: RL算法的经验存储

### datasets/
- `retarget_config.py`: 运动重定向配置
- `retarget_kp_motions.py`: 运动重定向流水线
- `retarget_utils.py`: 运动处理实用工具
- `mocap_motions_a1/`: 为A1机器人处理的运动捕捉数据

## 开发约定

### AMP（对抗运动先验）
- 结合任务奖励和运动模仿使用判别器
- 判别器学习区分专家运动捕捉数据和策略生成的运动
- 策略试图欺骗判别器同时实现任务目标

### 运动重定向过程
1. 加载运动捕捉数据（来自动物/人类的关键点）
2. 应用坐标变换和缩放
3. 使用逆运动学映射到机器人关节角度
4. 生成平滑过渡的速度信息
5. 保存处理后的运动用于训练

### 环境结构
- 每个机器人都有一个基础环境类和配置
- 配置将环境参数与训练参数分离
- 奖励函数是模块化且可配置的
- 域随机化用于仿真到现实迁移

## 关键技术

- **Isaac Gym**: 物理仿真和图形渲染
- **PyTorch**: 深度学习框架
- **rsl_rl**: 自定义RL算法库
- **PyBullet**: 用于运动重定向IK计算
- **CUDA**: 仿真实验和训练的GPU加速

## 重要说明

- 该项目基于Isaac Gym（非Isaac Sim），已被NVIDIA弃用
- 维护者建议新项目迁移到Orbit框架
- 运动捕捉数据使用逆运动学处理以匹配机器人运动学
- AMP结合传统RL奖励和运动模仿以获得更自然的运动