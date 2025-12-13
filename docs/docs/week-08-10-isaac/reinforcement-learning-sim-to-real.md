---
title: Reinforcement Learning and Sim-to-Real Transfer
sidebar_position: 3
description: Implementing reinforcement learning for robot control and sim-to-real transfer techniques
duration: 240
difficulty: advanced
learning_objectives:
  - Implement reinforcement learning algorithms for robot control
  - Apply domain randomization techniques for sim-to-real transfer
  - Optimize policies for real-world deployment
  - Evaluate and improve robot learning performance
---

# Reinforcement Learning and Sim-to-Real Transfer

## Learning Objectives

By the end of this section, you will be able to:
- Implement reinforcement learning algorithms for robot control tasks
- Apply domain randomization techniques to improve sim-to-real transfer
- Optimize learned policies for real-world deployment
- Evaluate and improve robot learning performance
- Integrate reinforcement learning with perception and manipulation systems

## Introduction to Reinforcement Learning in Robotics

Reinforcement Learning (RL) is a powerful approach for learning robot control policies through interaction with the environment. In robotics, RL can be used for:

- **Locomotion Control**: Learning walking, running, or other movement patterns
- **Manipulation Skills**: Learning grasping, pushing, or assembly tasks
- **Navigation**: Learning to navigate complex environments
- **Task Planning**: Learning high-level decision making

### Key Components of RL in Robotics

1. **Agent**: The robot learning to perform tasks
2. **Environment**: The physical or simulated world the robot interacts with
3. **State**: Robot sensor data and environmental information
4. **Action**: Motor commands sent to the robot
5. **Reward**: Feedback signal indicating task success/failure
6. **Policy**: Mapping from states to actions

## Isaac RL Components

### Isaac's RL Framework

NVIDIA Isaac provides specialized tools for reinforcement learning:

- **Isaac Gym**: GPU-accelerated RL environment
- **Domain Randomization**: Techniques for sim-to-real transfer
- **Pre-trained Models**: Starting points for learning
- **Simulation Tools**: High-fidelity physics and sensor simulation

### Isaac Gym for Robot Learning

```python
# isaac_gym_robot.py
import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict

class IsaacGymRobot:
    """Robot environment for reinforcement learning in Isaac Gym"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg['device']

        # Initialize gym
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.envs = []
        self.actor_handles = []

        # Robot properties
        self.num_envs = cfg['num_envs']
        self.num_obs = cfg['num_obs']
        self.num_actions = cfg['num_actions']

        # RL parameters
        self.dt = cfg['dt']
        self.max_episode_length = cfg['max_episode_length']

        # Initialize the simulation
        self._create_sim()
        self._create_envs()

    def _create_sim(self):
        """Create physics simulation"""
        # Configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # Set physics engine parameters
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.parallel_thread_count = 4
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        sim_params.physx.num_subscenes = 4
        sim_params.physx.contact_collection = gymapi.ContactCollection.KINEMATIC_KINEMATIC_FIX_NEAREST

        # Create sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.gym.viewer_camera_look_at(
            self.viewer, None, gymapi.Vec3(5, 5, 1), gymapi.Vec3(0, 0, 0)
        )

    def _create_envs(self):
        """Create environments"""
        # Load robot asset
        asset_root = self.cfg['asset_root']
        asset_file = self.cfg['asset_file']

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 1000.0
        asset_options.max_linear_velocity = 1000.0
        asset_options.slices_per_cylinder = 4
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = False
        asset_options.use_mesh_materials = False

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Configure robot DOFs
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for i in range(len(robot_dof_props)):
            robot_dof_props['drive_mode'][i] = gymapi.DOF_MODE_EFFORT
            robot_dof_props['stiffness'][i] = 0.0
            robot_dof_props['damping'][i] = 0.0

        # Configure robot shape properties
        robot_shape_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        for shape_prop in robot_shape_props:
            shape_prop.friction = 1.0
            shape_prop.rolling_friction = 0.0
            shape_prop.torsion_friction = 0.0
            shape_prop.restitution = 0.0
            shape_prop.thickness = 0.01

        # Create environments
        spacing = 2.5
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(self.num_envs):
            # Create environment
            env = self.gym.create_env(self.sim, lower, upper, 1)
            self.envs.append(env)

            # Add ground plane
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.distance = 0.0
            plane_params.static_friction = 1.0
            plane_params.dynamic_friction = 1.0
            plane_params.restitution = 0.0
            self.gym.add_ground(self.sim, plane_params)

            # Set default pose
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            # Create actor
            actor_handle = self.gym.create_actor(env, robot_asset, start_pose, "robot", i, 1, 0)
            self.actor_handles.append(actor_handle)

            # Set DOF properties
            self.gym.set_actor_dof_properties(env, actor_handle, robot_dof_props)

    def reset_idx(self, env_ids):
        """Reset specific environments"""
        # Reset robot positions and velocities
        positions = torch.zeros((len(env_ids), self.num_actions), device=self.device)
        velocities = torch.zeros((len(env_ids), self.num_actions), device=self.device)

        # Set new state
        self.root_tensor[env_ids, 0:3] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.root_tensor[env_ids, 7:10] = torch.zeros(3, device=self.device)

        # Reset DOF states
        self.dof_state_tensor[env_ids, :] = torch.cat([positions, velocities], dim=1)

        # Reset progress buffer
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def pre_physics_step(self, actions):
        """Apply actions before physics simulation step"""
        # Convert actions to torques
        torques = actions * self.cfg['action_scale']

        # Apply torques to DOFs
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.GymTensorProperties.FORCE_TENSORS, torques)

    def post_physics_step(self):
        """Process simulation results after physics step"""
        # Retrieve root state tensor
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # Compute observations
        self.compute_observations()

        # Compute rewards
        self.compute_rewards()

        # Update episode progress
        self.progress_buf += 1
        self.reset_buf = torch.zeros_like(self.reset_buf)

        # Reset environments that reached max episode length
        env_ids = self.progress_buf >= self.max_episode_length
        self.reset_buf = env_ids
        if torch.count_nonzero(env_ids) > 0:
            self.reset_idx(env_ids.nonzero(as_tuple=False).squeeze(-1))

    def compute_observations(self):
        """Compute robot observations"""
        # This would compute observations from sensor data
        # For example: joint positions, velocities, IMU data, etc.
        pass

    def compute_rewards(self):
        """Compute rewards for each environment"""
        # This would compute rewards based on task completion
        # For example: reaching target, maintaining balance, etc.
        pass

class PPOAgent:
    """Proximal Policy Optimization agent for robot control"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor and Critic networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        action_mean, action_std = self.actor(state)
        cov_mat = torch.diag(action_std).unsqueeze(0)

        # Create distribution
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        # Sample action
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach().cpu().numpy().flatten(), action_logprob.detach().cpu().numpy().flatten()

    def update(self, state_batch, action_batch, logprob_batch, reward_batch, terminal_batch):
        """Update policy using PPO"""
        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        logprob_batch = torch.FloatTensor(logprob_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        terminal_batch = torch.FloatTensor(terminal_batch).to(self.device)

        # Compute discounted rewards (returns)
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(reward_batch), reversed(terminal_batch)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Convert old logprobs to tensor
        old_logprob_batch = logprob_batch

        # Optimize policy for K epochs
        for _ in range(80):  # K epochs
            # Evaluate old actions and values using current policy
            action_mean, action_std = self.actor(state_batch)
            cov_mat = torch.diag_embed(action_std)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

            logprobs = dist.log_prob(action_batch)
            state_values = self.critic(state_batch).squeeze()

            # Compute advantages
            advantages = returns - state_values.detach()

            # Compute ratio
            ratios = torch.exp(logprobs - old_logprob_batch)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute critic loss
            critic_loss = self.MseLoss(state_values, returns)

            # Take optimization step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

class ActorNetwork(nn.Module):
    """Actor network for policy"""

    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(ActorNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.l4 = nn.Linear(256, action_dim)  # For std

        self.max_action = max_action
        self.tanh = nn.Tanh()

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))

        action_mean = self.max_action * self.tanh(self.l3(a))
        action_std = torch.ones_like(action_mean) * 0.5  # Fixed std for simplicity

        return action_mean, action_std

class CriticNetwork(nn.Module):
    """Critic network for value function"""

    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        v = torch.relu(self.l1(state))
        v = torch.relu(self.l2(v))
        v = self.l3(v)
        return v
```

## Domain Randomization

### Implementing Domain Randomization

Domain randomization is a key technique for sim-to-real transfer that involves randomizing simulation parameters to make the agent robust to environmental variations.

```python
# domain_randomization.py
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class DomainParams:
    """Parameters for domain randomization"""
    # Physics parameters
    friction_range: Tuple[float, float] = (0.5, 1.5)
    restitution_range: Tuple[float, float] = (0.0, 0.2)
    mass_multiplier_range: Tuple[float, float] = (0.8, 1.2)

    # Visual parameters
    light_intensity_range: Tuple[float, float] = (0.5, 2.0)
    color_variance_range: Tuple[float, float] = (0.0, 0.1)
    texture_randomization: bool = True

    # Sensor parameters
    sensor_noise_range: Tuple[float, float] = (0.0, 0.01)
    sensor_bias_range: Tuple[float, float] = (-0.005, 0.005)

    # Actuator parameters
    motor_strength_range: Tuple[float, float] = (0.9, 1.1)
    control_delay_range: Tuple[int, int] = (0, 3)  # in simulation steps

class DomainRandomizer:
    """Domain randomization manager for Isaac Sim"""

    def __init__(self, params: DomainParams):
        self.params = params
        self.applied_params = {}

    def randomize_environment(self, env_id: int):
        """Randomize environment parameters for specific environment"""
        # Randomize physics properties
        friction = random.uniform(*self.params.friction_range)
        restitution = random.uniform(*self.params.restitution_range)
        mass_mult = random.uniform(*self.params.mass_multiplier_range)

        # Apply randomization to environment
        self.apply_physics_randomization(env_id, friction, restitution, mass_mult)

        # Randomize visual properties
        light_intensity = random.uniform(*self.params.light_intensity_range)
        color_variance = random.uniform(*self.params.color_variance_range)

        self.apply_visual_randomization(env_id, light_intensity, color_variance)

        # Randomize sensor properties
        sensor_noise = random.uniform(*self.params.sensor_noise_range)
        sensor_bias = random.uniform(*self.params.sensor_bias_range)

        self.apply_sensor_randomization(env_id, sensor_noise, sensor_bias)

        # Randomize actuator properties
        motor_strength = random.uniform(*self.params.motor_strength_range)
        control_delay = random.randint(*self.params.control_delay_range)

        self.apply_actuator_randomization(env_id, motor_strength, control_delay)

        # Store applied parameters for this environment
        self.applied_params[env_id] = {
            'friction': friction,
            'restitution': restitution,
            'mass_multiplier': mass_mult,
            'light_intensity': light_intensity,
            'sensor_noise': sensor_noise,
            'motor_strength': motor_strength,
            'control_delay': control_delay
        }

    def apply_physics_randomization(self, env_id: int, friction: float, restitution: float, mass_mult: float):
        """Apply physics randomization to environment"""
        # In Isaac Sim, this would modify physics properties of objects
        print(f"Env {env_id}: Applied physics randomization - friction={friction:.3f}, "
              f"restitution={restitution:.3f}, mass_mult={mass_mult:.3f}")

    def apply_visual_randomization(self, env_id: int, light_intensity: float, color_variance: float):
        """Apply visual randomization to environment"""
        # In Isaac Sim, this would modify lighting and materials
        print(f"Env {env_id}: Applied visual randomization - light={light_intensity:.3f}, "
              f"color_var={color_variance:.3f}")

    def apply_sensor_randomization(self, env_id: int, noise: float, bias: float):
        """Apply sensor randomization to environment"""
        # In Isaac Sim, this would modify sensor properties
        print(f"Env {env_id}: Applied sensor randomization - noise={noise:.5f}, bias={bias:.5f}")

    def apply_actuator_randomization(self, env_id: int, strength: float, delay: int):
        """Apply actuator randomization to environment"""
        # In Isaac Sim, this would modify actuator properties
        print(f"Env {env_id}: Applied actuator randomization - strength={strength:.3f}, delay={delay}")

class AdvancedDomainRandomizer(DomainRandomizer):
    """Advanced domain randomization with curriculum learning"""

    def __init__(self, params: DomainParams):
        super().__init__(params)
        self.curriculum_stage = 0
        self.max_curriculum_stages = 5
        self.training_progress = 0.0  # 0.0 to 1.0

    def update_curriculum(self, progress: float):
        """Update curriculum based on training progress"""
        self.training_progress = progress
        self.curriculum_stage = min(int(progress * self.max_curriculum_stages), self.max_curriculum_stages - 1)

        # Adjust randomization ranges based on curriculum stage
        self.adjust_randomization_ranges()

    def adjust_randomization_ranges(self):
        """Adjust randomization ranges based on curriculum stage"""
        base_range = 0.1  # Base randomization at stage 0
        max_range = 0.5   # Maximum randomization at final stage

        # Calculate current range based on curriculum stage
        current_range_factor = base_range + (max_range - base_range) * (self.curriculum_stage / (self.max_curriculum_stages - 1))

        # Adjust parameters
        self.params.friction_range = (1.0 - current_range_factor, 1.0 + current_range_factor)
        self.params.mass_multiplier_range = (1.0 - current_range_factor, 1.0 + current_range_factor)
        self.params.sensor_noise_range = (0.0, current_range_factor * 0.02)

    def randomize_environment_curriculum(self, env_id: int):
        """Randomize environment with curriculum-based ranges"""
        # Update ranges if needed
        self.adjust_randomization_ranges()

        # Apply randomization
        self.randomize_environment(env_id)

# Example usage of domain randomization
def train_with_domain_randomization():
    """Example of training with domain randomization"""
    # Initialize domain randomizer
    domain_params = DomainParams(
        friction_range=(0.5, 1.5),
        restitution_range=(0.0, 0.2),
        sensor_noise_range=(0.0, 0.01),
        motor_strength_range=(0.9, 1.1)
    )

    randomizer = AdvancedDomainRandomizer(domain_params)

    # Training loop with domain randomization
    num_episodes = 10000
    episode_length = 1000

    for episode in range(num_episodes):
        # Update curriculum based on training progress
        progress = episode / num_episodes
        randomizer.update_curriculum(progress)

        # Reset environment with randomization
        env_id = episode % 64  # Assuming 64 parallel environments
        randomizer.randomize_environment_curriculum(env_id)

        # Execute episode
        total_reward = 0
        for step in range(episode_length):
            # Get action from policy
            action = get_action_from_policy()

            # Apply action (with possible delay from randomization)
            execute_action_with_delay(action, randomizer.applied_params[env_id].get('control_delay', 0))

            # Observe environment (with noise from randomization)
            obs = get_observation_with_noise(randomizer.applied_params[env_id].get('sensor_noise', 0.0))

            # Compute reward
            reward = compute_reward(obs)
            total_reward += reward

            # Check if episode is done
            if is_episode_done():
                break

        # Log training progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {total_reward / episode_length:.2f}, "
                  f"Curriculum Stage: {randomizer.curriculum_stage}")
```

## Sim-to-Real Transfer Techniques

### Implementing Sim-to-Real Transfer

```python
# sim_to_real_transfer.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import cv2
import random

class SimToRealTransfer:
    """Framework for sim-to-real transfer in robotics"""

    def __init__(self):
        self.sim_model = None
        self.real_model = None
        self.domain_classifier = DomainClassifier()
        self.adaptation_network = AdaptationNetwork()

    def train_domain_adversarial(self, sim_data_loader, real_data_loader, epochs=100):
        """Train with domain adversarial adaptation"""
        optimizer_g = torch.optim.Adam(list(self.adaptation_network.parameters()), lr=0.001)
        optimizer_d = torch.optim.Adam(self.domain_classifier.parameters(), lr=0.001)

        for epoch in range(epochs):
            for (sim_batch, _), (real_batch, _) in zip(sim_data_loader, real_data_loader):
                # Train domain classifier
                optimizer_d.zero_grad()

                # Sim features
                sim_features = self.adaptation_network(sim_batch)
                sim_domain_pred = self.domain_classifier(sim_features)
                sim_domain_loss = F.binary_cross_entropy(sim_domain_pred, torch.zeros_like(sim_domain_pred))

                # Real features
                real_features = self.adaptation_network(real_batch)
                real_domain_pred = self.domain_classifier(real_features)
                real_domain_loss = F.binary_cross_entropy(real_domain_pred, torch.ones_like(real_domain_pred))

                domain_loss = sim_domain_loss + real_domain_loss
                domain_loss.backward()
                optimizer_d.step()

                # Train feature extractor to fool domain classifier
                optimizer_g.zero_grad()

                sim_features = self.adaptation_network(sim_batch)
                sim_domain_pred = self.domain_classifier(sim_features)
                gen_loss = F.binary_cross_entropy(sim_domain_pred, torch.ones_like(sim_domain_pred))

                gen_loss.backward()
                optimizer_g.step()

            print(f"Epoch {epoch}, Domain Loss: {domain_loss.item():.4f}, Gen Loss: {gen_loss.item():.4f}")

class DomainClassifier(nn.Module):
    """Domain classifier for adversarial domain adaptation"""

    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

class AdaptationNetwork(nn.Module):
    """Feature adaptation network"""

    def __init__(self):
        super(AdaptationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Assuming 64x64 input -> 8x8 after pooling

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        return x

class SystematicDifferencesCorrector:
    """Correct systematic differences between sim and real"""

    def __init__(self):
        self.sim_to_real_mapping = {}
        self.calibration_data = []

    def collect_calibration_data(self, sim_obs, real_obs):
        """Collect paired sim and real observations for calibration"""
        self.calibration_data.append((sim_obs, real_obs))

    def learn_mapping(self):
        """Learn mapping from sim to real observations"""
        if len(self.calibration_data) < 10:
            print("Not enough calibration data")
            return

        # Simple linear regression example
        # In practice, use more sophisticated methods
        sim_data = np.array([x[0] for x in self.calibration_data])
        real_data = np.array([x[1] for x in self.calibration_data])

        # Learn affine transformation: real = A * sim + b
        A, b = self.estimate_affine_mapping(sim_data, real_data)

        self.sim_to_real_mapping = {'A': A, 'b': b}

    def estimate_affine_mapping(self, sim_data, real_data):
        """Estimate affine mapping between sim and real data"""
        # Add bias term to sim_data
        sim_data_bias = np.column_stack([sim_data, np.ones(sim_data.shape[0])])

        # Solve: real_data = [A|b] * [sim_data|1]
        params, _, _, _ = np.linalg.lstsq(sim_data_bias, real_data, rcond=None)

        A = params[:-1, :].T  # Extract transformation matrix
        b = params[-1, :]     # Extract bias

        return A, b

    def apply_correction(self, sim_obs):
        """Apply correction to simulation observation"""
        if not self.sim_to_real_mapping:
            return sim_obs  # Return as-is if no mapping learned

        A = self.sim_to_real_mapping['A']
        b = self.sim_to_real_mapping['b']

        corrected_obs = sim_obs @ A + b
        return corrected_obs

class RealityGapBridger:
    """Bridge the reality gap using multiple techniques"""

    def __init__(self):
        self.domain_randomizer = AdvancedDomainRandomizer(DomainParams())
        self.systematic_corrector = SystematicDifferencesCorrector()
        self.ensemble_learner = EnsembleLearner()

    def adapt_policy(self, policy, sim_env, real_env):
        """Adapt policy from sim to real using multiple techniques"""
        # 1. Train with domain randomization
        print("Training with domain randomization...")
        self.train_with_increasing_randomization(policy, sim_env)

        # 2. Collect calibration data
        print("Collecting calibration data...")
        self.collect_calibration_data(policy, sim_env, real_env)

        # 3. Learn systematic corrections
        print("Learning systematic corrections...")
        self.systematic_corrector.learn_mapping()

        # 4. Fine-tune on real data
        print("Fine-tuning on real data...")
        self.fine_tune_on_real(policy, real_env)

        return policy

    def train_with_increasing_randomization(self, policy, sim_env):
        """Gradually increase randomization during training"""
        for epoch in range(100):
            # Increase randomization based on epoch
            progress = epoch / 100.0
            self.domain_randomizer.update_curriculum(progress)

            # Train policy in randomized environment
            self.train_policy_epoch(policy, sim_env)

    def collect_calibration_data(self, policy, sim_env, real_env):
        """Collect paired sim/real data for systematic correction"""
        for i in range(100):  # Collect 100 pairs
            # Reset both environments
            sim_obs = sim_env.reset()
            real_obs = real_env.reset()

            # Store paired observations
            self.systematic_corrector.collect_calibration_data(sim_obs, real_obs)

            # Take same action in both environments
            action = policy.get_action(sim_obs)

            sim_obs, _, _ = sim_env.step(action)
            real_obs, _, _ = real_env.step(action)

            # Store paired observations
            self.systematic_corrector.collect_calibration_data(sim_obs, real_obs)

    def fine_tune_on_real(self, policy, real_env):
        """Fine-tune policy on real environment"""
        # Use small learning rate for fine-tuning
        policy.set_learning_rate(0.0001)

        for episode in range(50):  # Fine-tune for 50 episodes
            obs = real_env.reset()
            total_reward = 0

            for step in range(200):  # Max 200 steps per episode
                # Apply systematic correction if available
                corrected_obs = self.systematic_corrector.apply_correction(obs)

                action = policy.get_action(corrected_obs)
                obs, reward, done = real_env.step(action)
                total_reward += reward

                if done:
                    break

            print(f"Fine-tuning episode {episode}, reward: {total_reward}")

class EnsembleLearner:
    """Ensemble learning for robust sim-to-real transfer"""

    def __init__(self, num_models=5):
        self.num_models = num_models
        self.models = [self.create_model() for _ in range(num_models)]
        self.model_weights = [1.0/num_models] * num_models  # Equal weights initially

    def create_model(self):
        """Create a neural network model"""
        return nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 actions
        )

    def predict_ensemble(self, state):
        """Get ensemble prediction"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(state_tensor)
                predictions.append(pred)

        # Average predictions weighted by model weights
        weighted_pred = sum(w * p for w, p in zip(self.model_weights, predictions))
        return weighted_pred / len(predictions)

    def train_ensemble(self, data_loader, epochs=50):
        """Train all models in the ensemble"""
        for model_idx, model in enumerate(self.models):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(epochs):
                for batch_idx, (data, target) in enumerate(data_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.mse_loss(output, target)
                    loss.backward()
                    optimizer.step()

                if epoch % 10 == 0:
                    print(f"Model {model_idx}, Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Isaac-Specific Sim-to-Real Techniques

### Isaac Sim Integration

```python
# isaac_sim_integration.py
import omni
from omni.isaac.core import World, Actor
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.controllers import DifferentialController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import torch

class IsaacSimToRealTransfer:
    """Isaac Sim specific implementation for sim-to-real transfer"""

    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.domain_randomizer = DomainRandomizer()

    def setup_simulation(self):
        """Set up Isaac Sim environment"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets path")
            return False

        # Add robot to simulation
        robot_path = "/World/Robot"
        add_reference_to_stage(
            usd_path=f"{assets_root_path}/Isaac/Robots/Turtlebot/turtlebot3_kobuki.usd",
            prim_path=robot_path
        )

        # Create robot object
        self.robot = self.world.scene.add(
            Actor(
                prim_path=robot_path,
                name="turtlebot",
                position=np.array([0.0, 0.0, 0.1]),
                scale=np.array([1.0, 1.0, 1.0])
            )
        )

        return True

    def randomize_simulation_properties(self, epoch: int):
        """Randomize simulation properties based on training epoch"""
        # Calculate randomization intensity based on epoch
        intensity = min(epoch / 200.0, 1.0)  # Max intensity after 200 epochs

        # Randomize physics properties
        self.randomize_physics_materials(intensity)

        # Randomize lighting conditions
        self.randomize_lighting(intensity)

        # Randomize sensor properties
        self.randomize_sensors(intensity)

        # Randomize actuator properties
        self.randomize_actuators(intensity)

    def randomize_physics_materials(self, intensity: float):
        """Randomize physics materials for sim-to-real transfer"""
        # Create a physics material with randomized properties
        material_path = f"/World/PhysicsMaterial_{int(intensity*100)}"

        # Randomize friction based on intensity
        friction = 0.5 + (random.uniform(-0.5, 0.5) * intensity)
        friction = max(0.1, min(1.0, friction))  # Clamp between 0.1 and 1.0

        # Randomize restitution based on intensity
        restitution = 0.0 + (random.uniform(0.0, 0.2) * intensity)
        restitution = min(0.5, restitution)  # Clamp at 0.5

        # Create physics material
        physics_material = PhysicsMaterial(
            prim_path=material_path,
            static_friction=friction,
            dynamic_friction=friction,
            restitution=restitution
        )

        # Apply to relevant objects in the scene
        # This would be done based on the specific scene setup
        print(f"Applied physics material: friction={friction:.3f}, restitution={restitution:.3f}")

    def randomize_lighting(self, intensity: float):
        """Randomize lighting conditions"""
        # In Isaac Sim, you would modify light properties
        # This is a conceptual example
        light_intensity = 1000 + (random.uniform(-500, 500) * intensity)
        light_color = [
            1.0 + (random.uniform(-0.2, 0.2) * intensity),
            1.0 + (random.uniform(-0.2, 0.2) * intensity),
            1.0 + (random.uniform(-0.2, 0.2) * intensity)
        ]

        # Clamp color values
        light_color = [max(0.1, min(2.0, c)) for c in light_color]

        print(f"Applied lighting randomization: intensity={light_intensity:.1f}, color={light_color}")

    def randomize_sensors(self, intensity: float):
        """Randomize sensor properties"""
        # Add noise to sensors based on intensity
        sensor_noise_level = 0.0 + (random.uniform(0.0, 0.05) * intensity)
        sensor_bias = random.uniform(-0.01, 0.01) * intensity

        print(f"Applied sensor randomization: noise={sensor_noise_level:.4f}, bias={sensor_bias:.4f}")

    def randomize_actuators(self, intensity: float):
        """Randomize actuator properties"""
        # Randomize motor strength and response time
        motor_strength = 1.0 + (random.uniform(-0.2, 0.2) * intensity)
        response_delay = random.uniform(0, 3) * intensity  # in control steps

        print(f"Applied actuator randomization: strength={motor_strength:.3f}, delay={response_delay:.1f}")

    def train_with_randomization(self, policy, num_episodes=1000):
        """Train policy with increasing domain randomization"""
        self.setup_simulation()

        for episode in range(num_episodes):
            # Update randomization based on training progress
            progress = episode / num_episodes
            epoch = int(progress * 200)  # Map to 200 randomization epochs

            self.randomize_simulation_properties(epoch)

            # Reset world
            self.world.reset()

            # Execute episode with policy
            total_reward = 0
            for step in range(200):  # Max 200 steps per episode
                # Get robot observations
                obs = self.get_robot_observations()

                # Apply policy
                action = policy.get_action(obs)

                # Apply action to robot
                self.apply_robot_action(action)

                # Step simulation
                self.world.step(render=True)

                # Get reward
                reward = self.compute_reward()
                total_reward += reward

                if self.is_episode_done():
                    break

            # Log progress
            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}, Randomization Epoch: {epoch}")

    def get_robot_observations(self):
        """Get robot observations from Isaac Sim"""
        # This would get actual sensor data from Isaac Sim
        # For example: joint positions, velocities, IMU, camera, etc.

        # Simulated observation vector
        obs = np.random.random(24).astype(np.float32)  # Placeholder
        return obs

    def apply_robot_action(self, action):
        """Apply action to robot in Isaac Sim"""
        # This would apply actual motor commands to the robot
        # For differential drive robot, this might be wheel velocities
        pass

    def compute_reward(self):
        """Compute reward for current state"""
        # This would compute reward based on task
        # For example: reaching target, avoiding obstacles, etc.
        return 0.0  # Placeholder

    def is_episode_done(self):
        """Check if episode is done"""
        # This would check termination conditions
        return False  # Placeholder

def main():
    """Main function to demonstrate Isaac Sim to real transfer"""
    # Initialize the transfer framework
    transfer_framework = IsaacSimToRealTransfer()

    # Create a simple policy (placeholder)
    class SimplePolicy:
        def get_action(self, obs):
            # Return random action for demonstration
            return np.random.random(4).astype(np.float32)

    policy = SimplePolicy()

    # Train with domain randomization
    transfer_framework.train_with_randomization(policy)

    print("Training with domain randomization completed!")

if __name__ == "__main__":
    main()
```

## Assessment Questions

<Assessment
  question="What is the primary purpose of domain randomization in sim-to-real transfer?"
  type="multiple-choice"
  options={[
    "To make simulation run faster",
    "To randomize simulation parameters so the robot becomes robust to environmental variations",
    "To reduce the need for real-world training",
    "To make the simulation more visually appealing"
  ]}
  correctIndex={1}
  explanation="Domain randomization randomizes simulation parameters (physics, visuals, sensors) so that a robot trained in simulation becomes robust to variations it will encounter in the real world."
/>

<Assessment
  question="Which technique is most effective for bridging the sim-to-real gap?"
  type="multiple-choice"
  options={[
    "Using identical simulation and real-world conditions",
    "Applying domain randomization combined with systematic difference correction and fine-tuning",
    "Only training on real robots",
    "Using simpler simulation models"
  ]}
  correctIndex={1}
  explanation="The most effective approach combines multiple techniques: domain randomization during training, systematic difference correction, and fine-tuning on real data."
/>

## Summary

In this section, we've covered:

- Reinforcement learning fundamentals for robotics
- Isaac's RL framework and Gym integration
- Domain randomization techniques for sim-to-real transfer
- Systematic approaches to bridge the sim-to-real gap
- Practical implementation of transfer learning techniques

These techniques are essential for developing robots that can learn complex behaviors in simulation and successfully transfer those skills to the real world, significantly reducing the need for expensive real-world training.