---
title: Gazebo Simulation Environment Setup
sidebar_position: 1
description: Setting up and configuring the Gazebo simulation environment for robotics development
duration: 150
difficulty: intermediate
learning_objectives:
  - Install and configure Gazebo for robotics simulation
  - Understand Gazebo's architecture and components
  - Set up a basic simulation environment
  - Configure physics properties and world settings
---

# Gazebo Simulation Environment Setup

## Learning Objectives

By the end of this section, you will be able to:
- Install and configure Gazebo for robotics simulation
- Understand Gazebo's architecture and components
- Set up a basic simulation environment
- Configure physics properties and world settings

## Introduction to Gazebo

Gazebo is a powerful 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in robotics research and development for testing algorithms, validating designs, and training AI systems.

### Key Features

- **Realistic Physics**: Accurate simulation of rigid body dynamics, contact forces, and collisions
- **High-Quality Graphics**: Advanced rendering with support for shadows, reflections, and lighting
- **Sensor Simulation**: Realistic simulation of cameras, lidars, IMUs, GPS, and other sensors
- **Plugin Architecture**: Extensible system for custom sensors, controllers, and world interactions
- **ROS Integration**: Seamless integration with ROS and ROS 2 for robot simulation

## Installation

### Installing Gazebo Garden

Gazebo Garden is the latest version at the time of writing. Install it using the appropriate method for your system:

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install gazebo libgazebo-dev
```

#### Using ROS 2 Installation
If you have ROS 2 installed, Gazebo often comes as a dependency:
```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros-control
```

### Verifying Installation

Test that Gazebo is properly installed:
```bash
gz sim --version
# or for older versions
gazebo --version
```

Launch Gazebo to verify it works:
```bash
gz sim
# or for older versions
gazebo
```

## Gazebo Architecture

### Core Components

Gazebo consists of several key components:

1. **Gazebo Server (gzserver)**: The physics simulation engine
2. **Gazebo Client (gzclient)**: The graphical user interface
3. **Gazebo Transport**: Communication layer between components
4. **Gazebo Plugins**: Extensibility system for custom functionality

### New vs. Old Architecture

Gazebo has evolved from libgazebo to the new gz-sim architecture:
- **Old**: `gazebo`, `libgazebo`
- **New**: `gz sim`, `gz-sim` (Garden and later)

## Basic Simulation Setup

### Creating a Simple World

Create a basic world file (`basic_world.sdf`):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_world">
    <!-- Include a default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a default sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.3 0.3 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Launching a Custom World

Launch Gazebo with your custom world:
```bash
gz sim -r basic_world.sdf
# or for older versions
gazebo -world basic_world.world
```

## Physics Configuration

### Physics Engine Settings

Gazebo supports different physics engines. Configure in your world file:

```xml
<world name="physics_world">
  <!-- Physics engine configuration -->
  <physics name="1ms" type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    <gravity>0 0 -9.8</gravity>

    <!-- ODE-specific settings -->
    <ode>
      <solver>
        <type>quick</type>
        <iters>10</iters>
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>0.0</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>

  <!-- Rest of world definition -->
</world>
```

### Understanding Physics Parameters

- **max_step_size**: Simulation time step (smaller = more accurate but slower)
- **real_time_factor**: Target simulation speed relative to real time
- **real_time_update_rate**: Updates per second
- **gravity**: Gravity vector (x, y, z components)

## World Configuration

### Environment Settings

Configure environmental properties in your world file:

```xml
<world name="environment_world">
  <!-- Physics -->
  <physics name="dynamics" type="ode">
    <gravity>0 0 -9.8</gravity>
  </physics>

  <!-- Audio -->
  <audio>
    <device>default</device>
  </audio>

  <!-- Wind -->
  <wind>
    <linear_velocity>0.5 0 0</linear_velocity>
  </wind>

  <!-- Spherical coordinates (for GPS simulation) -->
  <spherical_coordinates>
    <surface_model>EARTH_WGS84</surface_model>
    <latitude_deg>37.405</latitude_deg>
    <longitude_deg>-122.079</longitude_deg>
    <elevation>0.0</elevation>
    <heading_deg>0</heading_deg>
  </spherical_coordinates>

  <!-- Models and other elements -->
</world>
```

## Sensor Simulation

### Camera Sensor

Add a camera sensor to a model:

```xml
<model name="camera_robot">
  <link name="camera_link">
    <!-- Visual and collision properties -->
    <visual name="visual">
      <geometry>
        <box><size>0.1 0.1 0.1</size></box>
      </geometry>
    </visual>
    <collision name="collision">
      <geometry>
        <box><size>0.1 0.1 0.1</size></box>
      </geometry>
    </collision>

    <!-- Camera sensor -->
    <sensor name="camera" type="camera">
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
    </sensor>
  </link>
</model>
```

### Lidar Sensor

Add a 3D lidar sensor:

```xml
<sensor name="lidar_3d" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>640</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>    <!-- 90 degrees -->
      </horizontal>
      <vertical>
        <samples>8</samples>
        <resolution>1</resolution>
        <min_angle>-0.174533</min_angle>  <!-- -10 degrees -->
        <max_angle>0.174533</max_angle>   <!-- 10 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

## ROS 2 Integration

### Gazebo ROS Packages

Install the necessary ROS 2 Gazebo packages:

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros-control ros-humble-ros2-control ros-humble-ros2-controllers
```

### Launching Gazebo with ROS 2

Create a launch file to start Gazebo with ROS 2 integration:

```python
# launch/gazebo_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # World file argument
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/gazebo_ros_pkgs/gazebo_ros/worlds`'
    )

    # Gazebo server
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
        }.items()
    )

    # Gazebo client
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gzclient.launch.py')
        )
    )

    return LaunchDescription([
        world_arg,
        gzserver,
        gzclient,
    ])
```

### Spawning Robots

Use the spawn entity service to add robots to the simulation:

```python
# Python script to spawn a robot
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity

class SpawnRobot(Node):
    def __init__(self):
        super().__init__('spawn_robot')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def spawn_robot(self, robot_name, robot_xml, robot_namespace):
        req = SpawnEntity.Request()
        req.name = robot_name
        req.xml = robot_xml
        req.robot_namespace = robot_namespace
        req.initial_pose.position.x = 0.0
        req.initial_pose.position.y = 0.0
        req.initial_pose.position.z = 0.0

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f'Successfully spawned {robot_name}')
        else:
            self.get_logger().error(f'Failed to spawn {robot_name}')

def main(args=None):
    rclpy.init(args=args)
    spawn_robot = SpawnRobot()

    # Load robot XML from file or define here
    robot_xml = """<robot name="my_robot">
        <!-- Robot definition -->
    </robot>"""

    spawn_robot.spawn_robot('my_robot', robot_xml, '')

    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### Reducing Simulation Load

Optimize simulation performance by adjusting settings:

```xml
<physics name="optimized_physics" type="ode">
  <!-- Larger time step for better performance -->
  <max_step_size>0.01</max_step_size>

  <!-- Lower update rate if real-time performance isn't critical -->
  <real_time_update_rate>100</real_time_update_rate>

  <!-- Simplified collision geometry -->
  <ode>
    <solver>
      <iters>20</iters>  <!-- Reduce from default 50 -->
    </solver>
  </ode>
</physics>
```

### Visual Optimization

Disable unnecessary visual elements for headless simulation:

```bash
# Run without GUI
gz sim -s -r my_world.sdf
# or for older versions
gazebo -s -u my_world.world
```

## Troubleshooting Common Issues

### Slow Simulation

- Check physics step size and real-time update rate
- Reduce complexity of collision meshes
- Limit the number of active sensors
- Consider using simpler physics engine settings

### Model Spawning Issues

- Verify URDF/SDF model syntax
- Check that all referenced mesh files exist
- Ensure proper permissions on model files
- Validate that ROS packages are properly sourced

### Sensor Data Problems

- Verify sensor topics are being published
- Check sensor configuration in URDF/SDF
- Ensure Gazebo ROS plugins are loaded
- Validate sensor noise parameters

## Interactive Elements

### Gazebo Simulation Quiz

import Assessment from '@site/src/components/Assessment/Assessment';

<Assessment
  question="What is the primary purpose of the max_step_size parameter in Gazebo physics configuration?"
  options={[
    {id: 'a', text: 'To control the rendering frame rate'},
    {id: 'b', text: 'To determine the simulation time step for physics calculations'},
    {id: 'c', text: 'To limit the maximum velocity of objects'},
    {id: 'd', text: 'To set the maximum size of models'}
  ]}
  correctAnswerId="b"
  explanation="The max_step_size parameter determines the time step used for physics calculations in the simulation. Smaller values provide more accurate physics but require more computation."
/>

## Summary

Gazebo provides a powerful simulation environment for robotics development with realistic physics, sensor simulation, and ROS integration. Proper setup and configuration are essential for effective simulation. Understanding the architecture, physics settings, and integration with ROS 2 allows you to create realistic and efficient simulation environments for testing and development.

In the next section, we'll explore URDF and SDF robot description formats in detail.