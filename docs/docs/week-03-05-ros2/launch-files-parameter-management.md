---
title: Launch Files and Parameter Management
sidebar_position: 4
description: Managing ROS 2 nodes and parameters through launch files and configuration
duration: 120
difficulty: intermediate
learning_objectives:
  - Create and use launch files to manage multiple nodes
  - Configure parameters using YAML files and launch files
  - Implement conditional node launching
  - Use launch arguments for flexible configurations
---

# Launch Files and Parameter Management

## Learning Objectives

By the end of this section, you will be able to:
- Create and use launch files to manage multiple nodes
- Configure parameters using YAML files and launch files
- Implement conditional node launching
- Use launch arguments for flexible configurations

## Launch System Overview

The ROS 2 launch system provides a Python-based framework for starting multiple nodes and configuring their parameters. It replaces the XML-based launch system from ROS 1 with a more flexible and powerful Python-based approach.

### Why Use Launch Files?

- **Convenience**: Start multiple nodes with a single command
- **Configuration**: Set parameters and configurations for all nodes
- **Flexibility**: Use arguments to change behavior without modifying files
- **Organization**: Group related nodes together logically
- **Testing**: Create specific configurations for different scenarios

## Basic Launch File Structure

### Simple Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='my_talker'
        ),
        Node(
            package='demo_nodes_cpp',
            executable='listener',
            name='my_listener'
        )
    ])
```

### Launch File with Parameters

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                {'wheel_radius': 0.05},
                {'base_frame': 'base_link'},
                {'odom_frame': 'odom'},
            ]
        )
    ])
```

## Launch Arguments

Launch arguments allow you to customize launch files without modifying them:

### Adding Arguments to Launch Files

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='turtlebot4',
        description='Name of the robot'
    )

    return LaunchDescription([
        # Add argument declarations
        use_sim_time,
        robot_name,

        # Use arguments in nodes
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name=[LaunchConfiguration('robot_name'), '_controller'],
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'robot_name': LaunchConfiguration('robot_name')},
            ]
        )
    ])
```

### Using Arguments When Launching

```bash
# Use default values
ros2 launch my_robot_package my_launch_file.py

# Override specific arguments
ros2 launch my_robot_package my_launch_file.py use_sim_time:=true robot_name:=my_robot
```

## Advanced Launch Features

### Conditional Launch

Launch nodes based on conditions:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        package='rviz2',
        executable='rviz2',
        name='rviz2'
    )

    return LaunchDescription([
        use_rviz,
        rviz_node,
        # Other nodes...
    ])
```

### Launch Included Files

Include other launch files in your launch file:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Include another launch file
    other_launch_file = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('other_package'),
                'launch',
                'other_launch_file.py'
            ])
        ])
    )

    return LaunchDescription([
        other_launch_file,
        # Additional nodes...
    ])
```

## YAML Parameter Files

### Creating Parameter Files

Create parameter files in a `config` directory:

```yaml
# config/robot_params.yaml
robot_controller:
  ros__parameters:
    # Basic configuration
    robot_name: 'turtlebot4'
    use_sim_time: false

    # Hardware parameters
    wheel_radius: 0.05
    wheel_separation: 0.3
    encoder_resolution: 4096

    # Control parameters
    max_linear_velocity: 0.5
    max_angular_velocity: 1.0
    linear_acceleration: 1.0
    angular_acceleration: 2.0

    # Sensor parameters
    laser_scan_topic: '/scan'
    camera_topic: '/camera/image_raw'

    # Navigation parameters
    planner_frequency: 5.0
    controller_frequency: 20.0
    recovery_enabled: true
```

### Loading Parameters from YAML

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare parameter file argument
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            get_package_share_directory('my_robot_package'),
            'config',
            'robot_params.yaml'
        ]),
        description='Full path to the ROS2 parameters file'
    )

    return LaunchDescription([
        params_file,
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        )
    ])
```

## Complex Parameter Management

### Multiple Parameter Sources

You can combine multiple parameter sources:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Define configuration
    config_file = PathJoinSubstitution([
        get_package_share_directory('my_robot_package'),
        'config',
        'robot_params.yaml'
    ])

    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                config_file,  # From YAML file
                {'robot_name': 'custom_robot'},  # Override specific parameter
                {'use_sim_time': True},  # Additional parameter
            ],
            output='screen'
        )
    ])
```

### Parameter Remapping

Remap topics, services, and parameters:

```python
Node(
    package='my_robot_package',
    executable='robot_controller',
    name='robot_controller',
    parameters=[
        {'robot_name': 'turtlebot4'},
    ],
    remappings=[
        ('/original_topic', '/remapped_topic'),
        ('/cmd_vel', '/navigation/cmd_vel'),
        ('/scan', '/laser_scan'),
    ]
)
```

## Launch Actions

### Execute Processes

Run external processes alongside ROS nodes:

```python
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start a background process
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-a'],
            output='screen'
        ),

        # Start ROS nodes
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker'
        )
    ])
```

### Timer Actions

Delay node startup or run actions after a delay:

```python
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker'
        ),
        # Start listener 5 seconds after talker
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='demo_nodes_cpp',
                    executable='listener',
                    name='listener'
                )
            ]
        )
    ])
```

## Best Practices for Launch Files

### Organize by Functionality

Group related functionality in separate launch files:

```
launch/
├── robot.launch.py          # Core robot nodes
├── navigation.launch.py     # Navigation stack
├── perception.launch.py     # Perception stack
├── visualization.launch.py  # RViz and other visualization
└── bringup.launch.py        # Includes all of the above
```

### Use Descriptive Names

```python
# Good: Descriptive names
Node(
    package='my_robot_driver',
    executable='wheel_odometry_node',
    name='wheel_odometry'
)

# Less good: Generic names
Node(
    package='my_robot_driver',
    executable='node',
    name='n1'
)
```

### Modular Design

Create reusable launch file components:

```python
# launch/robot_description.py
from launch import LaunchDescription
from launch_ros.actions import Node

def get_robot_description_nodes():
    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[
                {'robot_description': Command(['xacro ', FindFile('my_robot_description', 'urdf/my_robot.urdf.xacro')])}
            ]
        )
    ]

# launch/bringup.py
from launch import LaunchDescription
from launch_ros.actions import Node
from .robot_description import get_robot_description_nodes

def generate_launch_description():
    nodes = get_robot_description_nodes()
    nodes.extend([
        Node(
            package='my_robot_driver',
            executable='driver_node',
            name='driver'
        )
    ])

    return LaunchDescription(nodes)
```

## Common Launch Patterns

### Sensor Processing Pipeline

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    return LaunchDescription([
        use_sim_time,

        # Point cloud processing
        Node(
            package='my_robot_perception',
            executable='point_cloud_filter',
            name='point_cloud_filter',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
            remappings=[
                ('input_cloud', '/velodyne_points'),
                ('output_cloud', '/filtered_points'),
            ]
        ),

        # Object detection
        Node(
            package='my_robot_perception',
            executable='object_detector',
            name='object_detector',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
            remappings=[
                ('input_cloud', '/filtered_points'),
                ('detections', '/object_detections'),
            ]
        )
    ])
```

### Multi-Robot Launch

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    # Robot-specific arguments
    robot1_namespace = DeclareLaunchArgument(
        'robot1_namespace',
        default_value='robot1',
        description='Namespace for robot 1'
    )

    robot2_namespace = DeclareLaunchArgument(
        'robot2_namespace',
        default_value='robot2',
        description='Namespace for robot 2'
    )

    return LaunchDescription([
        robot1_namespace,
        robot2_namespace,

        # Robot 1 nodes
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='controller',
            namespace=LaunchConfiguration('robot1_namespace'),
            parameters=[
                {'robot_name': [LaunchConfiguration('robot1_namespace'), TextSubstitution(text='_robot')]},
            ]
        ),

        # Robot 2 nodes
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='controller',
            namespace=LaunchConfiguration('robot2_namespace'),
            parameters=[
                {'robot_name': [LaunchConfiguration('robot2_namespace'), TextSubstitution(text='_robot')]},
            ]
        )
    ])
```

## Interactive Elements

### Launch System Assessment

import Assessment from '@site/src/components/Assessment/Assessment';

<Assessment
  question="What is the primary purpose of launch arguments in ROS 2 launch files?"
  options={[
    {id: 'a', text: 'To permanently change the launch file configuration'},
    {id: 'b', text: 'To provide flexible configurations without modifying the launch file'},
    {id: 'c', text: 'To create multiple launch files automatically'},
    {id: 'd', text: 'To speed up the launch process'}
  ]}
  correctAnswerId="b"
  explanation="Launch arguments provide a way to customize launch file behavior without modifying the file itself, allowing for flexible configurations for different scenarios."
/>

## Summary

Launch files and parameter management are essential tools for organizing and configuring ROS 2 systems. They allow you to start multiple nodes with a single command, manage complex parameter configurations, and create flexible, reusable system configurations. Proper use of launch files improves the maintainability and usability of robotic systems.

In the next week, we'll explore robot simulation with Gazebo, where we'll see how launch files are used to start simulation environments alongside robot controllers.