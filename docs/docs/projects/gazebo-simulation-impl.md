---
title: Gazebo Simulation Implementation Project
sidebar_position: 2
description: Creating a complete robot simulation environment with Gazebo
duration: 300
difficulty: advanced
learning_objectives:
  - Design and implement a complete robot model for Gazebo simulation
  - Configure physics properties and sensor integration
  - Create custom Gazebo plugins for robot control
  - Implement realistic sensor simulation and visualization
---

# Gazebo Simulation Implementation Project

## Learning Objectives

By the end of this project, you will be able to:
- Design and implement a complete robot model for Gazebo simulation
- Configure physics properties and sensor integration
- Create custom Gazebo plugins for robot control
- Implement realistic sensor simulation and visualization

## Project Overview

This project requires you to develop a complete robot simulation environment in Gazebo. You'll create a robot model from scratch, implement realistic physics and sensor simulation, and integrate it with ROS 2 for control and perception tasks.

### Project Requirements

1. **Complete Robot Model**: Create a URDF/SDF model with realistic physics properties
2. **Sensor Integration**: Include multiple sensor types (camera, lidar, IMU, etc.)
3. **ROS 2 Integration**: Full integration with ROS 2 for control and perception
4. **World Environment**: Create a realistic simulation environment
5. **Control System**: Implement a control system for the robot

## Robot Model Design

### Physical Design Considerations

Your robot should be designed with the following specifications:

- **Differential Drive Base**: Two driven wheels with one or more castor wheels
- **Dimensions**: Reasonable size for navigation (e.g., 0.5m length × 0.3m width × 0.3m height)
- **Weight**: Appropriate mass distribution (e.g., 10-20 kg total)
- **Actuators**: Wheel motors with realistic torque and speed limits
- **Sensors**: At least 3 different sensor types

### URDF Model Structure

Create a comprehensive URDF model:

```xml
<?xml version="1.0"?>
<robot name="simulation_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Properties -->
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_height" value="0.15" />
  <xacro:property name="wheel_radius" value="0.05" />
  <xacro:property name="wheel_width" value="0.03" />
  <xacro:property name="wheel_offset_x" value="0.15" />
  <xacro:property name="wheel_offset_y" value="0.15" />
  <xacro:property name="caster_offset" value="0.2" />

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}" />
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}" />
      </geometry>
    </collision>

    <inertial>
      <mass value="15.0" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.4" />
    </inertial>
  </link>

  <!-- Left Wheel -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link" />
    <child link="left_wheel_link" />
    <origin xyz="${wheel_offset_x} ${wheel_offset_y} 0" rpy="0 0 0" />
    <axis xyz="0 1 0" />
  </joint>

  <link name="left_wheel_link">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0" />
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}" />
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0" />
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}" />
      </geometry>
    </collision>

    <inertial>
      <mass value="0.5" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002" />
    </inertial>
  </link>

  <!-- Right Wheel -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link" />
    <child link="right_wheel_link" />
    <origin xyz="${wheel_offset_x} ${-wheel_offset_y} 0" rpy="0 0 0" />
    <axis xyz="0 1 0" />
  </joint>

  <link name="right_wheel_link">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0" />
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}" />
      </geometry>
      <material name="black" />
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0" />
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}" />
      </geometry>
    </collision>

    <inertial>
      <mass value="0.5" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002" />
    </inertial>
  </link>

  <!-- Caster Wheel -->
  <joint name="caster_joint" type="fixed">
    <parent link="base_link" />
    <child link="caster_wheel" />
    <origin xyz="${-caster_offset} 0 0" rpy="0 0 0" />
  </joint>

  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="0.04" />
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1" />
      </material>
    </visual>

    <collision>
      <geometry>
        <sphere radius="0.04" />
      </geometry>
    </collision>

    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001" />
    </inertial>
  </link>

  <!-- Sensors -->
  <!-- Camera -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link" />
    <child link="camera_link" />
    <origin xyz="${base_length/2 - 0.05} 0 ${base_height/2}" rpy="0 0 0" />
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05" />
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1" />
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.05 0.05 0.05" />
      </geometry>
    </collision>

    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001" />
    </inertial>
  </link>

  <!-- IMU -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link" />
    <child link="imu_link" />
    <origin xyz="0 0 0" rpy="0 0 0" />
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001" />
    </inertial>
  </link>

  <!-- Transmissions for ROS Control -->
  <transmission name="left_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Gazebo Plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_wheel_link">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="right_wheel_link">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="caster_wheel">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="camera_link">
    <material>Gazebo/Red</material>
  </gazebo>

  <!-- Differential Drive Controller -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>robot</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <update_rate>30</update_rate>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>${2 * wheel_offset_y}</wheel_separation>
      <wheel_diameter>${2 * wheel_radius}</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>10.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
    </plugin>
  </gazebo>

  <!-- Camera Sensor -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>300</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>robot</namespace>
          <remapping>~/image_raw:=camera/image_raw</remapping>
          <remapping>~/camera_info:=camera/camera_info</remapping>
        </ros>
        <camera_name>camera</camera_name>
        <frame_name>camera_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU Sensor -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <ros>
          <namespace>robot</namespace>
          <remapping>~/out:=imu/data</remapping>
        </ros>
        <frame_name>imu_link</frame_name>
        <topic_name>imu/data</topic_name>
        <gaussian_noise>0.0017</gaussian_noise>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Laser Scanner -->
  <joint name="laser_joint" type="fixed">
    <parent link="base_link" />
    <child link="laser_link" />
    <origin xyz="${base_length/2 - 0.02} 0 ${base_height/2 + 0.05}" rpy="0 0 0" />
  </joint>

  <link name="laser_link">
    <visual>
      <geometry>
        <cylinder radius="0.02" length="0.04" />
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1" />
      </material>
    </visual>

    <collision>
      <geometry>
        <cylinder radius="0.02" length="0.04" />
      </geometry>
    </collision>

    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" rpy="0 0 0" />
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0002" />
    </inertial>
  </link>

  <gazebo reference="laser_link">
    <sensor name="laser_scanner" type="ray">
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
        <ros>
          <namespace>robot</namespace>
          <remapping>~/out:=scan</remapping>
        </ros>
        <frame_name>laser_link</frame_name>
        <topic_name>scan</topic_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

## Simulation Environment

### Creating a World File

Create a comprehensive world file with obstacles and navigation challenges:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simulation_world">
    <!-- Physics -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>

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

    <!-- Environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Walls -->
    <model name="wall_1">
      <pose>0 5 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_2">
      <pose>5 0 0.5 0 0 1.5708</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacles -->
    <model name="obstacle_1">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.4 0.0 1</ambient>
            <diffuse>1.0 0.5 0.0 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>0.625</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.625</iyy>
            <iyz>0</iyz>
            <izz>0.45</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>-2 -2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.8 0.8 1.0</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.8 0.8 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.0 0.8 1</ambient>
            <diffuse>0.5 0.0 1.0 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>0.833</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.833</iyy>
            <iyz>0</iyz>
            <izz>0.533</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Navigation Target -->
    <model name="navigation_target">
      <pose>3 -3 0.05 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.0 0.8 0.0 0.5</ambient>
            <diffuse>0.0 1.0 0.0 0.5</diffuse>
            <specular>0.1 0.1 0.1 0.5</specular>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

## Control System Implementation

### ROS 2 Navigation Stack

Implement a basic navigation system:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs
import numpy as np
import math


class SimulationNavigator(Node):
    def __init__(self):
        super().__init__('simulation_navigator')

        # Parameters
        self.declare_parameter('target_x', 3.0)
        self.declare_parameter('target_y', -3.0)
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('safe_distance', 0.5)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

        # Variables
        self.current_pose = None
        self.target_pose = PoseStamped()
        self.target_pose.pose.position.x = self.get_parameter('target_x').value
        self.target_pose.pose.position.y = self.get_parameter('target_y').value
        self.laser_ranges = []

        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Simulation Navigator initialized')

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        self.laser_ranges = msg.ranges

    def get_distance_to_target(self):
        if self.current_pose is None:
            return float('inf')

        dx = self.target_pose.pose.position.x - self.current_pose.position.x
        dy = self.target_pose.pose.position.y - self.current_pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def get_angle_to_target(self):
        if self.current_pose is None:
            return 0.0

        # Calculate angle to target
        dx = self.target_pose.pose.position.x - self.current_pose.position.x
        dy = self.target_pose.pose.position.y - self.current_pose.position.y
        target_angle = math.atan2(dy, dx)

        # Get current orientation (assuming quaternion to yaw conversion)
        q = self.current_pose.orientation
        current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Calculate angle difference
        angle_diff = target_angle - current_yaw
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        return angle_diff

    def control_loop(self):
        if self.current_pose is None:
            return

        cmd_msg = Twist()

        # Check for obstacles
        if len(self.laser_ranges) > 0:
            min_distance = min([r for r in self.laser_ranges if not math.isnan(r)])
            if min_distance < self.get_parameter('safe_distance').value:
                # Emergency stop if too close to obstacle
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m, stopping!')
                return

        # Calculate distance and angle to target
        distance_to_target = self.get_distance_to_target()
        angle_to_target = self.get_angle_to_target()

        # Navigation logic
        if distance_to_target > 0.2:  # Still need to move
            # Rotate toward target if significantly off course
            if abs(angle_to_target) > 0.2:
                cmd_msg.angular.z = self.get_parameter('angular_speed').value * np.sign(angle_to_target)
            else:
                # Move forward if facing the right direction
                cmd_msg.linear.x = self.get_parameter('linear_speed').value
        else:
            # Reached target
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.get_logger().info('Target reached!')

        self.cmd_vel_pub.publish(cmd_msg)

        # Log status
        self.get_logger().info(
            f'Distance to target: {distance_to_target:.2f}m, '
            f'Angle to target: {math.degrees(angle_to_target):.2f}°'
        )


def main(args=None):
    rclpy.init(args=args)

    navigator = SimulationNavigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch Files

Create a launch file to start the complete simulation:

```python
# launch/simulation_project_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    world_name = DeclareLaunchArgument(
        'world_name',
        default_value='simulation_world',
        description='Choose one of the world files from `/models` or specify full path'
    )

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('simulation_robot_description'),
                'worlds',
                LaunchConfiguration('world_name')
            ]),
            'verbose': 'false',
        }.items()
    )

    # Robot spawn node
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'simulation_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.1',
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('/joint_states', 'joint_states'),
        ]
    )

    # Static transform publisher for odom
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
    )

    # Navigation controller
    navigation_controller = Node(
        package='simulation_navigation',
        executable='navigator',
        name='simulation_navigator',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'target_x': 3.0},
            {'target_y': -3.0},
            {'linear_speed': 0.3},
            {'angular_speed': 0.5},
            {'safe_distance': 0.5},
        ],
        remappings=[
            ('cmd_vel', 'cmd_vel'),
            ('odom', 'odom'),
            ('scan', 'scan'),
        ]
    )

    return LaunchDescription([
        use_sim_time,
        world_name,
        gazebo,
        spawn_robot,
        robot_state_publisher,
        static_transform_publisher,
        navigation_controller,
    ])
```

## Testing and Validation

### Simulation Testing

Test your simulation with the following scenarios:

1. **Basic Movement**: Verify the robot can move forward, backward, and turn
2. **Sensor Data**: Check that all sensors provide reasonable data
3. **Obstacle Avoidance**: Test navigation with obstacles in the environment
4. **Target Navigation**: Verify the robot can navigate to a target location

### Performance Metrics

Evaluate your simulation based on:

- **Physics Accuracy**: Does the robot move realistically?
- **Sensor Quality**: Are sensor readings realistic and useful?
- **Navigation Performance**: How efficiently does the robot navigate?
- **Stability**: Does the simulation run smoothly without instabilities?

## Assessment Criteria

### Technical Implementation (70%)

- **Robot Model** (20%): Complete, realistic URDF/SDF model with proper physics
- **Sensor Integration** (20%): Proper sensor configuration and realistic simulation
- **Control System** (15%): Functional navigation and control algorithms
- **Simulation Environment** (15%): Realistic world with appropriate challenges

### Documentation and Testing (30%)

- **Documentation** (10%): Clear documentation of the model and system
- **Testing** (10%): Comprehensive testing of all components
- **Validation** (10%): Demonstration of system functionality

## Project Deliverables

1. **Complete Robot Model**: URDF/SDF files with all necessary configurations
2. **Simulation Environment**: World file with obstacles and navigation challenges
3. **Control System**: ROS 2 nodes for robot control and navigation
4. **Launch Files**: Complete launch system to start the simulation
5. **Documentation**: README with setup and usage instructions
6. **Test Results**: Demonstration of system functionality

## Resources

- [Gazebo Documentation](http://gazebosim.org/)
- [ROS 2 Gazebo Integration](https://github.com/ros-simulation/gazebo_ros_pkgs)
- [URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [ROS 2 Navigation Stack](https://navigation.ros.org/)

## Extension Ideas

- Implement SLAM capabilities
- Add more complex sensors (3D lidar, stereo cameras)
- Implement advanced navigation algorithms (A*, Dijkstra)
- Create dynamic obstacles that move in the environment
- Implement multi-robot simulation scenarios

This project provides a comprehensive understanding of robot simulation in Gazebo with ROS 2 integration, preparing you for more advanced robotics applications.