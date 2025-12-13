---
title: URDF and SDF Robot Description Formats
sidebar_position: 5
description: Understanding URDF and SDF formats for robot description in simulation
duration: 180
difficulty: intermediate
learning_objectives:
  - Understand URDF and SDF formats for robot description
  - Create robot models using URDF for ROS integration
  - Use SDF for Gazebo simulation environments
  - Implement complex robot kinematics and dynamics
---

# URDF and SDF Robot Description Formats

## Learning Objectives

By the end of this section, you will be able to:
- Understand URDF and SDF formats for robot description
- Create robot models using URDF for ROS integration
- Use SDF for Gazebo simulation environments
- Implement complex robot kinematics and dynamics

## Introduction to Robot Description Languages

Robot description languages allow us to define robot models in a standardized way. The two main formats are:

- **URDF (Unified Robot Description Format)**: Used primarily by ROS/ROS 2 for robot representation
- **SDF (Simulation Description Format)**: Used by Gazebo for simulation

Both formats describe robot kinematics, dynamics, visual appearance, and sensors.

## URDF (Unified Robot Description Format)

### URDF Basics

URDF is an XML-based format that describes robots. A robot is represented as a tree of links connected by joints.

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joint -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.3 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
</robot>
```

### URDF Elements

#### Links
Links represent rigid bodies in the robot:
- **visual**: How the link appears visually
- **collision**: Collision geometry for physics simulation
- **inertial**: Mass, center of mass, and inertia tensor

#### Joints
Joints connect links:
- **fixed**: No movement between links
- **revolute**: Rotational joint with limits
- **continuous**: Revolute without limits
- **prismatic**: Linear sliding joint with limits
- **floating**: 6DOF connection
- **planar**: Movement on a plane

### Advanced URDF Features

#### Materials
```xml
<material name="Red">
  <color rgba="1 0 0 1"/>
</material>

<material name="Blue">
  <color rgba="0 0 1 1"/>
</material>
```

#### Transmission Elements
```xml
<transmission name="wheel_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

#### Gazebo Plugins
```xml
<gazebo>
  <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
    <left_joint>wheel_left_joint</left_joint>
    <right_joint>wheel_right_joint</right_joint>
    <wheel_separation>0.34</wheel_separation>
    <wheel_diameter>0.15</wheel_diameter>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_footprint</robot_base_frame>
  </plugin>
</gazebo>
```

## SDF (Simulation Description Format)

### SDF Basics

SDF is used by Gazebo for simulation. It can represent entire worlds with multiple robots and environments.

### Basic SDF Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>

      <!-- Visual element -->
      <visual name="chassis_visual">
        <geometry>
          <box>
            <size>1.0 0.5 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.8 1</ambient>
          <diffuse>0.3 0.3 0.9 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>

      <!-- Collision element -->
      <collision name="chassis_collision">
        <geometry>
          <box>
            <size>1.0 0.5 0.2</size>
          </box>
        </geometry>
      </collision>

      <!-- Inertial properties -->
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.416</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1.04</iyy>
          <iyz>0</iyz>
          <izz>1.25</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Joint -->
    <joint name="wheel_joint" type="revolute">
      <parent>chassis</parent>
      <child>wheel</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.570796</lower>
          <upper>1.570796</upper>
          <effort>10.0</effort>
          <velocity>3.14159</velocity>
        </limit>
      </axis>
    </joint>

    <link name="wheel">
      <visual name="wheel_visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </visual>
      <collision name="wheel_collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.005</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.005</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
```

### SDF vs URDF Comparison

| Feature | URDF | SDF |
|---------|------|-----|
| Primary Use | ROS robot description | Gazebo simulation |
| World Description | No | Yes |
| Multiple Models | No | Yes |
| Sensors | Through plugins | Native support |
| Controllers | Through plugins | Plugin system |
| Materials | Separate material elements | Inline in visual elements |

## Xacro for Complex URDF

Xacro is an XML macro language that extends URDF capabilities.

### Xacro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot_with_xacro">

  <!-- Properties -->
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.3" />

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="gray">
          <color rgba="0.5 0.5 0.5 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} 0.2"/>
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Instantiate wheels using macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.2 -0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="rear_left" parent="base_link" xyz="-0.2 0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="rear_right" parent="base_link" xyz="-0.2 -0.2 0" rpy="0 0 0"/>

</robot>
```

## Integration with ROS and Gazebo

### URDF to SDF Conversion

Gazebo can automatically convert URDF files to SDF for simulation:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or launch directly from URDF in Gazebo
roslaunch gazebo_ros spawn_model.launch file:=robot.urdf model_name:=my_robot
```

### Robot State Publisher

Use robot_state_publisher to broadcast joint states:

```xml
<node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />
```

### Joint State Publisher

For interactive joint control:

```xml
<node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher">
  <param name="use_gui" value="true"/>
</node>
```

## Best Practices

### URDF/SDF Design

1. **Consistent Naming**: Use clear, consistent naming conventions
2. **Proper Scaling**: Ensure units are consistent (usually meters, kilograms)
3. **Mass Properties**: Accurate inertial properties for realistic simulation
4. **Collision vs Visual**: Use simplified collision geometry for performance
5. **Origin Definitions**: Carefully define origins for joints and links

### Performance Optimization

1. **Simplified Meshes**: Use simplified collision meshes
2. **Appropriate Resolution**: Balance visual quality with performance
3. **Efficient Joints**: Minimize unnecessary joint constraints
4. **Material Optimization**: Use efficient material definitions

## Validation and Testing

### Checking URDF Validity

```bash
# Check URDF validity
check_urdf robot.urdf

# Visualize URDF
urdf_to_graphiz robot.urdf
```

### Testing in Simulation

```bash
# Launch robot in Gazebo
roslaunch gazebo_ros empty_world.launch
rosrun gazebo_ros spawn_model -file robot.urdf -urdf -model robot_name
```

## Interactive Elements

### Robot Description Formats Assessment

import Assessment from '@site/src/components/Assessment/Assessment';

<Assessment
  question="What is the primary difference between URDF and SDF?"
  options={[
    {id: 'a', text: 'URDF is for simulation, SDF is for robot description'},
    {id: 'b', text: 'URDF is for ROS robot description, SDF is for Gazebo simulation'},
    {id: 'c', text: 'URDF is more complex than SDF'},
    {id: 'd', text: 'There is no difference between URDF and SDF'}
  ]}
  correctAnswerId="b"
  explanation="URDF (Unified Robot Description Format) is primarily used by ROS for robot description, while SDF (Simulation Description Format) is used by Gazebo for simulation. Although they serve similar purposes, they are used in different contexts."
/>

## Summary

URDF and SDF are essential for representing robots in ROS and Gazebo respectively. Understanding both formats and how to use them effectively is crucial for robot simulation and development. Xacro provides powerful macro capabilities for creating complex, reusable robot models. Proper validation and optimization ensure realistic simulation performance.

In the next section, we'll explore physics simulation and sensor integration in detail.