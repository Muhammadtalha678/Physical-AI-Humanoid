---
title: Physics Simulation and Sensor Integration
sidebar_position: 6
description: Understanding physics simulation in Gazebo and integrating various sensors
duration: 200
difficulty: advanced
learning_objectives:
  - Configure physics properties and parameters for realistic simulation
  - Implement sensor simulation including cameras, lidars, and IMUs
  - Understand collision detection and contact mechanics
  - Integrate sensors with ROS 2 for real-time feedback
---

# Physics Simulation and Sensor Integration

## Learning Objectives

By the end of this section, you will be able to:
- Configure physics properties and parameters for realistic simulation
- Implement sensor simulation including cameras, lidars, and IMUs
- Understand collision detection and contact mechanics
- Integrate sensors with ROS 2 for real-time feedback

## Physics Simulation Fundamentals

### Understanding the Physics Engine

Gazebo uses Open Dynamics Engine (ODE), Bullet, or DART as its underlying physics engines. Each engine has different strengths:

- **ODE**: Fast and lightweight, good for real-time simulation
- **Bullet**: More accurate collision detection, good for complex geometries
- **DART**: Advanced contact mechanics and soft body simulation

### Physics Parameters

The physics configuration determines how objects behave in the simulation:

```xml
<physics name="dynamics" type="ode">
  <!-- Time stepping -->
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- Solver parameters -->
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
```

### Key Physics Parameters Explained

- **max_step_size**: Simulation time step - smaller values are more accurate but slower
- **real_time_factor**: Target simulation speed relative to real time
- **real_time_update_rate**: How often physics is updated per second
- **CFM (Constraint Force Mixing)**: Softness of constraints (higher = softer)
- **ERP (Error Reduction Parameter)**: How quickly constraint errors are corrected

## Collision Detection

### Collision Geometry Types

Different collision geometries serve different purposes:

#### Primitive Shapes
- **Box**: Fast, simple collision detection
- **Sphere**: Fast, rotationally symmetric
- **Cylinder**: Good for wheels and simple cylindrical objects
- **Capsule**: Smooth cylinder with rounded ends

#### Complex Shapes
- **Mesh**: Exact collision based on mesh geometry
- **Heightmap**: Terrain collision from elevation data

### Collision Parameters

```xml
<link name="collision_link">
  <collision name="collision">
    <geometry>
      <mesh>
        <uri>package://my_robot_description/meshes/collision.stl</uri>
      </mesh>
    </geometry>

    <!-- Surface properties -->
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>
          <mu2>1.0</mu2>
          <fdir1>0 0 1</fdir1>
          <slip1>0.0</slip1>
          <slip2>0.0</slip2>
        </ode>
        <torsional>
          <coefficient>1.0</coefficient>
          <use_patch_radius>false</use_patch_radius>
          <surface_radius>0.01</surface_radius>
        </torsional>
      </friction>

      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>

      <contact>
        <collide_without_contact>false</collide_without_contact>
        <collide_without_contact_bitmask>1</collide_without_contact_bitmask>
        <collide_bitmask>15</collide_bitmask>
        <ode>
          <soft_cfm>0</soft_cfm>
          <soft_erp>0.2</soft_erp>
          <kp>1000000000000.0</kp>
          <kd>1.0</kd>
          <max_vel>100.0</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### Contact Mechanics

Understanding how objects interact when they collide:

- **Restitution**: Bounciness (0 = no bounce, 1 = perfectly elastic)
- **Friction**: Resistance to sliding motion
- **Contact stiffness**: How stiff the contact is
- **Contact damping**: How much energy is absorbed during contact

## Sensor Simulation

### Camera Sensors

Camera sensors simulate RGB cameras for computer vision applications:

```xml
<sensor name="camera" type="camera">
  <camera name="head_camera">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
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
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### Depth Camera Sensors

Depth cameras provide both RGB and depth information:

```xml
<sensor name="depth_camera" type="depth">
  <camera name="head_depth_camera">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <depth_camera>
      <output>depths</output>
    </depth_camera>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### Lidar Sensors

Lidar sensors simulate 2D or 3D laser range finders:

#### 2D Lidar
```xml
<sensor name="laser_2d" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>  <!-- -π -->
        <max_angle>3.14159</max_angle>   <!-- π -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

#### 3D Lidar
```xml
<sensor name="laser_3d" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>640</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>64</samples>
        <resolution>1</resolution>
        <min_angle>-0.5236</min_angle>  <!-- -30 degrees -->
        <max_angle>0.5236</max_angle>   <!-- 30 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### IMU Sensors

IMU sensors simulate accelerometers, gyroscopes, and magnetometers:

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-05</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-05</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-05</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

### GPS Sensors

GPS sensors simulate location data:

```xml
<sensor name="gps_sensor" type="gps">
  <always_on>1</always_on>
  <update_rate>1</update_rate>
  <gps>
    <position_sensing>
      <horizontal>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.2</stddev>
        </noise>
      </horizontal>
      <vertical>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.2</stddev>
        </noise>
      </vertical>
    </position_sensing>
  </gps>
</sensor>
```

## ROS 2 Integration

### Sensor Plugins

Connect sensors to ROS 2 using plugins:

```xml
<gazebo>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>camera</namespace>
      <remapping>~/image_raw:=image</remapping>
      <remapping>~/camera_info:=camera_info</remapping>
    </ros>
    <camera_name>head_camera</camera_name>
    <frame_name>camera_optical_frame</frame_name>
    <hack_baseline>0.07</hack_baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
  </plugin>
</gazebo>
```

### IMU Plugin Example

```xml
<gazebo>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>imu</namespace>
      <remapping>~/out:=data</remapping>
    </ros>
    <frame_name>imu_link</frame_name>
    <body_name>imu_body</body_name>
    <update_rate>100</update_rate>
    <gaussian_noise>0.0017</gaussian_noise>
    <topic>/imu/data</topic>
  </plugin>
</gazebo>
```

### Lidar Plugin Example

```xml
<gazebo>
  <plugin name="laser_plugin" filename="libgazebo_ros_laser.so">
    <ros>
      <namespace>laser</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <frame_name>laser_link</frame_name>
    <topic_name>scan</topic_name>
    <gaussian_noise>0.005</gaussian_noise>
    <update_rate>10</update_rate>
  </plugin>
</gazebo>
```

## Advanced Physics Concepts

### Contact Stabilization

For stable contact between objects:

```xml
<physics name="stabilized_physics" type="ode">
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>  <!-- More iterations for stability -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>1e-5</cfm>  <!-- Very small CFM for stiff constraints -->
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Soft Contacts

For more realistic contact with deformable objects:

```xml
<surface>
  <contact>
    <ode>
      <soft_cfm>0.001</soft_cfm>  <!-- Soft constraint force mixing -->
      <soft_erp>0.8</soft_erp>    <!-- Soft error reduction -->
      <kp>1000000000000.0</kp>   <!-- High stiffness -->
      <kd>1.0</kd>               <!-- Damping -->
      <max_vel>100.0</max_vel>
      <min_depth>0.0001</min_depth>  <!-- Very small penetration allowed -->
    </ode>
  </contact>
</surface>
```

## Simulation Accuracy Considerations

### Realism vs. Performance

Balance simulation accuracy with performance:

- **Higher update rates**: More accurate but slower
- **Smaller time steps**: More accurate but slower
- **More solver iterations**: More stable but slower
- **Complex collision geometry**: More accurate but slower

### Tuning Parameters

For different simulation requirements:

#### Real-time Simulation
```xml
<max_step_size>0.01</max_step_size>
<real_time_update_rate>100</real_time_update_rate>
<ode>
  <solver>
    <iters>20</iters>  <!-- Fewer iterations for speed -->
  </solver>
</ode>
```

#### Accurate Simulation
```xml
<max_step_size>0.001</max_step_size>
<real_time_update_rate>1000</real_time_update_rate>
<ode>
  <solver>
    <iters>100</iters>  <!-- More iterations for accuracy -->
  </solver>
</ode>
```

## Troubleshooting Common Issues

### Instability

- **Increase solver iterations** to improve stability
- **Reduce time step** for more accurate integration
- **Adjust ERP and CFM** values for constraint stability
- **Check mass properties** for realistic values

### Penetration

- **Increase contact stiffness (kp)**
- **Reduce min_depth** to allow more penetration before force kicks in
- **Check collision geometry** for proper sizing

### Drift

- **Increase ERP** to correct constraint errors faster
- **Check joint limits** and constraints
- **Verify mass distribution**

## Performance Optimization

### Efficient Collision Geometry

- Use primitive shapes when possible
- Simplify mesh geometry for collision
- Use bounding boxes for distant objects
- Implement Level of Detail (LOD) for complex objects

### Sensor Optimization

- Reduce update rates for sensors that don't need high frequency
- Lower resolution for sensors when possible
- Use narrow FOV when full view isn't needed
- Disable visualization when not debugging

## Interactive Elements

### Physics and Sensors Assessment

import Assessment from '@site/src/components/Assessment/Assessment';

<Assessment
  question="What is the primary purpose of the ERP (Error Reduction Parameter) in Gazebo physics configuration?"
  options={[
    {id: 'a', text: 'To control the simulation time step'},
    {id: 'b', text: 'To determine how quickly constraint errors are corrected'},
    {id: 'c', text: 'To set the maximum velocity of objects'},
    {id: 'd', text: 'To define the coefficient of friction'}
  ]}
  correctAnswerId="b"
  explanation="The ERP (Error Reduction Parameter) controls how quickly constraint errors are corrected in the simulation. Higher values mean errors are corrected more quickly, but can lead to instability."
/>

## Summary

Physics simulation and sensor integration are critical for realistic robot simulation. Understanding how to configure physics parameters, implement various sensors, and integrate them with ROS 2 allows for effective testing and development of robotic systems. Proper tuning balances simulation accuracy with performance requirements.

In the next section, we'll explore Unity visualization for robotics and alternative simulation approaches.