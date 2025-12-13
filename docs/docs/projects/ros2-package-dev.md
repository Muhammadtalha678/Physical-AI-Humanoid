---
title: ROS 2 Package Development Project
sidebar_position: 1
description: Developing a complete ROS 2 package with nodes, topics, and services
duration: 240
difficulty: intermediate
learning_objectives:
  - Create a complete ROS 2 package from scratch
  - Implement nodes with publishers and subscribers
  - Design and implement services and actions
  - Test and validate the package functionality
---

# ROS 2 Package Development Project

## Learning Objectives

By the end of this project, you will be able to:
- Create a complete ROS 2 package from scratch
- Implement nodes with publishers and subscribers
- Design and implement services and actions
- Test and validate the package functionality

## Project Overview

This project requires you to develop a complete ROS 2 package that demonstrates core concepts learned in the ROS 2 fundamentals weeks. You'll create a robot controller package that integrates multiple nodes with different communication patterns.

### Project Requirements

1. **Package Structure**: Create a well-organized ROS 2 package
2. **Nodes**: Implement at least 2 nodes with different responsibilities
3. **Communication**: Use topics, services, and actions appropriately
4. **Configuration**: Include launch files and parameter files
5. **Testing**: Implement basic tests for your nodes

## Package Design

### Project Structure

Your package should follow the standard ROS 2 structure:

```
ros2_robot_controller/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup.cfg
├── ros2_robot_controller/
│   ├── __init__.py
│   ├── robot_controller.py
│   ├── sensor_processor.py
│   └── utils/
│       ├── __init__.py
│       └── conversions.py
├── launch/
│   ├── robot_controller_launch.py
│   └── sensor_processor_launch.py
├── config/
│   ├── robot_params.yaml
│   └── sensor_params.yaml
├── test/
│   ├── test_robot_controller.py
│   └── test_sensor_processor.py
└── README.md
```

### Package.xml Configuration

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>ros2_robot_controller</name>
  <version>0.0.1</version>
  <description>A complete ROS 2 robot controller package</description>
  <maintainer email="student@university.edu">Student Name</maintainer>
  <license>Apache License 2.0</license>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>builtin_interfaces</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Implementation Requirements

### 1. Robot Controller Node

Create a `robot_controller.py` file with a node that:

- Subscribes to sensor data
- Publishes movement commands
- Provides services for controlling the robot
- Implements an action for complex navigation tasks

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ros2_robot_controller.action import NavigateToPosition
from ros2_robot_controller.srv import SetSpeed, GetRobotStatus


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Parameters
        self.declare_parameter('robot_name', 'turtlebot4')
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        # Services
        self.set_speed_srv = self.create_service(
            SetSpeed, 'set_speed', self.set_speed_callback)
        self.get_status_srv = self.create_service(
            GetRobotStatus, 'get_status', self.get_status_callback)

        # Actions
        self.navigate_action_server = ActionServer(
            self,
            NavigateToPosition,
            'navigate_to_position',
            self.navigate_execute_callback,
            callback_group=ReentrantCallbackGroup())

        # Internal state
        self.current_speed = 0.0
        self.current_pose = None
        self.obstacle_detected = False

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Robot Controller initialized')

    def scan_callback(self, msg):
        # Process laser scan data
        min_range = min(msg.ranges)
        self.obstacle_detected = min_range < 0.5
        if self.obstacle_detected:
            self.get_logger().warn(f'Obstacle detected at {min_range:.2f}m')

    def odom_callback(self, msg):
        # Update robot pose
        self.current_pose = msg.pose.pose

    def set_speed_callback(self, request, response):
        # Validate and set new speed
        max_speed = self.get_parameter('max_linear_speed').value
        if abs(request.speed) <= max_speed:
            self.current_speed = request.speed
            response.success = True
            response.message = f'Speed set to {self.current_speed}'
        else:
            response.success = False
            response.message = f'Speed {request.speed} exceeds maximum {max_speed}'
        return response

    def get_status_callback(self, request, response):
        # Return current robot status
        response.speed = self.current_speed
        response.obstacle_detected = self.obstacle_detected
        response.pose = self.current_pose if self.current_pose else Odometry().pose.pose
        response.status = "ACTIVE" if not self.obstacle_detected else "OBSTACLE_DETECTED"
        return response

    def navigate_execute_callback(self, goal_handle):
        # Execute navigation action
        self.get_logger().info(f'Navigating to position: {goal_handle.request.target_pose}')

        # Simulate navigation
        feedback_msg = NavigateToPosition.Feedback()
        result = NavigateToPosition.Result()

        # Navigation logic would go here
        for i in range(10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                return result

            feedback_msg.distance_remaining = 10.0 - i
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Distance remaining: {feedback_msg.distance_remaining}')

            # Sleep for a bit
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.5))

        goal_handle.succeed()
        result.success = True
        result.final_pose = self.current_pose if self.current_pose else Odometry().pose.pose
        return result

    def control_loop(self):
        # Main control loop
        cmd_msg = Twist()

        if not self.obstacle_detected:
            cmd_msg.linear.x = self.current_speed
            cmd_msg.angular.z = 0.0  # Simple control - no turning
        else:
            # Stop if obstacle detected
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)

    robot_controller = RobotController()

    executor = MultiThreadedExecutor()
    executor.add_node(robot_controller)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. Sensor Processor Node

Create a `sensor_processor.py` file with a node that:

- Processes multiple sensor inputs
- Fuses sensor data
- Publishes processed information

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, BatteryState
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from ros2_robot_controller.msg import SensorFusionData


class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Subscribers for different sensors
        self.laser_sub = self.create_subscription(
            LaserScan, 'raw_scan', self.laser_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.battery_sub = self.create_subscription(
            BatteryState, 'battery/state', self.battery_callback, 10)

        # Publishers
        self.fused_data_pub = self.create_publisher(
            SensorFusionData, 'sensor_fusion_data', 10)
        self.obstacle_warning_pub = self.create_publisher(
            Float32, 'obstacle_distance', 10)

        # Internal state
        self.latest_laser = None
        self.latest_imu = None
        self.latest_battery = None

        # Timer for fusion processing
        self.fusion_timer = self.create_timer(0.05, self.process_sensor_fusion)

        self.get_logger().info('Sensor Processor initialized')

    def laser_callback(self, msg):
        self.latest_laser = msg

    def imu_callback(self, msg):
        self.latest_imu = msg

    def battery_callback(self, msg):
        self.latest_battery = msg

    def process_sensor_fusion(self):
        if self.latest_laser is not None:
            # Process laser data
            min_range = min(self.latest_laser.ranges)

            # Publish obstacle distance
            obstacle_msg = Float32()
            obstacle_msg.data = min_range
            self.obstacle_warning_pub.publish(obstacle_msg)

            # Create fused data message
            fused_msg = SensorFusionData()
            fused_msg.header.stamp = self.get_clock().now().to_msg()
            fused_msg.header.frame_id = "base_link"

            fused_msg.closest_obstacle_distance = min_range
            fused_msg.battery_level = self.latest_battery.percentage if self.latest_battery else 100.0
            fused_msg.linear_acceleration = self.latest_imu.linear_acceleration if self.latest_imu else Vector3()
            fused_msg.angular_velocity = self.latest_imu.angular_velocity if self.latest_imu else Vector3()

            self.fused_data_pub.publish(fused_msg)


def main(args=None):
    rclpy.init(args=args)

    sensor_processor = SensorProcessor()

    try:
        rclpy.spin(sensor_processor)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 3. Service and Action Definitions

Create the necessary interface files:

**srv/SetSpeed.srv**
```
float32 speed
---
bool success
string message
```

**srv/GetRobotStatus.srv**
```
---
float32 speed
bool obstacle_detected
geometry_msgs/Pose pose
string status
```

**action/NavigateToPosition.action**
```
geometry_msgs/Pose target_pose
float32 tolerance
---
bool success
geometry_msgs/Pose final_pose
---
float32 distance_remaining
```

**msg/SensorFusionData.msg**
```
std_msgs/Header header
float32 closest_obstacle_distance
float32 battery_level
geometry_msgs/Vector3 linear_acceleration
geometry_msgs/Vector3 angular_velocity
```

### 4. Launch File

Create `launch/robot_controller_launch.py`:

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

    return LaunchDescription([
        use_sim_time,

        # Robot controller node
        Node(
            package='ros2_robot_controller',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'robot_name': 'turtlebot4'},
                {'max_linear_speed': 0.5},
                {'max_angular_speed': 1.0},
            ],
            remappings=[
                ('cmd_vel', '/cmd_vel'),
                ('scan', '/scan'),
                ('odom', '/odom'),
            ],
            output='screen'
        ),

        # Sensor processor node
        Node(
            package='ros2_robot_controller',
            executable='sensor_processor',
            name='sensor_processor',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
            remappings=[
                ('raw_scan', '/scan'),
                ('imu/data', '/imu/data'),
                ('battery/state', '/battery/state'),
            ],
            output='screen'
        )
    ])
```

### 5. Parameter File

Create `config/robot_params.yaml`:

```yaml
# Robot Controller Parameters
robot_controller:
  ros__parameters:
    robot_name: 'turtlebot4'
    max_linear_speed: 0.5
    max_angular_speed: 1.0
    linear_acceleration: 1.0
    angular_acceleration: 2.0

    # Sensor processing parameters
    laser_processing_rate: 10.0
    fusion_rate: 20.0

    # Safety parameters
    obstacle_detection_threshold: 0.5
    emergency_stop_distance: 0.2
    safety_margin: 0.3

# Sensor Processor Parameters
sensor_processor:
  ros__parameters:
    use_sim_time: false

    # Processing parameters
    fusion_frequency: 20.0

    # Thresholds
    battery_low_threshold: 20.0
    battery_critical_threshold: 10.0
```

## Testing and Validation

### Unit Tests

Create basic tests for your nodes:

```python
# test/test_robot_controller.py
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from ros2_robot_controller.robot_controller import RobotController


class TestRobotController(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = RobotController()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_initial_state(self):
        # Test initial state of the controller
        self.assertEqual(self.node.current_speed, 0.0)
        self.assertFalse(self.node.obstacle_detected)

    def test_set_speed_within_limits(self):
        # Test setting speed within limits
        from ros2_robot_controller.srv import SetSpeed

        # Call the service directly
        request = SetSpeed.Request()
        request.speed = 0.3
        response = self.node.set_speed_callback(request, SetSpeed.Response())

        self.assertTrue(response.success)
        self.assertEqual(self.node.current_speed, 0.3)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
```

## Project Submission Requirements

### Deliverables

1. **Complete ROS 2 Package**: A fully functional ROS 2 package with all required components
2. **Documentation**: README.md explaining how to build and run your package
3. **Configuration Files**: All launch and parameter files
4. **Test Results**: Output from running your tests
5. **Demonstration Script**: A script showing the package functionality

### Evaluation Criteria

- **Functionality** (40%): Does the package work as expected?
- **Code Quality** (25%): Is the code well-structured, documented, and follows best practices?
- **ROS 2 Integration** (20%): Are ROS 2 concepts properly implemented (topics, services, actions)?
- **Testing** (15%): Are there adequate tests and do they pass?

## Assessment Rubric

| Criteria | Excellent (A) | Good (B) | Satisfactory (C) | Needs Improvement (D) |
|----------|---------------|----------|------------------|----------------------|
| Code Quality | Well-structured, documented, follows best practices | Good structure and documentation | Basic structure, minimal documentation | Poor structure, inadequate documentation |
| ROS 2 Concepts | All concepts implemented correctly with advanced features | Most concepts implemented correctly | Basic concepts implemented | Incorrect or missing concepts |
| Functionality | All features work flawlessly | Most features work correctly | Basic functionality works | Significant functionality issues |
| Testing | Comprehensive tests with high coverage | Good test coverage | Basic tests included | Inadequate or no testing |

## Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Python Client Library (rclpy)](https://docs.ros.org/en/humble/p/rclpy/)
- [ROS 2 Launch System](https://docs.ros.org/en/humble/p/launch/)
- [ROS 2 Parameter System](https://docs.ros.org/en/humble/p/rclpy/topics/parameters.html)

## Next Steps

After completing this project, you should be able to:
- Create complex ROS 2 packages with multiple nodes
- Implement various communication patterns effectively
- Configure and launch systems with launch files
- Test and validate your ROS 2 packages

This project serves as a foundation for more advanced robotics projects in subsequent weeks.