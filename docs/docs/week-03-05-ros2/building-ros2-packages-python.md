---
title: Building ROS 2 Packages with Python
sidebar_position: 3
description: Creating and building ROS 2 packages using Python
duration: 180
difficulty: intermediate
learning_objectives:
  - Create ROS 2 packages using Python
  - Understand the package structure and dependencies
  - Build and run Python-based ROS 2 nodes
  - Use ament_python build system effectively
---

# Building ROS 2 Packages with Python

## Learning Objectives

By the end of this section, you will be able to:
- Create ROS 2 packages using Python
- Understand the package structure and dependencies
- Build and run Python-based ROS 2 nodes
- Use ament_python build system effectively

## Package Creation

### Using ros2 pkg create

The easiest way to create a new ROS 2 package is using the `ros2 pkg create` command:

```bash
ros2 pkg create --build-type ament_python my_robot_package
```

This creates a basic package structure with the necessary files for a Python-based ROS 2 package.

### Package Structure

A typical ROS 2 Python package has the following structure:

```
my_robot_package/
├── my_robot_package/          # Python package directory
│   ├── __init__.py           # Python package initialization
│   ├── my_node.py            # Python node implementation
│   └── my_module.py          # Additional Python modules
├── setup.py                  # Python setup script
├── setup.cfg                 # Configuration for installation
├── package.xml               # Package metadata
├── resource/                 # Resource files
└── test/                     # Test files
    └── test_copyright.py
    └── test_flake8.py
    └── test_pep257.py
```

## Package Configuration Files

### package.xml

The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 package using Python</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### setup.py

The `setup.py` file configures how the package is built and installed:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User Name',
    maintainer_email='user@example.com',
    description='Example ROS 2 package using Python',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_robot_package.my_node:main',
            'another_node = my_robot_package.another_node:main',
        ],
    },
)
```

### setup.cfg

The `setup.cfg` file specifies installation options:

```ini
[develop]
script-dir=$base/lib/my_robot_package
[install]
install-scripts=$base/lib/my_robot_package
```

## Creating Python Nodes

### Basic Node Structure

A Python ROS 2 node follows this basic structure:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        self.get_logger().info('MyNode has been started')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node with Publishers and Subscribers

Here's a more complete example with communication:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import math

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Create publisher
        self.publisher = self.create_publisher(String, 'robot_commands', 10)

        # Create subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            'laser_scan',
            self.laser_callback,
            10
        )

        # Create timer
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('Robot Controller Node has been started')

    def laser_callback(self, msg):
        # Process laser scan data
        min_distance = min(msg.ranges)
        self.get_logger().info(f'Min distance: {min_distance:.2f}')

    def timer_callback(self):
        # Send a command
        msg = String()
        msg.data = 'move_forward'
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Dependencies Management

### ROS Dependencies

In `package.xml`, specify ROS dependencies:

```xml
<depend>rclpy</depend>
<depend>std_msgs</depend>
<depend>geometry_msgs</depend>
<depend>nav_msgs</depend>
<depend>sensor_msgs</depend>
```

### Python Dependencies

For non-ROS Python packages, add them to `setup.py`:

```python
setup(
    # ... other parameters
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'scipy',
        # Add other Python dependencies here
    ],
    # ... rest of setup
)
```

## Building and Running

### Building the Package

To build a Python-based ROS 2 package:

```bash
colcon build --packages-select my_robot_package
```

Or to build all packages in the workspace:

```bash
colcon build
```

### Sourcing the Environment

After building, source the environment:

```bash
source install/setup.bash
```

### Running the Node

Run a specific node from the package:

```bash
ros2 run my_robot_package my_node
```

## Launch Files

Launch files allow you to start multiple nodes at once. Create a `launch` directory in your package:

### Python Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='my_node',
            name='robot_controller',
            parameters=[
                {'param1': 'value1'},
                {'param2': 42},
            ],
            remappings=[
                ('original_topic', 'remapped_topic'),
            ],
            output='screen'
        ),
        Node(
            package='my_robot_package',
            executable='another_node',
            name='sensor_processor',
            output='screen'
        )
    ])
```

### Running Launch Files

Run the launch file:

```bash
ros2 launch my_robot_package my_launch_file.py
```

## Parameters

### Parameter Declaration and Usage

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('use_sim_time', False)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.use_sim_time = self.get_parameter('use_sim_time').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### YAML Parameter Files

Create parameter files in a `config` directory:

```yaml
# config/robot_params.yaml
/**:
  ros__parameters:
    robot_name: 'turtlebot4'
    max_velocity: 0.5
    use_sim_time: false
    sensors:
      laser_scan_topic: '/scan'
      camera_topic: '/camera/image_raw'
    navigation:
      planner_frequency: 5.0
      controller_frequency: 20.0
```

Use the parameter file:

```bash
ros2 run my_robot_package my_node --ros-args --params-file config/robot_params.yaml
```

## Testing

### Unit Tests

Create tests in the `test` directory:

```python
# test/test_my_node.py
import unittest
import rclpy
from my_robot_package.my_node import MyNode

class TestMyNode(unittest.TestCase):
    def setUp(self):
        rclpy.init()

    def tearDown(self):
        rclpy.shutdown()

    def test_node_creation(self):
        node = MyNode()
        self.assertEqual(node.get_name(), 'my_node_name')
        node.destroy_node()

if __name__ == '__main__':
    unittest.main()
```

### Running Tests

Run tests using colcon:

```bash
colcon test --packages-select my_robot_package
colcon test-result --all
```

## Best Practices

### Code Structure

1. **Separate concerns**: Keep node logic separate from business logic
2. **Use modules**: Break large nodes into smaller modules
3. **Follow naming conventions**: Use snake_case for Python functions and variables
4. **Document code**: Use docstrings for classes and functions

### Error Handling

```python
def safe_divide(self, a, b):
    if b == 0:
        self.get_logger().error('Division by zero attempted')
        return None
    return a / b
```

### Logging

Use appropriate log levels:

```python
self.get_logger().debug('Debug information')
self.get_logger().info('General information')
self.get_logger().warn('Warning message')
self.get_logger().error('Error message')
self.get_logger().fatal('Fatal error message')
```

## Interactive Elements

### ROS 2 Python Package Quiz

import Assessment from '@site/src/components/Assessment/Assessment';

<Assessment
  question="What is the correct build type to use in package.xml for a Python-based ROS 2 package?"
  options={[
    {id: 'a', text: 'ament_cmake'},
    {id: 'b', text: 'ament_python'},
    {id: 'c', text: 'cmake'},
    {id: 'd', text: 'colcon_python'}
  ]}
  correctAnswerId="b"
  explanation="For Python-based ROS 2 packages, you must use 'ament_python' as the build type in the package.xml file."
/>

## Summary

Building ROS 2 packages with Python involves understanding the package structure, configuration files, and the ament_python build system. Proper package organization, dependency management, and following best practices are essential for creating maintainable and reusable ROS 2 packages. The combination of Python's simplicity with ROS 2's powerful communication patterns makes it an excellent choice for robotics development.

In the next section, we'll explore launch files and parameter management in more detail.