---
title: ROS 2 Architecture and Core Concepts
sidebar_position: 1
description: Understanding the architecture of ROS 2 and its core concepts for robotics development
duration: 120
difficulty: intermediate
learning_objectives:
  - Explain the architecture of ROS 2 and its key components
  - Understand the DDS-based communication layer
  - Identify the differences between ROS 1 and ROS 2
  - Describe the role of RMW in ROS 2
---

# ROS 2 Architecture and Core Concepts

## Learning Objectives

By the end of this section, you will be able to:
- Explain the architecture of ROS 2 and its key components
- Understand the DDS-based communication layer
- Identify the differences between ROS 1 and ROS 2
- Describe the role of RMW in ROS 2

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Evolution from ROS 1

ROS 2 was developed to address several limitations of ROS 1:

- **Real-time support**: Better real-time capabilities for time-critical applications
- **Multi-robot systems**: Improved support for multiple robots working together
- **Production environments**: Designed for industrial and commercial applications
- **Quality of Service (QoS)**: Configurable communication policies
- **Security**: Built-in security features for safe deployment
- **Architecture**: More robust and scalable architecture

## Core Architecture

### DDS (Data Distribution Service)

ROS 2 uses DDS as its underlying communication middleware. DDS provides:

- **Data-centricity**: Communication is based on data rather than connections
- **Quality of Service (QoS)**: Configurable policies for reliability, durability, etc.
- **Discovery**: Automatic discovery of participants in the system
- **Real-time performance**: Optimized for time-critical applications

### RMW (ROS Middleware)

The ROS Middleware (RMW) layer abstracts the underlying DDS implementation:

- **Middleware agnostic**: Can work with different DDS implementations
- **Abstraction layer**: Provides ROS 2 API regardless of DDS vendor
- **Flexibility**: Allows switching between different DDS implementations

## Key Components

### Nodes

Nodes are the fundamental units of computation in ROS 2:

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

### Topics and Messages

Topics enable asynchronous communication between nodes:

- **Messages**: Data structures that can be published and subscribed
- **Publishers**: Send messages to topics
- **Subscribers**: Receive messages from topics
- **Topic names**: Used to connect publishers and subscribers

### Services and Actions

- **Services**: Synchronous request-response communication
- **Actions**: Long-running tasks with feedback and goal management

## Quality of Service (QoS)

QoS profiles allow fine-tuning communication behavior:

### Reliability Policy
- **Reliable**: All messages are delivered (like TCP)
- **Best effort**: Messages may be lost (like UDP)

### Durability Policy
- **Transient local**: Late-joining subscribers receive last message
- **Volatile**: Only new messages are sent

### History Policy
- **Keep all**: Store all messages
- **Keep last**: Store only the most recent messages

## Execution Model

### Single-threaded Executor
Executes all callbacks in a single thread:

```python
executor = SingleThreadedExecutor()
executor.add_node(node)
executor.spin()
```

### Multi-threaded Executor
Distributes callbacks across multiple threads:

```python
executor = MultiThreadedExecutor()
executor.add_node(node)
executor.spin()
```

## Lifecycle Management

ROS 2 provides lifecycle management for more complex node management:

- **Unconfigured**: Node created but not configured
- **Inactive**: Configured but not active
- **Active**: Fully operational
- **Finalized**: Ready for cleanup

## Package System

ROS 2 uses the colcon build system:

- **Packages**: Organize related functionality
- **Dependencies**: Managed through package.xml
- **Build system**: CMake for C++, ament_python for Python

### Package Structure
```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── src/
│   └── my_node.cpp
├── include/
│   └── my_robot_package/
│       └── my_header.h
├── launch/
│   └── my_launch_file.py
├── config/
│   └── parameters.yaml
└── test/
    └── test_my_node.cpp
```

## Communication Patterns

### Publisher-Subscriber Pattern
Asynchronous communication for streaming data:

```python
# Publisher
publisher = node.create_publisher(String, 'chatter', 10)

# Subscriber
subscriber = node.create_subscription(
    String,
    'chatter',
    callback_function,
    10
)
```

### Client-Server Pattern
Synchronous request-response communication:

```python
# Service Server
service = node.create_service(AddTwoInts, 'add_two_ints', callback)

# Client
client = node.create_client(AddTwoInts, 'add_two_ints')
```

### Action Pattern
Long-running tasks with feedback:

```python
# Action Server
action_server = ActionServer(
    node,
    Fibonacci,
    'fibonacci',
    execute_callback
)

# Action Client
action_client = ActionClient(node, Fibonacci, 'fibonacci')
```

## Launch System

ROS 2 provides a Python-based launch system:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='my_node'
        )
    ])
```

## Parameters

Dynamic configuration system:

```python
# Declare parameter
self.declare_parameter('my_parameter', 'default_value')

# Get parameter
value = self.get_parameter('my_parameter').value

# Set parameter
self.set_parameters([Parameter('my_parameter', Parameter.Type.STRING, 'new_value')])
```

## Interactive Elements

### ROS 2 Architecture Quiz

import Assessment from '@site/src/components/Assessment/Assessment';

<Assessment
  question="What does RMW stand for in ROS 2 architecture?"
  options={[
    {id: 'a', text: 'Robot Middleware Wrapper'},
    {id: 'b', text: 'ROS Middleware'},
    {id: 'c', text: 'Real-time Message Worker'},
    {id: 'd', text: 'Remote Management Web'}
  ]}
  correctAnswerId="b"
  explanation="RMW stands for ROS Middleware, which is the abstraction layer that allows ROS 2 to work with different DDS implementations."
/>

## Summary

ROS 2 represents a significant evolution from ROS 1 with a more robust architecture based on DDS. The system provides better support for real-time applications, multi-robot systems, and production environments. Understanding the core concepts of nodes, topics, services, and actions along with QoS policies is crucial for effective ROS 2 development.

In the next section, we'll explore the communication patterns in more detail, focusing on topics, services, and actions.