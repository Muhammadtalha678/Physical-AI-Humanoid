---
title: Speech, Planning, and Navigation - Capstone Requirements
sidebar_position: 1
description: Requirements for speech recognition, path planning, and navigation systems in the capstone project
duration: 240
difficulty: advanced
learning_objectives:
  - Implement multimodal speech interaction for humanoid robots
  - Design path planning algorithms for autonomous navigation
  - Integrate perception and navigation systems
  - Create robust navigation behaviors for dynamic environments
---

# Speech, Planning, and Navigation - Capstone Requirements

## Learning Objectives

By the end of this section, you will be able to:
- Implement speech recognition and natural language understanding for humanoid robots
- Design and implement path planning algorithms for autonomous navigation
- Integrate perception data with navigation systems
- Create robust navigation behaviors that handle dynamic environments
- Evaluate and optimize navigation performance in real-world scenarios

## Project Overview

This capstone component focuses on three critical aspects of autonomous humanoid robotics: speech interaction, path planning, and navigation. You'll implement systems that allow a humanoid robot to understand spoken commands, plan safe paths through environments, and navigate autonomously while avoiding obstacles.

## Prerequisites

- Completion of ROS 2 fundamentals (Weeks 3-5)
- Understanding of perception systems (Weeks 8-10)
- Experience with Isaac Platform (Weeks 8-10)
- Basic knowledge of artificial intelligence and planning algorithms

## Speech Recognition and Understanding System

### Step 1: Speech Recognition Setup

First, set up the speech recognition system using ROS 2 and appropriate libraries:

```python
# speech_recognition_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import speech_recognition as sr
import threading
import queue

class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('speech_recognition_node')

        # Publisher for recognized text
        self.text_pub = self.create_publisher(String, 'speech_recognized', 10)

        # Audio data subscriber
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000  # Adjust based on environment
        self.audio_queue = queue.Queue()

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        self.get_logger().info('Speech recognition node initialized')

    def audio_callback(self, msg):
        """Callback for audio data"""
        # Convert ROS AudioData to audio data for speech recognition
        audio_data = sr.AudioData(
            msg.data,
            sample_rate=16000,  # Adjust based on your audio source
            sample_width=2      # 16-bit audio
        )

        # Add to processing queue
        self.audio_queue.put(audio_data)

    def process_audio(self):
        """Process audio data in a separate thread"""
        while rclpy.ok():
            try:
                # Get audio from queue
                audio_data = self.audio_queue.get(timeout=1.0)

                # Perform speech recognition
                try:
                    # Using Google Web Speech API (requires internet)
                    text = self.recognizer.recognize_google(audio_data)

                    # Publish recognized text
                    text_msg = String()
                    text_msg.data = text
                    self.text_pub.publish(text_msg)

                    self.get_logger().info(f'Recognized: {text}')

                except sr.UnknownValueError:
                    self.get_logger().info('Speech recognition could not understand audio')
                except sr.RequestError as e:
                    self.get_logger().error(f'Could not request results from speech recognition service; {e}')

            except queue.Empty:
                continue

def main(args=None):
    rclpy.init(args=args)
    speech_node = SpeechRecognitionNode()

    try:
        rclpy.spin(speech_node)
    except KeyboardInterrupt:
        pass
    finally:
        speech_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 2: Natural Language Understanding

Implement natural language understanding to interpret commands:

```python
# nlu_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import re
import json

class NaturalLanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('nlu_node')

        # Subscribe to recognized speech
        self.speech_sub = self.create_subscription(
            String,
            'speech_recognized',
            self.speech_callback,
            10
        )

        # Publishers for navigation commands
        self.nav_goal_pub = self.create_publisher(Pose, 'navigation_goal', 10)
        self.cmd_pub = self.create_publisher(String, 'robot_command', 10)

        # Define command patterns
        self.command_patterns = {
            'move_to': [
                r'.*go to (.+)',
                r'.*move to (.+)',
                r'.*navigate to (.+)',
                r'.*go (.+)'
            ],
            'follow': [
                r'.*follow me',
                r'.*follow (.+)',
                r'.*come with me'
            ],
            'stop': [
                r'.*stop',
                r'.*halt',
                r'.*wait'
            ],
            'dance': [
                r'.*dance',
                r'.*perform dance',
                r'.*show dance'
            ]
        }

        self.get_logger().info('Natural Language Understanding node initialized')

    def speech_callback(self, msg):
        """Process recognized speech"""
        text = msg.data.lower().strip()

        # Interpret the command
        command = self.interpret_command(text)

        if command:
            self.get_logger().info(f'Interpreted command: {command}')

            # Publish command
            cmd_msg = String()
            cmd_msg.data = json.dumps(command)
            self.cmd_pub.publish(cmd_msg)

            # Handle specific commands
            if command['type'] == 'move_to':
                self.handle_move_to_command(command['location'])
            elif command['type'] == 'follow':
                self.handle_follow_command(command.get('target', 'user'))
            elif command['type'] == 'stop':
                self.handle_stop_command()
            elif command['type'] == 'dance':
                self.handle_dance_command()

    def interpret_command(self, text):
        """Interpret natural language command"""
        for cmd_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, text)
                if match:
                    if cmd_type == 'move_to':
                        location = match.group(1).strip()
                        return {'type': cmd_type, 'location': location}
                    elif cmd_type == 'follow':
                        if match.groups():
                            target = match.group(1).strip()
                        else:
                            target = 'user'
                        return {'type': cmd_type, 'target': target}
                    else:
                        return {'type': cmd_type}

        # If no pattern matches, return None
        return None

    def handle_move_to_command(self, location):
        """Handle move to location command"""
        # In a real implementation, you would:
        # 1. Look up the location in a map
        # 2. Convert to coordinates
        # 3. Send navigation goal
        self.get_logger().info(f'Navigating to: {location}')

    def handle_follow_command(self, target):
        """Handle follow command"""
        self.get_logger().info(f'Following: {target}')
        # Implementation would use tracking algorithms

    def handle_stop_command(self):
        """Handle stop command"""
        self.get_logger().info('Stopping robot')

    def handle_dance_command(self):
        """Handle dance command"""
        self.get_logger().info('Performing dance routine')

def main(args=None):
    rclpy.init(args=args)
    nlu_node = NaturalLanguageUnderstandingNode()

    try:
        rclpy.spin(nlu_node)
    except KeyboardInterrupt:
        pass
    finally:
        nlu_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Path Planning and Navigation System

### Step 3: Global Path Planner

Implement a global path planner for navigation:

```python
# global_planner_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker
import numpy as np
from scipy.spatial import KDTree
import heapq

class GlobalPlannerNode(Node):
    def __init__(self):
        super().__init__('global_planner_node')

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            'move_base_simple/goal',
            self.goal_callback,
            10
        )

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.marker_pub = self.create_publisher(Marker, 'path_visualization', 10)

        # Store map data
        self.map_data = None
        self.map_resolution = 0.0
        self.map_origin = None
        self.map_width = 0
        self.map_height = 0

        # Path planning parameters
        self.planning_frequency = 1.0  # Hz
        self.timer = self.create_timer(1.0/self.planning_frequency, self.plan_path)

        self.start_pose = None
        self.goal_pose = None

        self.get_logger().info('Global planner node initialized')

    def map_callback(self, msg):
        """Callback for map data"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        self.map_width = msg.info.width
        self.map_height = msg.info.height

        self.get_logger().info('Map received and stored')

    def goal_callback(self, msg):
        """Callback for goal pose"""
        self.goal_pose = msg.pose
        self.get_logger().info(f'New goal received: ({msg.pose.position.x}, {msg.pose.position.y})')

    def plan_path(self):
        """Plan path using A* algorithm"""
        if self.map_data is None or self.goal_pose is None:
            return

        # Get current robot position (in a real system, this would come from localization)
        # For this example, we'll use a fixed start position
        start_x = 0.0
        start_y = 0.0

        # Convert world coordinates to map coordinates
        start_map_x = int((start_x - self.map_origin.position.x) / self.map_resolution)
        start_map_y = int((start_y - self.map_origin.position.y) / self.map_resolution)

        goal_map_x = int((self.goal_pose.position.x - self.map_origin.position.x) / self.map_resolution)
        goal_map_y = int((self.goal_pose.position.y - self.map_origin.position.y) / self.map_resolution)

        # Perform A* path planning
        path = self.a_star_plan(start_map_x, start_map_y, goal_map_x, goal_map_y)

        if path:
            # Convert path back to world coordinates
            world_path = self.map_path_to_world(path)

            # Publish path
            self.publish_path(world_path)

            # Visualize path
            self.visualize_path(world_path)

    def a_star_plan(self, start_x, start_y, goal_x, goal_y):
        """A* path planning algorithm"""
        # Check if start or goal is in obstacle space
        if (self.map_data[start_y, start_x] > 50 or  # 50 is threshold for obstacle
            self.map_data[goal_y, goal_x] > 50):
            self.get_logger().warn('Start or goal position is in obstacle space')
            return None

        # Define possible movements (8-connected)
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Convert movements to costs (diagonal movement costs more)
        move_costs = [
            1.414, 1.0, 1.414,
            1.0,          1.0,
            1.414, 1.0, 1.414
        ]

        # Initialize open and closed sets
        open_set = []
        heapq.heappush(open_set, (0, (start_x, start_y)))

        # Store costs and parents
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self.heuristic(start_x, start_y, goal_x, goal_y)}
        came_from = {}

        while open_set:
            current = heapq.heappop(open_set)[1]

            # Check if we reached the goal
            if current == (goal_x, goal_y):
                return self.reconstruct_path(came_from, current)

            # Explore neighbors
            for i, move in enumerate(movements):
                neighbor = (current[0] + move[0], current[1] + move[1])

                # Check if neighbor is within map bounds
                if (0 <= neighbor[0] < self.map_width and
                    0 <= neighbor[1] < self.map_height):

                    # Check if neighbor is traversable
                    if self.map_data[neighbor[1], neighbor[0]] < 50:  # Not an obstacle
                        tentative_g_score = g_score[current] + move_costs[i]

                        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + self.heuristic(
                                neighbor[0], neighbor[1], goal_x, goal_y
                            )
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None

    def heuristic(self, x1, y1, x2, y2):
        """Heuristic function for A* (Euclidean distance)"""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def map_path_to_world(self, map_path):
        """Convert path from map coordinates to world coordinates"""
        world_path = []
        for x, y in map_path:
            world_x = x * self.map_resolution + self.map_origin.position.x
            world_y = y * self.map_resolution + self.map_origin.position.y

            pose = Pose()
            pose.position.x = world_x
            pose.position.y = world_y
            pose.position.z = self.map_origin.position.z  # Assume z is constant

            world_path.append(pose)

        return world_path

    def publish_path(self, path):
        """Publish the planned path"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'  # or appropriate frame

        for pose in path:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose = pose
            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

    def visualize_path(self, path):
        """Visualize the path using markers"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Set the scale of the marker
        marker.scale.x = 0.05  # Line width

        # Set the color (green)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the points
        for pose in path:
            point = Point()
            point.x = pose.position.x
            point.y = pose.position.y
            point.z = pose.position.z
            marker.points.append(point)

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    planner_node = GlobalPlannerNode()

    try:
        rclpy.spin(planner_node)
    except KeyboardInterrupt:
        pass
    finally:
        planner_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Local Navigation and Obstacle Avoidance

Implement local navigation and obstacle avoidance:

```python
# local_navigator_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker
import numpy as np
import math

class LocalNavigatorNode(Node):
    def __init__(self):
        super().__init__('local_navigator_node')

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            'global_plan',
            self.path_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, 'obstacle_visualization', 10)

        # Navigation parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s
        self.safe_distance = 0.5  # meters
        self.arrival_threshold = 0.3  # meters

        # Robot state
        self.current_pose = None
        self.current_yaw = 0.0
        self.path = []
        self.current_path_index = 0

        # Timer for navigation control
        self.nav_timer = self.create_timer(0.1, self.navigate)

        self.get_logger().info('Local navigator node initialized')

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Check for obstacles in front of the robot
        min_distance = float('inf')
        min_angle_idx = 0

        # Check the front 60 degrees (30 degrees on each side)
        front_start = len(msg.ranges) // 2 - 30
        front_end = len(msg.ranges) // 2 + 30

        for i in range(front_start, front_end):
            if 0 < msg.ranges[i] < min_distance:
                min_distance = msg.ranges[i]
                min_angle_idx = i

        # Publish obstacle visualization
        self.visualize_obstacles(msg, min_distance, min_angle_idx)

    def path_callback(self, msg):
        """Receive global path"""
        self.path = msg.poses
        self.current_path_index = 0
        self.get_logger().info(f'New path received with {len(self.path)} waypoints')

    def odom_callback(self, msg):
        """Update robot's current pose"""
        self.current_pose = msg.pose.pose
        # Extract yaw from quaternion
        quat = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2 * (quat.w * quat.z + quat.x * quat.y),
            1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        )

    def navigate(self):
        """Main navigation control loop"""
        if not self.path or self.current_pose is None:
            return

        # Get next waypoint
        if self.current_path_index < len(self.path):
            target_pose = self.path[self.current_path_index].pose
        else:
            # Reached the end of the path
            self.stop_robot()
            return

        # Calculate distance to target
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Check if we've reached the current waypoint
        if distance < self.arrival_threshold:
            self.current_path_index += 1
            if self.current_path_index >= len(self.path):
                self.get_logger().info('Reached final destination')
                self.stop_robot()
                return
            # Get next waypoint
            target_pose = self.path[self.current_path_index].pose
            dx = target_pose.position.x - self.current_pose.position.x
            dy = target_pose.position.y - self.current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)

        # Calculate target angle
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.current_yaw

        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Create velocity command
        cmd = Twist()

        # Check for obstacles before moving
        safe_to_move = self.check_obstacles()

        if not safe_to_move:
            # Implement obstacle avoidance behavior
            cmd = self.avoid_obstacles()
        else:
            # Move toward target
            cmd.linear.x = min(self.linear_speed, distance)  # Slow down as we approach
            cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, angle_diff * 2.0))

        # Publish command
        self.cmd_pub.publish(cmd)

    def check_obstacles(self):
        """Check if it's safe to move forward"""
        # This would typically check laser scan data
        # For this example, we'll return True (safe)
        # In a real implementation, you'd check the actual sensor data
        return True

    def avoid_obstacles(self):
        """Implement obstacle avoidance behavior"""
        cmd = Twist()
        # Simple obstacle avoidance: turn away from obstacles
        cmd.angular.z = self.angular_speed
        return cmd

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def visualize_obstacles(self, scan_msg, min_distance, min_angle_idx):
        """Visualize obstacles using markers"""
        marker = Marker()
        marker.header.frame_id = 'base_link'  # or appropriate frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'obstacles'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Set the scale of the marker
        marker.scale.x = 0.1
        marker.scale.y = 0.1

        # Set the color (red for obstacles)
        marker.color.r = 1.0
        marker.color.a = 1.0

        # Add points for detected obstacles
        for i, range_val in enumerate(scan_msg.ranges):
            if 0 < range_val < 3.0:  # Only show obstacles within 3m
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                point = Point()
                point.x = range_val * math.cos(angle)
                point.y = range_val * math.sin(angle)
                point.z = 0.0
                marker.points.append(point)

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    navigator_node = LocalNavigatorNode()

    try:
        rclpy.spin(navigator_node)
    except KeyboardInterrupt:
        pass
    finally:
        navigator_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration and Testing

### Step 5: Integration Launch File

Create a launch file to integrate all components:

```python
# launch/capstone_integration_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Speech recognition node
    speech_recognition_node = Node(
        package='capstone_speech',
        executable='speech_recognition_node',
        name='speech_recognition',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Natural language understanding node
    nlu_node = Node(
        package='capstone_nlu',
        executable='nlu_node',
        name='natural_language_understanding',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Global planner node
    global_planner_node = Node(
        package='capstone_navigation',
        executable='global_planner_node',
        name='global_planner',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Local navigator node
    local_navigator_node = Node(
        package='capstone_navigation',
        executable='local_navigator_node',
        name='local_navigator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Isaac-specific perception integration (example)
    perception_node = Node(
        package='isaac_ros_detectnet',
        executable='isaac_ros_detectnet',
        name='perception_system',
        parameters=[{
            'input_topic': '/camera/image_raw',
            'model_name': 'ssd_mobilenet_v2_coco',
            'confidence_threshold': 0.5
        }],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        speech_recognition_node,
        nlu_node,
        global_planner_node,
        local_navigator_node,
        perception_node
    ])
```

## Assessment Questions

1. How does the speech recognition system integrate with the navigation system?

2. What are the key differences between global path planning and local navigation?

3. How does the robot handle dynamic obstacles during navigation?

<Assessment
  question="What is the primary function of the global planner in the navigation system?"
  type="multiple-choice"
  options={[
    "To avoid immediate obstacles in real-time",
    "To create a high-level path from start to goal",
    "To control the robot's motors directly",
    "To recognize speech commands"
  ]}
  correctIndex={1}
  explanation="The global planner creates a high-level path from the start location to the goal, considering the overall map and static obstacles."
/>

<Assessment
  question="In the speech recognition system, what happens when the robot cannot understand audio?"
  type="multiple-choice"
  options={[
    "The robot shuts down immediately",
    "The system logs the issue and continues listening",
    "The robot asks for clarification using text output",
    "The system sends an error message to the user"
  ]}
  correctIndex={1}
  explanation="When speech recognition cannot understand audio, the system logs the issue and continues listening for new commands."
/>

## Project Deliverables

Complete the following to finish this capstone component:

1. Implement speech recognition and natural language understanding
2. Create global path planning algorithm
3. Develop local navigation and obstacle avoidance
4. Integrate all components and test in simulation
5. Document the performance of each component
6. Evaluate the system's response to dynamic environments

## Extension Activities

- Implement more sophisticated natural language processing
- Add semantic mapping for better understanding of locations
- Integrate with Isaac's optimized perception algorithms
- Create fallback behaviors for when navigation fails