---
title: Perception and Manipulation - Capstone Components
sidebar_position: 2
description: Components for perception systems and robotic manipulation in the capstone project
duration: 300
difficulty: advanced
learning_objectives:
  - Implement perception systems for object detection and recognition
  - Design robotic manipulation algorithms for humanoid robots
  - Integrate perception with manipulation for task execution
  - Create robust manipulation behaviors for real-world scenarios
---

# Perception and Manipulation - Capstone Components

## Learning Objectives

By the end of this section, you will be able to:
- Implement perception systems for object detection and recognition using Isaac Platform
- Design and implement robotic manipulation algorithms for humanoid robots
- Integrate perception and manipulation systems for coordinated task execution
- Create robust manipulation behaviors that handle real-world uncertainties
- Evaluate and optimize manipulation performance in various scenarios

## Project Overview

This capstone component focuses on perception and manipulation systems for humanoid robots. You'll implement object detection, recognition, and grasping algorithms that work together to enable the robot to interact with objects in its environment.

## Prerequisites

- Completion of Isaac Platform fundamentals (Weeks 8-10)
- Understanding of ROS 2 (Weeks 3-5)
- Experience with perception systems (Weeks 8-10)
- Basic knowledge of robotic manipulation concepts

## Perception System Implementation

### Step 1: Object Detection and Recognition

First, implement the object detection and recognition system using Isaac's optimized algorithms:

```python
# object_detection_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import numpy as np
import message_filters

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Use message filters to synchronize image and camera info
        image_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera_info')

        # Approximate time synchronizer
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, info_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.image_info_callback)

        # Publisher for detections
        self.detection_pub = self.create_publisher(Detection2DArray, '/object_detections', 10)

        # Store camera parameters
        self.camera_matrix = None

        self.get_logger().info('Object detection node initialized')

    def image_info_callback(self, image_msg, info_msg):
        """Callback for synchronized image and camera info"""
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Update camera matrix
            self.camera_matrix = np.array(info_msg.k).reshape(3, 3)

            # Perform object detection
            detections = self.detect_objects(cv_image)

            # Create and publish detection array
            detection_array = self.create_detection_array(detections, image_msg.header)
            self.detection_pub.publish(detection_array)

        except Exception as e:
            self.get_logger().error(f'Error in image_info_callback: {str(e)}')

    def detect_objects(self, image):
        """Detect objects in the image using Isaac's optimized algorithms"""
        # This is a simplified implementation
        # In a real Isaac-based system, you would use Isaac's optimized detection networks
        detections = []

        # For demonstration, we'll use a simple color-based detection
        # In practice, use Isaac DetectNet or similar optimized networks
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different objects
        color_ranges = {
            'red': (np.array([0, 50, 50]), np.array([10, 255, 255])),
            'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
            'green': (np.array([40, 50, 50]), np.array([80, 255, 255]))
        }

        for obj_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center in image coordinates
                    center_x = x + w / 2
                    center_y = y + h / 2

                    # Calculate 3D position using camera parameters
                    if self.camera_matrix is not None:
                        z_depth = 1.0  # Placeholder depth - in real system, use depth camera
                        obj_x = (center_x - self.camera_matrix[0, 2]) * z_depth / self.camera_matrix[0, 0]
                        obj_y = (center_y - self.camera_matrix[1, 2]) * z_depth / self.camera_matrix[1, 1]
                        obj_z = z_depth
                    else:
                        obj_x, obj_y, obj_z = 0.0, 0.0, 1.0

                    detection = {
                        'class': obj_name,
                        'confidence': 0.8,
                        'bbox': (x, y, w, h),
                        'position_3d': (obj_x, obj_y, obj_z)
                    }

                    detections.append(detection)

        return detections

    def create_detection_array(self, detections, header):
        """Create Detection2DArray message from detections"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            detection_2d = Detection2D()
            detection_2d.header = header

            # Set bounding box
            bbox = BoundingBox2D()
            bbox.center.x = detection['bbox'][0] + detection['bbox'][2] / 2
            bbox.center.y = detection['bbox'][1] + detection['bbox'][3] / 2
            bbox.size_x = detection['bbox'][2]
            bbox.size_y = detection['bbox'][3]
            detection_2d.bbox = bbox

            # Set classification
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection['class']
            hypothesis.hypothesis.score = detection['confidence']

            detection_2d.results.append(hypothesis)

            # Add 3D position as pose
            pose = Pose2D()
            pose.x = detection['position_3d'][0]
            pose.y = detection['position_3d'][1]
            pose.theta = 0.0  # Placeholder orientation
            detection_2d.pose = pose

            detection_array.detections.append(detection_2d)

        return detection_array

def main(args=None):
    rclpy.init(args=args)
    object_detection_node = ObjectDetectionNode()

    try:
        rclpy.spin(object_detection_node)
    except KeyboardInterrupt:
        pass
    finally:
        object_detection_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 2: 3D Object Pose Estimation

Implement 3D pose estimation for objects:

```python
# pose_estimation_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, Point
from vision_msgs.msg import Detection2DArray
from tf2_ros import TransformBroadcaster
from tf2_geometry_msgs import PointStamped
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')

        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for object poses
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)

        # Transform broadcaster for object poses
        self.tf_broadcaster = TransformBroadcaster(self)

        # Store camera parameters
        self.camera_matrix = None
        self.depth_image = None

        self.get_logger().info('Pose estimation node initialized')

    def camera_info_callback(self, msg):
        """Callback for camera info"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def depth_callback(self, msg):
        """Callback for depth image"""
        try:
            # Convert depth image to numpy array
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def detection_callback(self, msg):
        """Callback for object detections"""
        for detection in msg.detections:
            # Get 3D position from bounding box center and depth
            if self.depth_image is not None and self.camera_matrix is not None:
                center_x = int(detection.bbox.center.x)
                center_y = int(detection.bbox.center.y)

                # Get depth at center of bounding box
                depth = self.depth_image[center_y, center_x] / 1000.0  # Convert to meters

                if depth > 0 and depth < 5.0:  # Valid depth range
                    # Convert pixel coordinates to 3D world coordinates
                    x = (center_x - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
                    y = (center_y - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
                    z = depth

                    # Create pose message
                    pose_msg = PoseStamped()
                    pose_msg.header = msg.header
                    pose_msg.header.frame_id = 'camera_link'  # or appropriate frame
                    pose_msg.pose.position.x = x
                    pose_msg.pose.position.y = y
                    pose_msg.pose.position.z = z

                    # Set orientation (identity for now)
                    pose_msg.pose.orientation.w = 1.0

                    # Publish pose
                    self.pose_pub.publish(pose_msg)

                    # Broadcast transform
                    self.broadcast_transform(pose_msg, detection.results[0].hypothesis.class_id)

    def broadcast_transform(self, pose_msg, object_name):
        """Broadcast transform for detected object"""
        from geometry_msgs.msg import TransformStamped

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = f'{object_name}_frame'

        t.transform.translation.x = pose_msg.pose.position.x
        t.transform.translation.y = pose_msg.pose.position.y
        t.transform.translation.z = pose_msg.pose.position.z

        t.transform.rotation = pose_msg.pose.orientation

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    pose_estimation_node = PoseEstimationNode()

    try:
        rclpy.spin(pose_estimation_node)
    except KeyboardInterrupt:
        pass
    finally:
        pose_estimation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Manipulation System Implementation

### Step 3: Grasp Planning and Execution

Implement grasp planning and execution for the humanoid robot:

```python
# grasp_planning_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Vector3
from sensor_msgs.msg import PointCloud2
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan
from std_msgs.msg import String
import numpy as np
from scipy.spatial import distance

class GraspPlanningNode(Node):
    def __init__(self):
        super().__init__('grasp_planning_node')

        # Subscribers
        self.object_pose_sub = self.create_subscription(
            Pose,
            '/target_object_pose',
            self.object_pose_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            10
        )

        # Publishers
        self.grasp_pose_pub = self.create_publisher(Pose, '/grasp_pose', 10)
        self.cmd_pub = self.create_publisher(String, '/manipulation_command', 10)

        # Service client for motion planning
        self.motion_plan_cli = self.create_client(GetMotionPlan, '/plan_kinematic_path')

        # Store object and point cloud data
        self.target_object_pose = None
        self.point_cloud = None

        self.get_logger().info('Grasp planning node initialized')

    def object_pose_callback(self, msg):
        """Callback for target object pose"""
        self.target_object_pose = msg
        self.plan_grasp()

    def pointcloud_callback(self, msg):
        """Callback for point cloud data"""
        # Convert PointCloud2 to numpy array (simplified)
        # In practice, use point_cloud2 library
        self.point_cloud = msg

    def plan_grasp(self):
        """Plan a grasp for the target object"""
        if self.target_object_pose is None:
            return

        # Calculate approach and grasp poses
        object_pose = self.target_object_pose

        # Calculate grasp pose - approach from above (simplified)
        grasp_pose = Pose()
        grasp_pose.position.x = object_pose.position.x
        grasp_pose.position.y = object_pose.position.y
        grasp_pose.position.z = object_pose.position.z + 0.15  # 15cm above object

        # Set orientation for grasp (simplified)
        # In practice, calculate optimal grasp orientation based on object shape
        grasp_pose.orientation.w = 1.0  # Identity quaternion (looking down)

        # Calculate pre-grasp pose
        pre_grasp_pose = Pose()
        pre_grasp_pose.position.x = object_pose.position.x
        pre_grasp_pose.position.y = object_pose.position.y
        pre_grasp_pose.position.z = object_pose.position.z + 0.25  # 25cm above object
        pre_grasp_pose.orientation = grasp_pose.orientation

        # Check if grasp is feasible
        if self.is_grasp_feasible(grasp_pose):
            # Publish grasp poses
            self.publish_grasp_sequence(pre_grasp_pose, grasp_pose)
        else:
            self.get_logger().warn('Planned grasp is not feasible')

    def is_grasp_feasible(self, grasp_pose):
        """Check if grasp is kinematically feasible"""
        # In a real system, this would check:
        # 1. Robot kinematic constraints
        # 2. Collision avoidance
        # 3. Reachability
        # 4. Grasp stability

        # For this example, we'll return True
        return True

    def publish_grasp_sequence(self, pre_grasp_pose, grasp_pose):
        """Publish sequence of grasp poses"""
        # Publish pre-grasp pose
        self.grasp_pose_pub.publish(pre_grasp_pose)

        # Publish grasp pose
        self.grasp_pose_pub.publish(grasp_pose)

        # Publish command to execute grasp
        cmd_msg = String()
        cmd_msg.data = 'execute_grasp'
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    grasp_planning_node = GraspPlanningNode()

    try:
        rclpy.spin(grasp_planning_node)
    except KeyboardInterrupt:
        pass
    finally:
        grasp_planning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Manipulation Control Node

Implement the manipulation control system:

```python
# manipulation_control_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from actionlib_msgs.msg import GoalStatus
import numpy as np
import math

class ManipulationControlNode(Node):
    def __init__(self):
        super().__init__('manipulation_control_node')

        # Subscribers
        self.grasp_pose_sub = self.create_subscription(
            Pose,
            '/grasp_pose',
            self.grasp_pose_callback,
            10
        )

        self.cmd_sub = self.create_subscription(
            String,
            '/manipulation_command',
            self.command_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publishers
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(Float64, '/gripper_controller/command', 10)

        # Store current state
        self.current_joint_positions = {}
        self.target_pose = None

        # Manipulation states
        self.manipulation_state = 'idle'  # idle, moving_to_pregrasp, grasping, lifting, placing

        self.get_logger().info('Manipulation control node initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            self.current_joint_positions[name] = msg.position[i]

    def grasp_pose_callback(self, msg):
        """Receive grasp pose"""
        self.target_pose = msg
        self.get_logger().info(f'Received grasp pose: ({msg.position.x}, {msg.position.y}, {msg.position.z})')

    def command_callback(self, msg):
        """Receive manipulation command"""
        command = msg.data

        if command == 'execute_grasp':
            self.execute_grasp_sequence()
        elif command == 'release_object':
            self.release_object()
        elif command == 'reset_arm':
            self.reset_arm()

    def execute_grasp_sequence(self):
        """Execute complete grasp sequence"""
        if self.target_pose is None:
            self.get_logger().warn('No target pose available for grasping')
            return

        self.manipulation_state = 'moving_to_pregrasp'
        self.get_logger().info('Starting grasp sequence')

        # Move to pre-grasp position
        pre_grasp_pose = self.calculate_pre_grasp_pose(self.target_pose)
        self.move_to_pose(pre_grasp_pose, speed=0.5)

        # Wait for movement to complete (simplified)
        # In practice, wait for trajectory execution feedback
        self.get_logger().info('Reached pre-grasp position')

        # Move to grasp position
        self.manipulation_state = 'grasping'
        self.move_to_pose(self.target_pose, speed=0.2)  # Slower for precision

        # Close gripper
        self.close_gripper()

        # Lift object
        self.manipulation_state = 'lifting'
        lift_pose = self.calculate_lift_pose(self.target_pose)
        self.move_to_pose(lift_pose, speed=0.3)

        self.get_logger().info('Grasp sequence completed')

    def calculate_pre_grasp_pose(self, target_pose):
        """Calculate pre-grasp pose above target"""
        pre_grasp = Pose()
        pre_grasp.position.x = target_pose.position.x
        pre_grasp.position.y = target_pose.position.y
        pre_grasp.position.z = target_pose.position.z + 0.15  # 15cm above
        pre_grasp.orientation = target_pose.orientation
        return pre_grasp

    def calculate_lift_pose(self, target_pose):
        """Calculate lift pose after grasping"""
        lift_pose = Pose()
        lift_pose.position.x = target_pose.position.x
        lift_pose.position.y = target_pose.position.y
        lift_pose.position.z = target_pose.position.z + 0.2  # Lift 20cm
        lift_pose.orientation = target_pose.orientation
        return lift_pose

    def move_to_pose(self, pose, speed=0.5):
        """Move arm to specified pose"""
        # In a real system, this would:
        # 1. Perform inverse kinematics to find joint angles
        # 2. Create joint trajectory
        # 3. Send to trajectory controller

        # For this example, we'll create a simple trajectory
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.header.frame_id = 'base_link'

        # Define joint names (example for a 6-DOF arm)
        trajectory_msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Create trajectory point
        point = JointTrajectoryPoint()
        # Calculate desired joint positions (simplified - in practice use IK)
        point.positions = [0.0, -1.0, 1.0, 0.0, 1.0, 0.0]  # Placeholder values
        point.velocities = [speed] * 6
        point.accelerations = [speed * 2] * 6

        # Set duration based on distance and speed
        point.time_from_start.sec = 2  # 2 seconds (simplified)

        trajectory_msg.points.append(point)

        self.trajectory_pub.publish(trajectory_msg)

    def close_gripper(self):
        """Close the gripper to grasp object"""
        gripper_cmd = Float64()
        gripper_cmd.data = 0.0  # Close position (0.0 to 1.0 range)
        self.gripper_pub.publish(gripper_cmd)
        self.get_logger().info('Gripper closed')

    def open_gripper(self):
        """Open the gripper to release object"""
        gripper_cmd = Float64()
        gripper_cmd.data = 1.0  # Open position (0.0 to 1.0 range)
        self.gripper_pub.publish(gripper_cmd)
        self.get_logger().info('Gripper opened')

    def release_object(self):
        """Release the currently grasped object"""
        self.open_gripper()
        self.manipulation_state = 'idle'
        self.get_logger().info('Object released')

    def reset_arm(self):
        """Reset arm to home position"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.header.frame_id = 'base_link'

        trajectory_msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Home position
        point.velocities = [0.5] * 6
        point.accelerations = [1.0] * 6
        point.time_from_start.sec = 3

        trajectory_msg.points.append(point)

        self.trajectory_pub.publish(trajectory_msg)
        self.manipulation_state = 'idle'
        self.get_logger().info('Arm reset to home position')

def main(args=None):
    rclpy.init(args=args)
    manipulation_control_node = ManipulationControlNode()

    try:
        rclpy.spin(manipulation_control_node)
    except KeyboardInterrupt:
        pass
    finally:
        manipulation_control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration and Testing

### Step 5: Perception-Manipulation Integration

Create an integration node that coordinates perception and manipulation:

```python
# perception_manipulation_integration_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String, Bool
from vision_msgs.msg import Detection2DArray
from tf2_ros import Buffer, TransformListener
import time

class PerceptionManipulationIntegrationNode(Node):
    def __init__(self):
        super().__init__('perception_manipulation_integration_node')

        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10
        )

        self.manip_status_sub = self.create_subscription(
            String,
            '/manipulation_status',
            self.manip_status_callback,
            10
        )

        # Publishers
        self.target_pose_pub = self.create_publisher(Pose, '/target_object_pose', 10)
        self.manip_cmd_pub = self.create_publisher(String, '/manipulation_command', 10)
        self.system_status_pub = self.create_publisher(String, '/system_status', 10)

        # TF buffer for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # System state
        self.current_task = 'idle'
        self.detected_objects = {}
        self.active_object = None

        # Timer for system monitoring
        self.system_timer = self.create_timer(1.0, self.system_monitor)

        self.get_logger().info('Perception-manipulation integration node initialized')

    def detection_callback(self, msg):
        """Process object detections"""
        for detection in msg.detections:
            obj_class = detection.results[0].hypothesis.class_id
            confidence = detection.results[0].hypothesis.score

            if confidence > 0.7:  # Confidence threshold
                # Store detected object with timestamp
                self.detected_objects[obj_class] = {
                    'pose': detection.pose,
                    'confidence': confidence,
                    'timestamp': time.time()
                }

        # Check if we have objects to manipulate
        if self.current_task == 'idle' and self.detected_objects:
            self.select_object_for_manipulation()

    def manip_status_callback(self, msg):
        """Receive manipulation status updates"""
        status = msg.data
        self.get_logger().info(f'Manipulation status: {status}')

        # Handle state transitions based on manipulation status
        if status == 'grasp_completed':
            self.current_task = 'object_grasped'
            self.publish_system_status('Object successfully grasped')
        elif status == 'grasp_failed':
            self.current_task = 'idle'
            self.publish_system_status('Grasp failed, returning to idle')
        elif status == 'object_released':
            self.current_task = 'idle'
            self.publish_system_status('Object released, ready for next task')

    def select_object_for_manipulation(self):
        """Select an object for manipulation based on criteria"""
        if not self.detected_objects:
            return

        # For this example, select the highest confidence object
        best_obj = max(self.detected_objects.items(), key=lambda x: x[1]['confidence'])
        obj_name, obj_data = best_obj

        if obj_data['confidence'] > 0.8:  # Higher threshold for manipulation
            self.active_object = obj_name
            self.current_task = 'approaching_object'

            # Transform pose to robot base frame
            try:
                # Convert pose to stamped for transformation
                pose_stamped = PoseStamped()
                pose_stamped.header = msg.header
                pose_stamped.pose = obj_data['pose']

                # Transform to robot base frame
                base_frame = 'base_link'  # or appropriate base frame
                transformed_pose = self.tf_buffer.transform(pose_stamped, base_frame, timeout=rclpy.duration.Duration(seconds=1.0))

                # Publish target pose
                self.target_pose_pub.publish(transformed_pose.pose)

                # Send command to manipulate
                cmd_msg = String()
                cmd_msg.data = 'execute_grasp'
                self.manip_cmd_pub.publish(cmd_msg)

                self.get_logger().info(f'Selected {obj_name} for manipulation')
                self.publish_system_status(f'Approaching {obj_name}')

            except Exception as e:
                self.get_logger().error(f'Transform failed: {str(e)}')
                self.current_task = 'idle'  # Return to idle if transform fails

    def system_monitor(self):
        """Monitor system status and handle timeouts"""
        current_time = time.time()

        # Check for stale object detections
        for obj_name, obj_data in list(self.detected_objects.items()):
            if current_time - obj_data['timestamp'] > 5.0:  # 5 seconds
                del self.detected_objects[obj_name]

        # Handle task timeouts
        if (self.current_task.startswith('approaching') or
            self.current_task.startswith('grasping')) and current_time % 30 < 1.0:
            # If task has been running for too long, reset
            self.get_logger().warn(f'Task {self.current_task} timed out')
            self.current_task = 'idle'
            self.publish_system_status('Task timed out, returning to idle')

    def publish_system_status(self, status):
        """Publish system status"""
        status_msg = String()
        status_msg.data = status
        self.system_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    integration_node = PerceptionManipulationIntegrationNode()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        pass
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Assessment Questions

1. How does the perception system integrate with the manipulation system?

2. What are the key steps in the grasp planning process?

3. How does the system handle object pose estimation in 3D space?

<Assessment
  question="What is the primary function of the perception-manipulation integration node?"
  type="multiple-choice"
  options={[
    "To detect objects in the environment",
    "To coordinate between perception and manipulation systems",
    "To control the robot's arm joints",
    "To recognize speech commands"
  ]}
  correctIndex={1}
  explanation="The integration node coordinates between perception and manipulation systems, managing the flow of information and commands between them."
/>

<Assessment
  question="In the grasp planning process, what is the purpose of the pre-grasp pose?"
  type="multiple-choice"
  options={[
    "To firmly grasp the object",
    "To safely approach the object before grasping",
    "To lift the object after grasping",
    "To detect the object in the first place"
  ]}
  correctIndex={1}
  explanation="The pre-grasp pose is an intermediate position that allows the robot to safely approach the object before attempting to grasp it, reducing the risk of collisions."
/>

## Project Deliverables

Complete the following to finish this capstone component:

1. Implement object detection and recognition system
2. Create 3D pose estimation for detected objects
3. Develop grasp planning and execution algorithms
4. Integrate perception and manipulation systems
5. Test the complete system in simulation
6. Document performance metrics and limitations

## Extension Activities

- Implement more sophisticated grasp planning algorithms
- Add force control for compliant manipulation
- Integrate with Isaac's optimized perception pipelines
- Create a library of manipulation primitives for different object types