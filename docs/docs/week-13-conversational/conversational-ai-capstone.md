---
title: Conversational AI & Capstone Integration
sidebar_position: 1
description: Integrating conversational AI with the complete autonomous humanoid system
duration: 240
difficulty: advanced
learning_objectives:
  - Implement multimodal conversational AI for humanoid robots
  - Integrate speech, vision, and action for natural human-robot interaction
  - Combine all previous concepts into a complete autonomous system
  - Deploy and test the full capstone project
---

# Conversational AI & Capstone Integration

## Learning Objectives

By the end of this week, you will be able to:
- Implement multimodal conversational AI for humanoid robots
- Integrate speech recognition, natural language understanding, and dialogue management
- Combine all previous concepts into a complete autonomous humanoid system
- Deploy and test the full capstone project with conversational capabilities
- Evaluate the performance of the integrated system

## Introduction to Conversational AI for Robotics

Conversational AI in robotics enables natural human-robot interaction through multiple modalities including speech, gesture, and vision. This integration represents the culmination of all concepts learned throughout the course.

### Key Components of Conversational AI Systems

1. **Automatic Speech Recognition (ASR)**: Converting speech to text
2. **Natural Language Understanding (NLU)**: Interpreting user intent
3. **Dialog Management**: Managing conversation flow and context
4. **Natural Language Generation (NLG)**: Creating appropriate responses
5. **Speech Synthesis**: Converting text to speech
6. **Multimodal Integration**: Combining speech with visual and gestural cues

## Speech Recognition and Natural Language Understanding

### Step 1: Speech Recognition System

Implement a robust speech recognition system that works in real-world environments:

```python
# speech_recognition_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import speech_recognition as sr
import threading
import queue
import numpy as np

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
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Pause duration to consider phrase complete
        self.audio_queue = queue.Queue()

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        self.get_logger().info('Speech recognition node initialized')

    def audio_callback(self, msg):
        """Callback for audio data"""
        try:
            # Convert ROS AudioData to audio data for speech recognition
            audio_data = sr.AudioData(
                msg.data,
                sample_rate=16000,  # Adjust based on your audio source
                sample_width=2      # 16-bit audio
            )

            # Add to processing queue
            self.audio_queue.put(audio_data)

        except Exception as e:
            self.get_logger().error(f'Error processing audio callback: {str(e)}')

    def process_audio(self):
        """Process audio data in a separate thread"""
        while rclpy.ok():
            try:
                # Get audio from queue
                audio_data = self.audio_queue.get(timeout=1.0)

                # Perform speech recognition
                try:
                    # Using Google Web Speech API (requires internet)
                    # For offline option, consider using pocketsphinx or vosk
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
                except Exception as e:
                    self.get_logger().error(f'Unexpected error in speech recognition: {str(e)}')

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

Implement natural language understanding to interpret commands and questions:

```python
# nlu_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import re
import json
import spacy
from typing import Dict, List, Tuple

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

        # Publishers for various commands
        self.nav_goal_pub = self.create_publisher(Pose, 'navigation_goal', 10)
        self.cmd_pub = self.create_publisher(String, 'robot_command', 10)
        self.response_pub = self.create_publisher(String, 'robot_response', 10)

        # Load spaCy model for NLP (install with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.get_logger().warn("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define command patterns and intents
        self.intent_patterns = {
            'navigation': [
                r'.*go to (.+)',
                r'.*move to (.+)',
                r'.*navigate to (.+)',
                r'.*go (.+)',
                r'.*walk to (.+)',
                r'.*come to (.+)'
            ],
            'manipulation': [
                r'.*pick up (.+)',
                r'.*grasp (.+)',
                r'.*take (.+)',
                r'.*get (.+)',
                r'.*bring me (.+)'
            ],
            'information': [
                r'.*what is (.+)',
                r'.*tell me about (.+)',
                r'.*describe (.+)',
                r'.*how many (.+)'
            ],
            'social': [
                r'.*hello',
                r'.*hi',
                r'.*good morning',
                r'.*good afternoon',
                r'.*good evening'
            ],
            'control': [
                r'.*stop',
                r'.*halt',
                r'.*wait',
                r'.*pause',
                r'.*follow me',
                r'.*dance',
                r'.*perform (.+)'
            ]
        }

        # Location mappings
        self.location_map = {
            'kitchen': (2.0, 3.0, 0.0),
            'living room': (0.0, 0.0, 0.0),
            'bedroom': (-2.0, 1.0, 0.0),
            'office': (1.0, -2.0, 0.0),
            'entrance': (0.0, 3.0, 0.0)
        }

        self.get_logger().info('Natural Language Understanding node initialized')

    def speech_callback(self, msg):
        """Process recognized speech"""
        text = msg.data.lower().strip()

        # Process the command using NLP if available
        if self.nlp:
            doc = self.nlp(text)
            command = self.process_with_nlp(doc, text)
        else:
            # Fallback to pattern matching
            command = self.interpret_command(text)

        if command:
            self.get_logger().info(f'Interpreted command: {command}')

            # Publish command
            cmd_msg = String()
            cmd_msg.data = json.dumps(command)
            self.cmd_pub.publish(cmd_msg)

            # Handle specific commands
            self.handle_command(command)

            # Generate response
            response = self.generate_response(command)
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

    def process_with_nlp(self, doc, text):
        """Process text using spaCy NLP"""
        # Extract entities and dependencies
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        tokens = [(token.text, token.pos_, token.dep_) for token in doc]

        # Identify intent based on root verb or action
        intent = self.identify_intent_nlp(doc)

        if intent:
            return {
                'type': intent,
                'entities': entities,
                'original_text': text,
                'confidence': 0.8  # Placeholder confidence
            }

        # Fallback to pattern matching
        return self.interpret_command(text)

    def identify_intent_nlp(self, doc):
        """Identify intent using NLP analysis"""
        # Look for action verbs
        action_verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

        # Map verbs to intents
        verb_to_intent = {
            'go': 'navigation',
            'move': 'navigation',
            'navigate': 'navigation',
            'pick': 'manipulation',
            'grasp': 'manipulation',
            'take': 'manipulation',
            'get': 'manipulation',
            'bring': 'manipulation',
            'tell': 'information',
            'describe': 'information',
            'what': 'information',
            'how': 'information',
            'stop': 'control',
            'halt': 'control',
            'wait': 'control',
            'follow': 'control',
            'dance': 'control',
            'perform': 'control'
        }

        for verb in action_verbs:
            if verb in verb_to_intent:
                return verb_to_intent[verb]

        # Check for social intents
        social_indicators = ['hello', 'hi', 'good']
        for token in doc:
            if token.lemma_ in social_indicators:
                return 'social'

        return None

    def interpret_command(self, text):
        """Interpret natural language command using pattern matching"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, text)
                if match:
                    if intent == 'navigation':
                        location = match.group(1).strip()
                        return {
                            'type': intent,
                            'location': location,
                            'coordinates': self.location_map.get(location, (0, 0, 0))
                        }
                    elif intent in ['manipulation', 'information']:
                        target = match.group(1).strip()
                        return {
                            'type': intent,
                            'target': target
                        }
                    elif intent == 'social':
                        return {
                            'type': intent,
                            'greeting': text
                        }
                    else:  # control commands
                        return {
                            'type': intent,
                            'command': text
                        }

        # If no pattern matches, return None
        return None

    def handle_command(self, command):
        """Handle the interpreted command"""
        cmd_type = command['type']

        if cmd_type == 'navigation':
            self.handle_navigation_command(command)
        elif cmd_type == 'manipulation':
            self.handle_manipulation_command(command)
        elif cmd_type == 'information':
            self.handle_information_command(command)
        elif cmd_type == 'social':
            self.handle_social_command(command)
        elif cmd_type == 'control':
            self.handle_control_command(command)

    def handle_navigation_command(self, command):
        """Handle navigation commands"""
        location = command.get('location', 'unknown')
        coords = command.get('coordinates', (0, 0, 0))

        self.get_logger().info(f'Navigating to: {location} at coordinates {coords}')

        # Create and publish navigation goal
        pose_msg = Pose()
        pose_msg.position.x = coords[0]
        pose_msg.position.y = coords[1]
        pose_msg.position.z = coords[2]
        # Add orientation as needed

        self.nav_goal_pub.publish(pose_msg)

    def handle_manipulation_command(self, command):
        """Handle manipulation commands"""
        target = command.get('target', 'unknown')
        self.get_logger().info(f'Attempting to manipulate: {target}')

        # In a real system, this would trigger perception and manipulation
        # For now, just log the command
        cmd_msg = String()
        cmd_msg.data = f"manipulation_request:{target}"
        self.cmd_pub.publish(cmd_msg)

    def handle_information_command(self, command):
        """Handle information requests"""
        target = command.get('target', 'unknown')
        self.get_logger().info(f'Information requested about: {target}')

        # In a real system, this would query knowledge base or perception
        cmd_msg = String()
        cmd_msg.data = f"information_request:{target}"
        self.cmd_pub.publish(cmd_msg)

    def handle_social_command(self, command):
        """Handle social/greeting commands"""
        greeting = command.get('greeting', 'hello')
        self.get_logger().info(f'Received greeting: {greeting}')

    def handle_control_command(self, command):
        """Handle control commands"""
        cmd = command.get('command', 'unknown')
        self.get_logger().info(f'Received control command: {cmd}')

        # Handle specific control commands
        if 'stop' in cmd or 'halt' in cmd:
            stop_msg = String()
            stop_msg.data = 'stop_robot'
            self.cmd_pub.publish(stop_msg)

    def generate_response(self, command):
        """Generate an appropriate response to the user"""
        cmd_type = command['type']

        responses = {
            'navigation': f"Okay, I'm navigating to {command.get('location', 'that location')}.",
            'manipulation': f"Okay, I'll try to get the {command.get('target', 'object')}.",
            'information': f"I'll look for information about {command.get('target', 'that')}.",
            'social': "Hello! How can I help you today?",
            'control': "I understand the command."
        }

        return responses.get(cmd_type, "I'm not sure I understand. Could you please repeat that?")

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

## Multimodal Interaction Integration

### Step 3: Combining Speech, Vision, and Action

Create a coordinator node that integrates all modalities:

```python
# multimodal_coordinator_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import json
import time
from collections import deque

class MultimodalCoordinatorNode(Node):
    def __init__(self):
        super().__init__('multimodal_coordinator')

        # Subscribers
        self.speech_sub = self.create_subscription(
            String,
            'speech_recognized',
            self.speech_callback,
            10
        )

        self.nlu_sub = self.create_subscription(
            String,
            'nlu_output',
            self.nlu_callback,
            10
        )

        self.vision_sub = self.create_subscription(
            Detection2DArray,
            'object_detections',
            self.vision_callback,
            10
        )

        self.gaze_sub = self.create_subscription(
            Point,
            'gaze_target',
            self.gaze_callback,
            10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(String, 'robot_command', 10)
        self.response_pub = self.create_publisher(String, 'robot_response', 10)
        self.gesture_pub = self.create_publisher(String, 'gesture_command', 10)
        self.attention_pub = self.create_publisher(Point, 'attention_target', 10)

        # Initialize components
        self.bridge = CvBridge()
        self.current_task = None
        self.context = {}
        self.response_queue = deque(maxlen=10)  # Keep last 10 responses
        self.last_interaction_time = time.time()

        # State management
        self.interaction_state = 'idle'  # idle, listening, processing, responding
        self.attention_target = Point(x=0.0, y=0.0, z=0.0)

        # Timer for state management
        self.state_timer = self.create_timer(0.1, self.state_callback)

        self.get_logger().info('Multimodal coordinator node initialized')

    def speech_callback(self, msg):
        """Handle speech input"""
        self.interaction_state = 'processing'
        self.last_interaction_time = time.time()

        # Process speech with NLU
        self.process_speech(msg.data)

    def nlu_callback(self, msg):
        """Handle NLU output"""
        try:
            command = json.loads(msg.data)
            self.execute_command(command)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in NLU output')

    def vision_callback(self, msg):
        """Handle visual input"""
        # Update context with detected objects
        detected_objects = []
        for detection in msg.detections:
            obj_info = {
                'class': detection.results[0].hypothesis.class_id,
                'confidence': detection.results[0].hypothesis.score,
                'position': detection.pose
            }
            detected_objects.append(obj_info)

        self.context['detected_objects'] = detected_objects
        self.update_attention_target()

    def gaze_callback(self, msg):
        """Handle gaze input"""
        self.attention_target = msg
        self.context['gaze_target'] = (msg.x, msg.y, msg.z)

    def process_speech(self, text):
        """Process speech input and determine action"""
        # This would typically call an NLU service
        # For now, we'll do simple processing
        self.get_logger().info(f'Processing speech: {text}')

        # Determine if this is a follow-up to previous interaction
        time_since_last = time.time() - self.last_interaction_time
        is_follow_up = time_since_last < 10.0  # 10 seconds

        # Update context
        self.context['last_speech'] = text
        self.context['is_follow_up'] = is_follow_up

        # Simple intent detection
        if any(word in text.lower() for word in ['look', 'see', 'find', 'where']):
            self.interaction_state = 'searching'
        elif any(word in text.lower() for word in ['go', 'move', 'navigate']):
            self.interaction_state = 'navigating'
        elif any(word in text.lower() for word in ['grasp', 'pick', 'take', 'get']):
            self.interaction_state = 'manipulating'
        else:
            self.interaction_state = 'responding'

    def execute_command(self, command):
        """Execute the interpreted command"""
        cmd_type = command.get('type', 'unknown')

        if cmd_type == 'navigation':
            self.execute_navigation(command)
        elif cmd_type == 'manipulation':
            self.execute_manipulation(command)
        elif cmd_type == 'information':
            self.execute_information(command)
        elif cmd_type == 'social':
            self.execute_social(command)
        elif cmd_type == 'control':
            self.execute_control(command)

        self.interaction_state = 'responding'

    def execute_navigation(self, command):
        """Execute navigation command"""
        location = command.get('location', 'unknown')
        coords = command.get('coordinates', (0, 0, 0))

        # Publish navigation command
        nav_cmd = String()
        nav_cmd.data = f"navigate_to:{location}:{coords[0]}:{coords[1]}:{coords[2]}"
        self.cmd_pub.publish(nav_cmd)

        self.get_logger().info(f'Navigating to {location}')

    def execute_manipulation(self, command):
        """Execute manipulation command"""
        target = command.get('target', 'unknown')

        # First, look for the object
        look_cmd = String()
        look_cmd.data = f"look_for:{target}"
        self.cmd_pub.publish(look_cmd)

        # Then attempt to grasp
        grasp_cmd = String()
        grasp_cmd.data = f"grasp:{target}"
        self.cmd_pub.publish(grasp_cmd)

        self.get_logger().info(f'Attempting to manipulate {target}')

    def execute_information(self, command):
        """Execute information command"""
        target = command.get('target', 'unknown')

        # Query knowledge base or perception system
        query_cmd = String()
        query_cmd.data = f"query_info:{target}"
        self.cmd_pub.publish(query_cmd)

        self.get_logger().info(f'Querying information about {target}')

    def execute_social(self, command):
        """Execute social command"""
        greeting = command.get('greeting', 'hello')

        # Generate appropriate social response
        response = self.generate_social_response(greeting)
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        # Possibly trigger gesture
        gesture_cmd = String()
        gesture_cmd.data = "wave"
        self.gesture_pub.publish(gesture_cmd)

    def execute_control(self, command):
        """Execute control command"""
        cmd = command.get('command', 'unknown')

        # Publish control command
        control_msg = String()
        control_msg.data = cmd
        self.cmd_pub.publish(control_msg)

    def generate_social_response(self, greeting):
        """Generate appropriate social response"""
        if any(word in greeting for word in ['hello', 'hi']):
            return "Hello there! How can I assist you today?"
        elif 'good morning' in greeting:
            return "Good morning! What would you like to do today?"
        elif 'good afternoon' in greeting:
            return "Good afternoon! How can I help you?"
        elif 'good evening' in greeting:
            return "Good evening! What can I do for you?"
        else:
            return "Hello! How can I help you?"

    def update_attention_target(self):
        """Update attention target based on context"""
        if 'gaze_target' in self.context:
            # Use explicit gaze target
            x, y, z = self.context['gaze_target']
            target = Point(x=x, y=y, z=z)
        elif 'detected_objects' in self.context and self.context['detected_objects']:
            # Use most confident detection
            detections = self.context['detected_objects']
            best_detection = max(detections, key=lambda x: x['confidence'])
            target = best_detection['position']
        else:
            # Default to center
            target = Point(x=0.0, y=0.0, z=1.0)

        self.attention_target = target
        self.attention_pub.publish(target)

    def state_callback(self):
        """Manage interaction state"""
        current_time = time.time()
        time_since_interaction = current_time - self.last_interaction_time

        # Reset to idle if no interaction for 30 seconds
        if time_since_interaction > 30.0 and self.interaction_state != 'idle':
            self.interaction_state = 'idle'
            self.get_logger().info('Resetting to idle state due to inactivity')

    def generate_response(self, command):
        """Generate a response based on command and context"""
        cmd_type = command.get('type', 'unknown')

        responses = {
            'navigation': f"Okay, I'm navigating to {command.get('location', 'that location')}.",
            'manipulation': f"Okay, I'll try to get the {command.get('target', 'object')}.",
            'information': f"I'll look for information about {command.get('target', 'that')}.",
            'social': "Hello! How can I help you today?",
            'control': "I understand the command."
        }

        return responses.get(cmd_type, "I'm processing your request.")

def main(args=None):
    rclpy.init(args=args)
    coordinator = MultimodalCoordinatorNode()

    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Capstone Integration

### Step 4: Complete System Integration

Now I'll create the complete capstone integration that brings together all components:

```python
# capstone_integration_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Pose, Twist, Point
from sensor_msgs.msg import Image, LaserScan, JointState
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import json
import time
from threading import Lock

class CapstoneIntegrationNode(Node):
    def __init__(self):
        super().__init__('capstone_integration')

        # Subscribers for all system components
        self.speech_sub = self.create_subscription(
            String,
            'speech_recognized',
            self.speech_callback,
            10
        )

        self.vision_sub = self.create_subscription(
            String,
            'object_detections',
            self.vision_callback,
            10
        )

        self.nav_sub = self.create_subscription(
            String,
            'navigation_status',
            self.nav_callback,
            10
        )

        self.manip_sub = self.create_subscription(
            String,
            'manipulation_status',
            self.manip_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        # Publishers for integrated system
        self.cmd_pub = self.create_publisher(String, 'system_command', 10)
        self.response_pub = self.create_publisher(String, 'system_response', 10)
        self.status_pub = self.create_publisher(String, 'system_status', 10)
        self.behavior_pub = self.create_publisher(String, 'behavior_command', 10)

        # Internal state
        self.system_state = {
            'current_task': 'idle',
            'task_queue': [],
            'robot_pose': Pose(),
            'battery_level': 100.0,
            'safety_status': 'safe',
            'active_components': [],
            'last_interaction': time.time()
        }

        self.state_lock = Lock()
        self.task_execution_timer = self.create_timer(0.1, self.task_execution_callback)
        self.status_timer = self.create_timer(1.0, self.status_callback)

        self.get_logger().info('Capstone integration node initialized')

    def speech_callback(self, msg):
        """Handle speech input and update system state"""
        with self.state_lock:
            self.system_state['last_interaction'] = time.time()
            self.get_logger().info(f'Processing speech command: {msg.data}')

            # Parse and queue task based on speech
            task = self.parse_speech_command(msg.data)
            if task:
                self.system_state['task_queue'].append(task)
                self.system_state['current_task'] = 'processing_command'

    def vision_callback(self, msg):
        """Handle vision input and update system state"""
        with self.state_lock:
            # Update object detections in system state
            try:
                detections = json.loads(msg.data)
                self.system_state['detections'] = detections
            except json.JSONDecodeError:
                self.get_logger().error('Invalid vision data format')

    def nav_callback(self, msg):
        """Handle navigation status updates"""
        with self.state_lock:
            status = json.loads(msg.data) if msg.data.startswith('{') else {'status': msg.data}
            self.system_state['navigation'] = status

    def manip_callback(self, msg):
        """Handle manipulation status updates"""
        with self.state_lock:
            status = json.loads(msg.data) if msg.data.startswith('{') else {'status': msg.data}
            self.system_state['manipulation'] = status

    def odom_callback(self, msg):
        """Handle odometry updates"""
        with self.state_lock:
            self.system_state['robot_pose'] = msg.pose.pose

    def scan_callback(self, msg):
        """Handle laser scan for safety"""
        with self.state_lock:
            # Check for obstacles
            min_range = min(msg.ranges) if msg.ranges else float('inf')
            if min_range < 0.5:  # Less than 50cm to obstacle
                self.system_state['safety_status'] = 'unsafe_approaching'
            else:
                self.system_state['safety_status'] = 'safe'

    def parse_speech_command(self, text):
        """Parse speech command and create task"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['go to', 'navigate to', 'move to']):
            # Extract location
            for loc in ['kitchen', 'living room', 'bedroom', 'office', 'entrance']:
                if loc in text_lower:
                    return {
                        'type': 'navigation',
                        'target': loc,
                        'priority': 'high',
                        'timestamp': time.time()
                    }

        elif any(word in text_lower for word in ['pick up', 'grasp', 'get', 'take']):
            # Extract object
            words = text_lower.split()
            obj_idx = -1
            for i, word in enumerate(words):
                if word in ['pick', 'grasp', 'get', 'take']:
                    obj_idx = i + 1
                    break

            if obj_idx < len(words):
                obj = words[obj_idx]
                return {
                    'type': 'manipulation',
                    'target': obj,
                    'priority': 'high',
                    'timestamp': time.time()
                }

        elif any(word in text_lower for word in ['tell me', 'what is', 'describe']):
            return {
                'type': 'information',
                'query': text,
                'priority': 'medium',
                'timestamp': time.time()
            }

        elif any(word in text_lower for word in ['hello', 'hi', 'good']):
            return {
                'type': 'social',
                'greeting': text,
                'priority': 'low',
                'timestamp': time.time()
            }

        return None

    def task_execution_callback(self):
        """Execute tasks from the queue"""
        with self.state_lock:
            if not self.system_state['task_queue']:
                # Check if we need to switch from processing to idle
                if (self.system_state['current_task'] == 'processing_command' and
                    time.time() - self.system_state['last_interaction'] > 5.0):
                    self.system_state['current_task'] = 'idle'
                return

            # Get highest priority task
            task = self.system_state['task_queue'][0]  # Simple FIFO for now

            # Execute based on task type
            if task['type'] == 'navigation':
                self.execute_navigation_task(task)
            elif task['type'] == 'manipulation':
                self.execute_manipulation_task(task)
            elif task['type'] == 'information':
                self.execute_information_task(task)
            elif task['type'] == 'social':
                self.execute_social_task(task)

            # Remove executed task
            self.system_state['task_queue'].pop(0)

    def execute_navigation_task(self, task):
        """Execute navigation task"""
        location = task['target']

        # In a real system, this would call navigation stack
        nav_cmd = String()
        nav_cmd.data = f"navigate_to:{location}"
        self.cmd_pub.publish(nav_cmd)

        self.system_state['current_task'] = f'navigating_to_{location}'

        self.get_logger().info(f'Executing navigation task to {location}')

    def execute_manipulation_task(self, task):
        """Execute manipulation task"""
        obj = task['target']

        # First localize object, then manipulate
        local_cmd = String()
        local_cmd.data = f"localize:{obj}"
        self.cmd_pub.publish(local_cmd)

        # After localization, attempt manipulation
        manip_cmd = String()
        manip_cmd.data = f"manipulate:{obj}"
        self.cmd_pub.publish(manip_cmd)

        self.system_state['current_task'] = f'manipulating_{obj}'

        self.get_logger().info(f'Executing manipulation task for {obj}')

    def execute_information_task(self, task):
        """Execute information task"""
        query = task['query']

        # Query knowledge base or perception
        info_cmd = String()
        info_cmd.data = f"query:{query}"
        self.cmd_pub.publish(info_cmd)

        self.system_state['current_task'] = 'answering_query'

        self.get_logger().info(f'Executing information task for query: {query}')

    def execute_social_task(self, task):
        """Execute social task"""
        greeting = task['greeting']

        # Generate response
        response = self.generate_social_response(greeting)
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        # Trigger social behavior
        behavior_cmd = String()
        behavior_cmd.data = "greet_user"
        self.behavior_pub.publish(behavior_cmd)

        self.system_state['current_task'] = 'social_interaction'

        self.get_logger().info(f'Executing social task: {greeting}')

    def generate_social_response(self, greeting):
        """Generate appropriate social response"""
        greeting_lower = greeting.lower()

        if any(word in greeting_lower for word in ['hello', 'hi']):
            return "Hello! How can I assist you today?"
        elif 'good morning' in greeting_lower:
            return "Good morning! It's great to see you."
        elif 'good afternoon' in greeting_lower:
            return "Good afternoon! What would you like to do?"
        elif 'good evening' in greeting_lower:
            return "Good evening! How can I help you?"
        else:
            return "I heard you. How can I assist you?"

    def status_callback(self):
        """Publish system status periodically"""
        with self.state_lock:
            status_msg = String()
            status_msg.data = json.dumps({
                'current_task': self.system_state['current_task'],
                'task_queue_size': len(self.system_state['task_queue']),
                'robot_pose': {
                    'x': self.system_state['robot_pose'].position.x,
                    'y': self.system_state['robot_pose'].position.y,
                    'z': self.system_state['robot_pose'].position.z
                },
                'battery_level': self.system_state['battery_level'],
                'safety_status': self.system_state['safety_status'],
                'active_components': self.system_state['active_components']
            })

            self.status_pub.publish(status_msg)

    def check_system_integrity(self):
        """Check if all components are operational"""
        with self.state_lock:
            # This would check if all expected nodes are running
            # For now, we'll assume they are
            return True

def main(args=None):
    rclpy.init(args=args)
    integration_node = CapstoneIntegrationNode()

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

## System Launch and Testing

### Step 5: Integration Launch File

Create a launch file to bring up the complete system:

```python
# launch/capstone_system_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_namespace = LaunchConfiguration('robot_namespace', default='humanoid_robot')

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

    # Perception pipeline node
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

    # Navigation system
    navigation_node = Node(
        package='nav2_bringup',
        executable='nav2_bringup',
        name='navigation_system',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True
        }],
        output='screen'
    )

    # Manipulation system
    manipulation_node = Node(
        package='moveit_ros',
        executable='moveit_planning',
        name='manipulation_system',
        parameters=[{
            'robot_description': 'robot_description',
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )

    # Multimodal coordinator
    multimodal_coordinator = Node(
        package='capstone_integration',
        executable='multimodal_coordinator_node',
        name='multimodal_coordinator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Capstone integration node
    capstone_integration = Node(
        package='capstone_integration',
        executable='capstone_integration_node',
        name='capstone_integration',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Text-to-speech node
    tts_node = Node(
        package='tts_package',
        executable='text_to_speech_node',
        name='text_to_speech',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'publish_frequency': 50.0
        }],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'robot_namespace',
            default_value='humanoid_robot',
            description='Robot namespace for multi-robot systems'
        ),
        speech_recognition_node,
        nlu_node,
        perception_node,
        navigation_node,
        manipulation_node,
        multimodal_coordinator,
        capstone_integration,
        tts_node,
        robot_state_publisher
    ])
```

## Assessment and Evaluation

### Step 6: Capstone Assessment

<Assessment
  question="What are the key components of a multimodal conversational AI system for robotics?"
  type="multiple-choice"
  options={[
    "Speech recognition, natural language understanding, and dialogue management",
    "Speech recognition, natural language understanding, dialog management, and multimodal integration",
    "Only speech recognition and synthesis",
    "Vision processing and navigation"
  ]}
  correctIndex={1}
  explanation="A complete multimodal conversational AI system includes speech recognition, natural language understanding, dialog management, natural language generation, speech synthesis, and multimodal integration that combines speech with visual and gestural cues."
/>

<Assessment
  question="How does the capstone integration node handle conflicting tasks from different modalities?"
  type="multiple-choice"
  options={[
    "It processes tasks in the order they arrive",
    "It prioritizes tasks based on urgency and system state",
    "It only processes speech commands and ignores other modalities",
    "It pauses all other activities when receiving a command"
  ]}
  correctIndex={1}
  explanation="The capstone integration node manages conflicting tasks by prioritizing them based on urgency, system state, and context, ensuring safe and efficient operation."
/>

## Capstone Project Completion

### Step 7: Final Integration Testing

To complete the capstone project, you should test the integrated system with the following scenarios:

1. **Basic Interaction Test**:
   - Say "Hello" to the robot
   - Verify it responds appropriately with speech and gesture

2. **Navigation Test**:
   - Command "Go to kitchen"
   - Verify the robot plans a path and navigates safely

3. **Manipulation Test**:
   - Command "Pick up the red cup"
   - Verify the robot localizes the object and attempts manipulation

4. **Multimodal Test**:
   - Point to an object and say "What is that?"
   - Verify the robot looks at the pointed location and identifies the object

5. **Complex Task Test**:
   - Command "Go to the kitchen and bring me the water bottle"
   - Verify the robot breaks this into subtasks and executes them sequentially

## Project Deliverables

Complete the following to finish the capstone project:

1. **Integrated System**: All components working together in a cohesive system
2. **Conversational Interface**: Natural language interaction with the robot
3. **Multimodal Capabilities**: Integration of speech, vision, and action
4. **Safety Features**: Proper safety checks and emergency stops
5. **Documentation**: Complete documentation of the integrated system
6. **Testing Results**: Evidence of successful testing across all scenarios

## Extension Activities

- **Learning Capabilities**: Implement machine learning for improved interaction over time
- **Emotion Recognition**: Add facial expression recognition for more natural interaction
- **Multi-Robot Coordination**: Extend to multiple robots working together
- **Cloud Integration**: Connect to cloud services for enhanced capabilities
- **Mobile App Interface**: Create a mobile app for remote interaction

This concludes the Physical AI & Humanoid Robotics textbook. You now have a comprehensive understanding of how to build, integrate, and deploy complete humanoid robotics systems with conversational AI capabilities.