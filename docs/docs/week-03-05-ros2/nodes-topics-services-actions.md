---
title: Nodes, Topics, Services, and Actions
sidebar_position: 2
description: Understanding the fundamental communication patterns in ROS 2
duration: 150
difficulty: intermediate
learning_objectives:
  - Implement ROS 2 nodes in Python and C++
  - Create and use topics for asynchronous communication
  - Implement services for synchronous request-response communication
  - Use actions for long-running tasks with feedback
---

# Nodes, Topics, Services, and Actions

## Learning Objectives

By the end of this section, you will be able to:
- Implement ROS 2 nodes in Python and C++
- Create and use topics for asynchronous communication
- Implement services for synchronous request-response communication
- Use actions for long-running tasks with feedback

## Nodes

Nodes are the fundamental building blocks of a ROS 2 system. Each node is a process that performs a specific function and communicates with other nodes through topics, services, and actions.

### Creating a Node in Python

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        self.get_logger().info('MyNode has been started')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Node in C++

```cpp
#include "rclcpp/rclcpp.hpp"

class MyNode : public rclcpp::Node
{
public:
    MyNode() : Node("my_node_name")
    {
        RCLCPP_INFO(this->get_logger(), "MyNode has been started");
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyNode>());
    rclcpp::shutdown();
    return 0;
}
```

### Node Lifecycle

Nodes go through several states during their lifetime:

1. **Construction**: Node object is created
2. **Initialization**: ROS 2 context is initialized
3. **Active**: Node is running and processing callbacks
4. **Shutdown**: Node is shutting down gracefully

## Topics

Topics provide asynchronous, many-to-many communication using a publish-subscribe pattern.

### Publisher Implementation

#### Python Publisher
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

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

#### C++ Publisher
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World: " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};
```

### Subscriber Implementation

#### Python Subscriber
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

#### C++ Subscriber
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber()
    : Node("minimal_subscriber")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic", 10,
            std::bind(&MinimalSubscriber::topic_callback, this, std::placeholders::_1));
    }

private:
    void topic_callback(const std_msgs::msg::String & msg) const
    {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg.data.c_str());
    }
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};
```

### Quality of Service (QoS) for Topics

Different QoS profiles can be used for different use cases:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# For sensor data (real-time, best effort)
sensor_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# For critical data (reliable, persistent)
critical_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

## Services

Services provide synchronous request-response communication.

### Service Definition

First, define the service in an `.srv` file (e.g., `AddTwoInts.srv`):

```
int64 a
int64 b
---
int64 sum
```

### Service Server Implementation

#### Python Service Server
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

#### C++ Service Server
```cpp
#include "example_interfaces/srv/add_two_ints.hpp"
#include "rclcpp/rclcpp.hpp"

class MinimalService : public rclcpp::Node
{
public:
    MinimalService()
    : Node("minimal_service")
    {
        service_ = create_service<example_interfaces::srv::AddTwoInts>(
            "add_two_ints",
            [this](const example_interfaces::srv::AddTwoInts::Request::SharedPtr request,
                   example_interfaces::srv::AddTwoInts::Response::SharedPtr response) {
                response->sum = request->a + request->b;
                RCLCPP_INFO(rclcpp::get_logger("minimal_service"),
                           "Incoming request\na: %" PRId64 " b: %" PRId64,
                           request->a, request->b);
                return response;
            });
    }

private:
    rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
};
```

### Service Client Implementation

#### Python Service Client
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

#### C++ Service Client
```cpp
#include "example_interfaces/srv/add_two_ints.hpp"
#include "rclcpp/rclcpp.hpp"

using AddTwoInts = example_interfaces::srv::AddTwoInts;

class MinimalClient : public rclcpp::Node
{
public:
    MinimalClient()
    : Node("minimal_client")
    {
        client_ = create_client<AddTwoInts>("add_two_ints");
        while (!client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
        }
    }

    void send_request(int64_t a, int64_t b)
    {
        auto request = std::make_shared<AddTwoInts::Request>();
        request->a = a;
        request->b = b;

        auto result_future = client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), result_future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_INFO(this->get_logger(), "Result of add_two_ints: %" PRId64,
                       result_future.get()->sum);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to call service add_two_ints");
        }
    }

private:
    rclcpp::Client<AddTwoInts>::SharedPtr client_;
};
```

## Actions

Actions are used for long-running tasks that require feedback and the ability to cancel.

### Action Definition

Define the action in an `.action` file (e.g., `Fibonacci.action`):

```
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

### Action Server Implementation

#### Python Action Server
```python
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.partial_sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result
```

### Action Client Implementation

#### Python Action Client
```python
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.partial_sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
```

## Communication Pattern Selection

### When to Use Each Pattern

| Pattern | Use Case | Characteristics |
|---------|----------|----------------|
| **Topics** | Streaming data, sensor readings | Asynchronous, many-to-many, fire-and-forget |
| **Services** | Simple request-response | Synchronous, one-to-one, blocking |
| **Actions** | Long-running tasks | Asynchronous, with feedback and cancelation |

### Performance Considerations

- **Topics**: Best for high-frequency, real-time data
- **Services**: Good for infrequent, blocking operations
- **Actions**: Ideal for complex, long-running operations

## Interactive Elements

### Communication Patterns Assessment

import Assessment from '@site/src/components/Assessment/Assessment';

<Assessment
  question="Which communication pattern is most appropriate for sending continuous sensor data from a robot's camera?"
  options={[
    {id: 'a', text: 'Service - because it provides reliable delivery'},
    {id: 'b', text: 'Topic - because it supports high-frequency streaming data'},
    {id: 'c', text: 'Action - because it provides feedback'},
    {id: 'd', text: 'Parameter - because it allows configuration'}
  ]}
  correctAnswerId="b"
  explanation="Topics are the most appropriate for streaming sensor data because they support high-frequency publishing, are asynchronous, and allow multiple subscribers to receive the same data stream."
/>

## Summary

Understanding the different communication patterns in ROS 2 is crucial for effective robot software development. Topics provide asynchronous communication for streaming data, services offer synchronous request-response interactions, and actions handle long-running tasks with feedback. Choosing the right pattern for each use case is essential for building robust and efficient robotic systems.

In the next section, we'll explore building ROS 2 packages with Python, focusing on the development workflow and best practices.