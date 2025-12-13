---
title: Physical AI Edge Kit Setup
sidebar_position: 2
description: Setting up the Physical AI Edge Kit for real-world robotics development and deployment
duration: 240
difficulty: advanced
learning_objectives:
  - Configure the Physical AI Edge Kit hardware for robotics applications
  - Install and optimize ROS 2 for edge computing environments
  - Set up perception and manipulation systems on edge hardware
  - Optimize AI models for edge deployment with resource constraints
---

# Physical AI Edge Kit Setup

## Learning Objectives

By the end of this section, you will be able to:
- Configure the Physical AI Edge Kit hardware for robotics applications
- Install and optimize ROS 2 for edge computing environments
- Set up perception and manipulation systems on edge hardware
- Optimize AI models for edge deployment with resource constraints

## Introduction to Edge Computing for Robotics

Edge computing in robotics refers to processing data locally on the robot rather than relying on cloud services. This approach provides:

- **Low Latency**: Immediate response to sensor data
- **Reliability**: Operation without internet connectivity
- **Privacy**: Sensitive data stays on-device
- **Bandwidth Efficiency**: Reduced data transmission requirements
- **Real-time Processing**: Critical for safety and performance

## Hardware Requirements and Specifications

### Edge Kit Components

The Physical AI Edge Kit typically includes:

- **Edge Computer**: NVIDIA Jetson AGX Orin, Jetson Orin NX, or similar
- **Sensors**: RGB-D camera, LIDAR, IMU, force/torque sensors
- **Actuators**: Servo motors, robotic arms, mobile base
- **Power System**: Batteries and power management
- **Connectivity**: WiFi, Ethernet, Bluetooth modules
- **Chassis**: Mounting system for components

### Recommended Specifications

For optimal performance:

- **Compute**: NVIDIA Jetson AGX Orin (64 CUDA cores, 2048 Max-QPUs)
- **Memory**: 32GB LPDDR5 RAM
- **Storage**: 1TB NVMe SSD
- **Power**: 12V DC input, 5-60W consumption
- **Connectivity**: Gigabit Ethernet, WiFi 6, Bluetooth 5.2
- **Operating Temperature**: -10°C to 50°C for reliable operation

## Edge Computer Setup

### Step 1: Initial Hardware Setup

1. **Unpack and Inspect**: Verify all components are present and undamaged
2. **Mount Edge Computer**: Secure the Jetson board in the chassis
3. **Connect Power**: Connect the power supply to the board
4. **Connect Peripherals**: Attach sensors and actuators according to wiring diagram

### Step 2: Flash the Operating System

Flash the Jetson board with the appropriate OS:

```bash
# For NVIDIA Jetson devices
# Download NVIDIA SDK Manager from NVIDIA developer website
# Connect Jetson to host PC via USB
# Put Jetson in recovery mode (hold REC button while pressing RST)
# Use SDK Manager to flash OS and install JetPack

# Alternative: Manual flashing (advanced users)
wget https://developer.nvidia.com/embedded/l4t/r35_release_v7.0/jetson_agx_orin_jetpack517_linux_r3531_aarch64 tbz2
tar -xjf jetson_agx_orin_jetpack517_linux_r3531_aarch64.tbz2
cd Linux_for_Tegra/
sudo ./flash.sh jetson-agx-orin-devkit mmcblk0p1
```

### Step 3: Initial Configuration

After the OS is flashed:

```bash
# Connect to the Jetson via SSH or direct connection
ssh jetson@<jetson-ip-address>
# Default password: nvidia

# Update the system
sudo apt update && sudo apt upgrade -y

# Configure wireless networking
sudo nano /etc/netplan/01-network-manager-all.yaml

# Example configuration:
network:
  version: 2
  renderer: networkd
  wifis:
    wlan0:
      access-points:
        "your_wifi_network":
          password: "your_password"
      dhcp4: true
      optional: true
```

## ROS 2 Installation and Configuration

### Step 1: Install ROS 2 Humble Hawksbill

Install ROS 2 optimized for the ARM64 architecture:

```bash
# Set locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.gpg | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-vcstool

# Source ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Install Robotics-Specific Packages

Install packages needed for robotics applications:

```bash
# Navigation and mapping
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox

# Computer vision
sudo apt install -y ros-humble-vision-opencv ros-humble-cv-bridge ros-humble-image-transport ros-humble-camera-calibration

# Manipulation
sudo apt install -y ros-humble-moveit ros-humble-moveit-visual-tools ros-humble-gripper-controllers

# Perception
sudo apt install -y ros-humble-pointcloud-to-laserscan ros-humble-robot-state-publisher ros-humble-joint-state-publisher

# Hardware interfaces
sudo apt install -y ros-humble-hardware-interface ros-humble-ros2-control ros-humble-ros2-controllers
```

### Step 3: Optimize ROS 2 for Edge Performance

Create optimization configurations:

```bash
# ~/edge_robot_ws/src/edge_optimization/config/ros_optimization.yaml
# Configuration for optimizing ROS 2 on edge hardware
system_optimization:
  cpu_affinity: true
  real_time_scheduling: false  # Use only if real-time kernel is installed
  memory_management:
    enable_swap: true
    swap_size_gb: 8

ros2_optimization:
  middleware:
    rmw_implementation: rmw_cyclonedds_cpp  # Lightweight DDS implementation
  qos_profiles:
    sensor_data: PRESET_QOS_PROFILE_SENSOR_DATA
    services_general: PRESET_QOS_PROFILE_SERVICES_GENERAL
    parameters: PRESET_QOS_PROFILE_PARAMETERS
  performance:
    intra_process_comms: true
    enable_statistics: false  # Disable for performance

logging_optimization:
  log_level: WARN  # Reduce logging overhead
  log_rotation: true
  max_log_size_mb: 10
```

## AI Model Optimization for Edge

### TensorRT Integration

Optimize deep learning models using NVIDIA TensorRT:

```python
# ~/edge_robot_ws/src/perception_pipeline/scripts/model_optimizer.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxBytes, TrtRunner

def optimize_model_for_edge(onnx_model_path, output_engine_path, precision="fp16"):
    """
    Optimize an ONNX model for edge deployment using TensorRT
    """
    # Load ONNX model
    with open(onnx_model_path, 'rb') as model_file:
        onnx_bytes = model_file.read()

    # Create TensorRT builder
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt.Logger())

    # Parse ONNX
    if not parser.parse(onnx_bytes):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return False

    # Configure builder
    config = builder.create_builder_config()

    # Set precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("FP16 not supported, using FP32")

    # Set workspace size (reduce for edge devices)
    config.max_workspace_size = 2 * 1024 * 1024 * 1024  # 2GB

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return False

    # Save optimized engine
    with open(output_engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Model optimized and saved to {output_engine_path}")
    return True

def benchmark_model(engine_path):
    """
    Benchmark the optimized model performance
    """
    # Load engine
    with open(engine_path, 'rb') as f:
        engine_data = f.read()

    engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    # Benchmark
    import time
    iterations = 100
    times = []

    for i in range(iterations):
        # Generate random input
        np.copyto(inputs[0]['host'], np.random.random(size=inputs[0]['host'].shape).astype(np.float32))

        # Transfer input data to device
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

        # Run inference
        start_time = time.time()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from device
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()
        end_time = time.time()

        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    avg_time = sum(times[10:]) / len(times[10:])  # Exclude first 10 for warmup
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.2f}")

    return avg_time

if __name__ == "__main__":
    # Example usage
    success = optimize_model_for_edge(
        "models/yolov8.onnx",
        "models/yolov8_optimized.engine",
        precision="fp16"
    )

    if success:
        benchmark_model("models/yolov8_optimized.engine")
```

### Model Compression Techniques

Implement model compression for resource-constrained environments:

```python
# ~/edge_robot_ws/src/perception_pipeline/scripts/model_compression.py
import torch
import torch.nn.utils.prune as prune
import numpy as np
from torch.quantization import quantize_dynamic, fuse_modules

def prune_model(model, sparsity=0.2):
    """
    Prune a PyTorch model to reduce size and improve inference speed
    """
    # Define layers to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))

    # Apply pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    print(f"Applied {sparsity*100}% sparsity to the model")
    return model

def quantize_model(model):
    """
    Quantize a PyTorch model to INT8 for reduced memory usage and faster inference
    """
    # Set model to evaluation mode
    model.eval()

    # Fuse operations for better quantization (optional)
    # fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )

    print("Model quantized to INT8")
    return quantized_model

def benchmark_compressed_models(original_model, pruned_model, quantized_model, input_tensor):
    """
    Compare performance of different model compression techniques
    """
    import time

    models = {
        "Original": original_model,
        "Pruned": pruned_model,
        "Quantized": quantized_model
    }

    results = {}

    for name, model in models.items():
        model.eval()
        times = []

        with torch.no_grad():
            for _ in range(20):  # Warmup
                _ = model(input_tensor)

            for _ in range(100):  # Benchmark
                start = time.time()
                _ = model(input_tensor)
                end = time.time()
                times.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        results[name] = avg_time
        print(f"{name} model average inference time: {avg_time:.2f} ms")

    return results

# Example usage for a dummy model
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    # Create a dummy model for demonstration
    model = DummyModel()

    # Create sample input
    input_tensor = torch.randn(1, 3, 224, 224)

    # Original model
    original_time = benchmark_compressed_models(
        model,
        prune_model(model, sparsity=0.2),
        quantize_model(model),
        input_tensor
    )

    print("\nModel compression results:")
    print("Lower inference time indicates better performance on edge hardware")

if __name__ == "__main__":
    main()
```

## Sensor Integration

### Camera Setup and Calibration

Configure and calibrate the RGB-D camera:

```bash
# Install camera drivers and calibration tools
sudo apt install -y ros-humble-camera-calibration ros-humble-image-proc ros-humble-depth-image-proc

# Create camera configuration
mkdir -p ~/edge_robot_ws/src/sensor_config/config/cameras/
```

```yaml
# ~/edge_robot_ws/src/sensor_config/config/cameras/rgbd_camera.yaml
# Camera configuration for RGB-D sensor
camera_name: rgbd_camera
image_height: 480
image_width: 640
camera_matrix:
  rows: 3
  cols: 3
  data: [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]  # Example calibration
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients from calibration
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]