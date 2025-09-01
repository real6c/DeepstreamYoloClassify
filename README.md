# TrashBot DeepStream YOLO Classification System

A production-ready YOLO11x classification system for Jetson devices using NVIDIA DeepStream SDK 7.0, packaged in Docker for easy deployment and development.

Note: This project was tested with the following yolo classify model, repo found [here](https://huggingface.co/datasets/lreal/BingRecycle40k)

## 🏗️ Project Structure

```
TrashBotDeepstreamYolo/
├── Dockerfile                          # DeepStream 7.0 container definition
├── build.sh                            # Build the Docker container
├── start.sh                            # Start the container with proper mounts
├── xServer.sh                          # X11 display setup for GUI
├── .gitignore                          # Git ignore rules
└── DeepStream-Yolo-Classification/     # Main classification system
    ├── run_classification.sh          # Production watchdog script
    ├── build_plugin.sh                # Build the plugin
    ├── nvdsinfer_custom_impl_YoloClassify/  # Custom DeepStream plugin
    ├── deepstream_app_config_webcam.txt     # Webcam pipeline config
    ├── config_infer_primary_yolo11_classify.txt  # YOLO inference config
    └── README_DEEPSTREAM.md                      # Classification system docs
```

## 🐳 Docker Setup

### Prerequisites

- **Jetson Device**: Tested on a Jetpack l4t 36.4.4 jetson orin nano device
- **Docker**: NVIDIA Container Runtime enabled
- **USB Webcam**: Connected to `/dev/video0`
- **Monitor**: For display output

### Quick Start

- **NOTE:** Please take a look at the ```start.sh``` and make sure the volume path is to this repo!

1. **Set up X11 display** (for GUI output):
   ```bash
   ./xServer.sh
   ```

2. **Build the container**:
   ```bash
   ./build.sh
   ```

3. **Start the container**:
   ```bash
   ./start.sh
   ```

## 🔧 Container Details

### Base Image
- **L4T JetPack**: `nvcr.io/nvidia/l4t-jetpack:r36.3.0`
- **CUDA**: 12.2 (included in JetPack)
- **TensorRT**: 8.6 (included in JetPack)

### DeepStream SDK
- **Version**: 7.0
- **Installation**: `/opt/nvidia/deepstream/deepstream-7.0`
- **Custom Plugin Path**: `/opt/nvidia/deepstream/deepstream-7.0/lib`

### Key Dependencies
- **Python 3.10**: With ML packages (OpenCV, ONNX, Ultralytics)
- **GStreamer**: Full multimedia framework
- **GLib 2.85.1**: From source for compatibility
- **librdkafka**: For streaming capabilities
- **ONNX Runtime GPU**: Optimized for ARM64

## 🚀 Usage

### Inside the Container

1. **Navigate to workspace**:
   ```bash
   cd /workspace/DeepStream-Yolo-Classification
   ```

2. **Follow instructions in ```README_DEEPSTREAM.md```**:

### Container Features

- **GPU Access**: Full NVIDIA GPU support via `--runtime nvidia`
- **Webcam Access**: USB camera mounted at `/dev/video0`
- **Display Output**: X11 forwarding for monitor display
- **Volume Mount**: Host directory mounted at `/workspace`
- **Network**: Host network mode for full access

## 📁 File Descriptions

### Docker Files
- **`Dockerfile`**: Multi-stage build with DeepStream 7.0
- **`build.sh`**: Container build script with network host mode
- **`start.sh`**: Container run script with device mounts
- **`xServer.sh`**: X11 display configuration

## 🔍 Troubleshooting

### Common Issues

1. **X11 Display Error**:
   ```bash
   ./xServer.sh
   # Then restart container
   ```

2. **Webcam Not Found**:
   ```bash
   # Check device exists
   ls -la /dev/video0
   # Verify permissions
   sudo chmod 666 /dev/video0
   ```

3. **GPU Memory Issues**:
   - Reduce `workspace-size` in inference config
   - Lower camera resolution in webcam config
   - Monitor with `nvidia-smi`

4. **Plugin Build Errors**:
   ```bash
   # Clean and rebuild
   cd nvdsinfer_custom_impl_YoloClassify
   make clean && make
   ```

### Performance Tuning

- **FP16 Precision**: Already configured for optimal performance
- **Engine Caching**: TensorRT engine cached for fast startup
- **Memory Management**: Optimized buffer sizes and queue lengths
- **Session Timeout**: 2-hour sessions with auto-restart
