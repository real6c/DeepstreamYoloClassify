# DeepStream YOLO Classification System

Production-ready YOLO11x classification system for USB webcam with automatic restart capability.

## Files

### Core System
- `run_classification.sh` - Main watchdog script for continuous classification
- `build.sh` - Build and install the classification plugin

### DeepStream Configuration
- `deepstream_app_config_webcam.txt` - Webcam pipeline configuration
- `config_infer_primary_yolo11_classify.txt` - YOLO model inference configuration

### Plugin
- `nvdsinfer_custom_impl_YoloClassify/` - Custom DeepStream classification plugin
  - `simple_classify_plugin.cpp` - Plugin source code
  - `Makefile` - Build configuration
  - `libnvdsinfer_custom_impl_YoloClassify.so` - Compiled plugin

### Models
- `my_model.onnx` - YOLO11x classification model (39 classes)
- `model_b1_gpu0_fp16.engine` - Cached TensorRT engine (FP16)
- `labels.txt` - Class labels

### Logs
- `classification.log` - Main classification log
- `errors.log` - Error log

## Configuring Deepstream Pipeline

1. Upload your yolo classify onnx model, I am using yolov11x, you may have to change some settings in the config files if you use a different model

2. Make sure to configure ```labels.txt``` with your class labels

3. Configure the num classes in ```config_infer_primary_yolo11_classify.txt```

## Running Deepstream Pipeline

1. Build the plugin:
   ```bash
   ./build.sh
   ```

2. Build cached model (only do this the first time running, takes a while to build .engine file):
   ```
   deepstream-app -c deepstream_app_config_webcam.txt
   ```

3. Rename built engine file (again, only do this the first time). If your model is for example ```my_model.onnx```, you will get the filename: ```my_model.onnx_b1_gpu0_fp32.engine```, regardless of filename, it needs to be renamed to: ```model_b1_gpu0_fp16.engine``` as the config expects: ```model_b1_gpu0_fp16.engine```
   ```
   mv my_model.onnx_b1_gpu0_fp32.engine model_b1_gpu0_fp16.engine
   ```

4. Start classification:
   ```bash
   ./run_classification.sh
   ```

## Features

- **Continuous Operation**: Runs indefinitely with automatic restart on crashes
- **Almost Fast Recovery**: 15-second restart delay (still working on making this quicker)
- **USB Webcam**: Uses `/dev/video0` at 640x480@15fps (looking into CSI for better latency soon)
- **FP16 Optimization**: TensorRT engine with FP16 precision
- **Engine Caching**: Reuses compiled TensorRT engine for fast startup

## Monitoring

The system automatically logs:
- Classification results
- Restart events
- Error messages
- System status

Press `Ctrl+C` to stop the system cleanly.
