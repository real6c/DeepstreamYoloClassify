#!/bin/bash

echo "Building YOLO Classification Plugin..."

cd nvdsinfer_custom_impl_YoloClassify

# Clean and build
make clean
make

if [ $? -eq 0 ]; then
    echo "Plugin built successfully!"
    echo "Installing to DeepStream lib directory..."
    cp libnvdsinfer_custom_impl_YoloClassify.so /opt/nvidia/deepstream/deepstream-7.0/lib/
    echo "Installation complete!"
else
    echo "Build failed!"
    exit 1
fi
