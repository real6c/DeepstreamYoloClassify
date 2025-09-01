FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive

# Apt package update/upgrade
RUN apt update
RUN apt install python3-pip -y
RUN apt install curl git nano -y

# Install TensorRT 8
#RUN apt install -y nvidia-tensorrt

# Install pip packages
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python onnxslim pillow matplotlib onnx ultralytics meson ninja
RUN pip3 install "numpy<2.0"
RUN pip3 install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

# DeepStream
RUN git clone https://github.com/GNOME/glib.git
WORKDIR /glib
RUN git checkout 2.85.1 && \
    meson build --prefix=/usr && \
    ninja -C build && \
    ninja -C build install

RUN apt install -y \
libssl3 \
libssl-dev \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev

RUN pkg-config --modversion glib-2.0

WORKDIR /
RUN git clone https://github.com/confluentinc/librdkafka.git
WORKDIR /librdkafka
RUN git checkout tags/v2.2.0
RUN ./configure --enable-ssl
RUN make
RUN make install
#RUN mkdir -p /opt/nvidia/deepstream/deepstream/lib

WORKDIR /workspace/pkg
RUN curl -L 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/7.0/files?redirect=true&path=deepstream_sdk_v7.0.0_jetson.tbz2' -o 'deepstream_sdk_v7.0.0_jetson.tbz2'
RUN tar --overwrite -xvf deepstream_sdk_v7.0.0_jetson.tbz2 -C /

RUN cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-7.0/lib

WORKDIR /opt/nvidia/deepstream/deepstream-7.0
RUN ./install.sh

ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-7.1/lib/:/usr/lib/aarch64-linux-gnu/:$LD_LIBRARY_PATH
RUN ldconfig

WORKDIR /workspace
