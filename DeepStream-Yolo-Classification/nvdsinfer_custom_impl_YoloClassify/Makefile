# Makefile for YOLO Classification DeepStream Plugin
# Modified for YOLO Classification by Assistant

CUDA_VER?=11.4
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CXX:= g++
CC:= gcc

# DeepStream paths (for container environment)
DEEPSTREAM_PATH?=/opt/nvidia/deepstream/deepstream-7.0
TENSORRT_PATH?=/usr/lib/aarch64-linux-gnu
CUDA_PATH?=/usr/local/cuda

# Include paths
INCLUDES:= -I$(DEEPSTREAM_PATH)/sources/includes \
           -I$(DEEPSTREAM_PATH)/sources/custom_impl_Yolo \
           -I$(CUDA_PATH)/include \
           -I$(TENSORRT_PATH)/include \
           -I/usr/include \
           -I/usr/include/opencv4

# Library paths
LIBS:= -L$(DEEPSTREAM_PATH)/lib \
       -L$(TENSORRT_PATH)/lib \
       -L$(CUDA_PATH)/lib64

# Compiler flags
CFLAGS:= -Wall -std=c++14 -shared -fPIC
LDFLAGS:= -shared -Wl,--no-as-needed

# Libraries to link
LIBS+= -lnvinfer -lnvparsers -lnvonnxparser -lcudart -lcublas -lcurand -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Source files - only the working classification plugin
SRCS:= simple_classify_plugin.cpp

# Object files
OBJS:= $(SRCS:.cpp=.o)

# Target library
TARGET_LIB:= libnvdsinfer_custom_impl_YoloClassify.so

# Default target
all: $(TARGET_LIB)

# Build the shared library
$(TARGET_LIB): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean target
clean:
	rm -rf $(OBJS) $(TARGET_LIB)

# Install target
install: $(TARGET_LIB)
	cp $(TARGET_LIB) $(DEEPSTREAM_PATH)/lib/

# Debug target
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET_LIB)

.PHONY: all clean install debug
