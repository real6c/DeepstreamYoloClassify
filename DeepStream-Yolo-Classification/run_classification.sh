#!/bin/bash

echo "=== DeepStream YOLO Classification ==="
echo "Simple continuous classification with fast auto-restart"
echo "Press Ctrl+C to stop everything cleanly"

# Configuration
LOG_FILE="classification.log"
ERROR_LOG="errors.log"
RESTART_DELAY=2  # 2 seconds between restarts

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$ERROR_LOG"
}

# Function to check if DeepStream is running
is_deepstream_running() {
    pgrep -f "deepstream-app" > /dev/null
}

# Function to start DeepStream (simple, no timeout)
start_deepstream() {
    log_message "Starting DeepStream classification..."
    
    # Set webcam format
    v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG 2>/dev/null || true
    
    # Start DeepStream without timeout - let it run until it crashes
    deepstream-app -c deepstream_app_config_webcam.txt 2>&1 | while IFS= read -r line; do
        echo "$line"
        # Log classification results
        if echo "$line" | grep -q "Classification:"; then
            echo "$line" >> "$LOG_FILE"
        fi
        # Check for errors
        if echo "$line" | grep -q "ERROR\|cudaError\|Failed"; then
            log_error "DeepStream error: $line"
        fi
    done &
    
    DEEPSTREAM_PID=$!
    log_message "DeepStream started with PID: $DEEPSTREAM_PID"
}

# Function to stop DeepStream (simple)
stop_deepstream() {
    log_message "Stopping DeepStream..."
    if [ ! -z "$DEEPSTREAM_PID" ]; then
        kill $DEEPSTREAM_PID 2>/dev/null
    fi
    pkill -f "deepstream-app" 2>/dev/null
    sleep 1
}

# Function to simple cleanup (no GPU reset)
simple_cleanup() {
    log_message "Simple cleanup..."
    # Just kill any remaining processes
    pkill -f "deepstream-app" 2>/dev/null
    sleep 1
}

# Function to check prerequisites
check_prerequisites() {
    log_message "Checking prerequisites..."
    
    if [ ! -e /dev/video0 ]; then
        log_error "Webcam /dev/video0 not found!"
        return 1
    fi
    
    if [ ! -f "model_b1_gpu0_fp16.engine" ]; then
        log_error "TensorRT engine not found!"
        return 1
    fi
    
    if [ ! -f "nvdsinfer_custom_impl_YoloClassify/libnvdsinfer_custom_impl_YoloClassify.so" ]; then
        log_error "Classification plugin not found!"
        return 1
    fi
    
    log_message "All prerequisites verified"
    return 0
}

# Function to monitor and restart DeepStream (simple)
monitor_and_restart() {
    local restart_count=0
    
    while true; do
        # Check if DeepStream is still running
        if ! is_deepstream_running; then
            restart_count=$((restart_count + 1))
            log_message "DeepStream stopped (restart #$restart_count)"
            
            # Simple cleanup and restart
            simple_cleanup
            start_deepstream
            
            # Wait before next check
            sleep $RESTART_DELAY
        else
            # Check every 30 seconds
            sleep 30
        fi
    done
}

# Function to cleanup everything
cleanup() {
    echo ""
    log_message "Shutting down classification system..."
    stop_deepstream
    log_message "Cleanup complete. Goodbye!"
    exit 0
}

# Main function
main() {
    log_message "=== Classification System Starting ==="
    log_message "Configuration: Continuous sessions, ${RESTART_DELAY}s restart delay"
    
    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed, exiting"
        exit 1
    fi
    
    # Start DeepStream
    start_deepstream
    
    # Wait a moment for DeepStream to start
    sleep 3
    
    # Check if DeepStream started successfully
    if is_deepstream_running; then
        log_message "DeepStream started successfully"
    else
        log_error "Failed to start DeepStream"
        exit 1
    fi
    
    # Start monitoring in background
    monitor_and_restart &
    MONITOR_PID=$!
    
    log_message "Monitoring started with PID: $MONITOR_PID"
    log_message "Classification system is running. Press Ctrl+C to stop."
    
    # Wait for user interrupt
    wait
}

# Set up signal handlers for clean shutdown
trap cleanup SIGINT SIGTERM

# Start the system
main
