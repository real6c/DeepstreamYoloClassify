sudo docker run --runtime nvidia --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --network=host -v ~/DeepStreamYoloCls:/workspace -it deepstreamyolo_cls
