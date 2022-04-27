
xhost +local:root 1>/dev/null 2>&1

CONTAINER_NAME=mmdetection3d-cuda11.1-pytorch1.9

docker exec -u root -it ${CONTAINER_NAME} /bin/bash

xhost -local:root 1>/dev/null 2>&1


