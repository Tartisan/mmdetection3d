#/bin/bash
CONTAINER_NAME=mmdetection3d-cuda11.1-pytorch1.9
IMG="mmdet3d:waymo-0417"

MMDET3D_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd -P )"

docker stop ${CONTAINER_NAME} > /dev/null 2>&1
docker rm ${CONTAINER_NAME} > /dev/null 2>&1
sleep 1

USER_ID=$(id -u)
GRP=$(id -g -n)
GRP_ID=$(id -g)
LOCAL_HOST=`hostname`
DOCKER_HOME="/home/${USER}"

if [ "${USER}" == "root" ];then
    DOCKER_HOME="/root"
fi
if [ ! -d "${HOME}/.cache" ];then
    mkdir "${HOME}/.cache"
fi

LOCAL_VOLUME="-v /media:/media \
			  -v ${MMDET3D_ROOT_DIR}:/mmdetection3d"

docker run -it --gpus all -d \
		--privileged \
		--name ${CONTAINER_NAME} \
		-e DOCKER_USER=${USER} \
		-e USER=${USER} \
		-e DOCKER_USER_ID=${USER_ID} \
		-e DOCKER_GRP=${GRP} \
		-e DOCKER_GRP_ID=${GRP_ID} \
		-e DISPLAY=${DISPLAY} \
		-e USE_GPU=${USE_GPU} \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,video,utility \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v ${HOME}/.cache:${DOCKER_HOME}/.cache \
		-v /etc/localtime:/etc/localtime:ro \
		${LOCAL_VOLUME} \
		--net host \
		--shm-size 8g \
		-w /mmdetection3d \
		$IMG \
		/bin/bash
