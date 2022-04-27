#docker kill $(docker ps -a -q); docker rm $(docker ps -a -q )

CONTAINER_NAME=mmdetection3d-cuda11.1-pytorch1.9

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}


