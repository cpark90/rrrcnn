#!/bin/bash

NAME=cpark90/dacon:detectron
DATAPATH=/media/TrainDataset/cpark
OUTPUTPATH=/media/LTDataset/cpark
BASELINE=~/git/rrrcnn
DIRNAME=$(cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)

docker run -it --rm --gpus all --ipc=host -e DISPLAY=:20 -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix -v $DATAPATH:/ws/data -v $OUTPUTPATH:/ws/output -v $BASELINE:/ws/external $NAME
