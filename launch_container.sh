DATA_DIR=$1
MODEL_DIR=$2
OUTPUT_DIR=$3

if [ "$5" = "--prepro" ]; then
    RO=""
else
    RO=",readonly"
fi


if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$DATA_DIR,dst=/src/_datasets,type=bind$RO \
    --mount src=$MODEL_DIR,dst=/src/_models,type=bind \
    --mount src=$OUTPUT_DIR,dst=/src/_snapshot,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 \
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    -w /src linjieli222/lavender:latest \
    bash -c "source ./docker_local_setup.sh && bash" 
