#!/bin/bash

set -x

usage() { echo "$__usage" >&2; }

while getopts "c:" opt
do
    case ${opt} in
        c) CONFIG_PATH=$OPTARG;;
        *) usage; exit 1 ;;
    esac
done

HOST_CODE_DIR="PATH/TO/part-based-scan-understanding"

CUDA_VISIBLE_DEVICES=0 python ${HOST_CODE_DIR}/train_gnn_scannet.py \
  --config ${HOST_CODE_DIR}/${CONFIG_PATH}
