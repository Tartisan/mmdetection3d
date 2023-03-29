#!/usr/bin/bash

python convert_onnx_to_trt.py \
    --dataset cpdet \
    --mode int8 \
    --max-batch-size 1 \
    --max-workspace-size 2 \
    --fast-mode \
    --calibrator-cache backbone.calib \
    --calibrator-image-file tools/onnx_tools/tensorrt-scripts/mb-bp_val.txt \
