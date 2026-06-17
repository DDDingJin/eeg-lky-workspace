#!/bin/bash

cd ..

# 执行训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node=4 \
    train.py \
    --use_ddp \
    --use_spatial_gat