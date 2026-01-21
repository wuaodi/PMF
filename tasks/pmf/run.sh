#!/bin/bash
# PMF训练脚本 - H200单卡版本

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
# GPU选择统一在config_server_kitti.yaml中配置，这里不再设置

python main.py config_server_kitti.yaml
