#!/bin/bash

# 定义所有要处理的场景
scenes=(
    "1-1_4-1_4-2_Office_sequence"
    # "1-1_ModernHouse_chenghu_sequence"
    # "1-2_Factory_shouwang_sequence"
    # "3-1_10_17_ModernHouse_sequence"
    # "5-1_8-1_15-2_Office_sequence"
)

# 遍历每个场景并执行训练脚本
for scene in "${scenes[@]}"; do
    echo "Processing scene: $scene"

    CUDA_VISIBLE_DEVICES=6 python incremental_train_ControlNode.py \
        -s "data/${scene}" \
        -m "output/${scene}_depthMask_5m" \
        --eval \
        --is_blender \
        --dt 0.01 \
        --incremental_warmup_time 0 \
        --incremental_max_time 0.9 \
        --iter_basis 50 \
        --iter_gs 150 \
        --noStatic_mask \
        --cn_init zero \
        --cn_KNN 1 \
        --cn_hyperdim 16 \
        --cn_KNN_method xyz \
        --control_num 10000 \
        --warmup_iter_basis 120 \
        --warmup_iter_gs 200 \
        --cn_interpolate_method dqb \
        --predict_steps 10 \
        --densify_grad_threshold 0.0002 \
        --size_threshold 20 \
        --static_warmup_iter 3000 \
        --zero_padding \
        --max_depth 5
        # --skip_render
        # --sampleTrainingView 100

    echo "Finished scene: $scene"
    echo "-----------------------------"
done

