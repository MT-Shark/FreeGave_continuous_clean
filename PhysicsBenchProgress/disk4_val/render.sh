#!/bin/bash

# 获取当前文件所在的文件夹路径，例如 dir_path = /media/SSD/siyuan/NGV_GVFi/FreeGave_continuous_clean/PhysicsBenchProgress/disk0_val
dir_path=$(dirname "$(realpath "$0")")

# 进一步获得文件夹名称 dir_name = disk0_val
dir_name=$(basename "$dir_path")

# 得到 存放所有目标文件夹路径的 txt 文件 all_paths_txt = {dir_path}/disk0_val.txt
all_paths_txt="${dir_path}/${dir_name}.txt"

# 直接读取所有路径
mapfile -t scenes < "$all_paths_txt"

# 指定 cuda_id
cuda_id=3

# 遍历每个场景并执行训练脚本
for scene in "${scenes[@]}"; do
    # scene 是一个完整路径 例如：/media/SSD2/data/PhysicsBenchmark/disk0/MovieRenders/Origin/FifthRound/1-1_11_15-2_Office_sequence

    # 需要得到 最后一级目录名称 scene_name 和 倒数第二级目录名称 round_name
    scene_name=$(basename "$scene")
    round_name=$(basename "$(dirname "$scene")")

    echo "Processing scene: $scene"

    CUDA_VISIBLE_DEVICES=${cuda_id} python incremental_train_ControlNode.py \
        -s "${scene}" \
        -m "output/${dir_name}/${round_name}/${scene_name}" \
        --eval \
        --is_blender \
        --dt 0.01 \
        --incremental_warmup_time 0 \
        --incremental_max_time 0.5 \
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
        --max_depth -1 
        # --sampleTrainingView 100

    echo "Finished scene: $scene"

    # 将这个 scene append 到 complete_paths_txt 中
    echo "$scene" >> "$complete_paths_txt"
    echo "-----------------------------"
done
