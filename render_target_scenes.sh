#!/bin/bash

# ✅ 直接指定场景列表
scene_list=(
    "/media/HDD2/siyuan/PhysicsBenchmark/disk10/MovieRenders/Origin/TwoCombinations/1-3_4-1_ModernHouse_sequence"
    "/media/HDD2/siyuan/PhysicsBenchmark/disk10/MovieRenders/Origin/TwoCombinations/1-3_5-1_Office_sequence"
    "/media/HDD2/siyuan/PhysicsBenchmark/disk10/MovieRenders/Origin/TwoCombinations/1-3_4-1_ModernHouse_sequence"
    "/media/HDD2/siyuan/PhysicsBenchmark/disk10/MovieRenders/Origin/TwoCombinations/1-3_4-1_ModernHouse_sequence"
    "/media/HDD2/siyuan/PhysicsBenchmark/disk10/MovieRenders/Origin/TwoCombinations/1-3_4-1_ModernHouse_sequence"
        
)

# ✅ 指定 CUDA ID
cuda_id=0

# ✅ 遍历每个场景并执行渲染
for scene in "${scene_list[@]}"; do
    # 获取最后一级目录名称 scene_name 和倒数第二级目录名称 round_name
    scene_name=$(basename "$scene")
    round_name=$(basename "$(dirname "$scene")")
    dir_name=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$scene")")")")")  # diskX_test
    echo "$dir_name"
    echo "Processing scene: $scene"

    CUDA_VISIBLE_DEVICES=${cuda_id} python incremental_train_ControlNode.py \
        -s "${scene}" \
        -m "output/${dir_name}_vis/${round_name}/${scene_name}" \
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
        --max_depth -1 

    echo "Finished scene: $scene"
    echo "-----------------------------"
done