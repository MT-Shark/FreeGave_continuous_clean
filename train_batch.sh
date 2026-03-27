#!/bin/bash

# 文件路径
scene_file="/media/SSD/siyuan/NGV_GVFi/FreeGave_continuous_clean/val.txt"
complete_file="freegave_continuous_complete.txt"

# 创建 complete.txt 如果不存在
touch "$complete_file"

# 读取所有 scenes 和已完成 scenes
mapfile -t all_scenes < "$scene_file"
mapfile -t completed_scenes < "$complete_file"

# 构建一个 hash 表用于快速查找已完成 scene
declare -A completed_map
for scene in "${completed_scenes[@]}"; do
    completed_map["$scene"]=1
done

# 设置 GPU 列表
gpus=(7 8 9)  # 根据你的机器修改
num_gpus=${#gpus[@]}

# 过滤出未完成的 scenes
pending_scenes=()
for scene in "${all_scenes[@]}"; do
    if [[ -z "${completed_map[$scene]}" ]]; then
        pending_scenes+=("$scene")
    fi
done

# 计算每个 GPU 分配多少个 scene
total_pending=${#pending_scenes[@]}
scenes_per_gpu=$(( (total_pending + num_gpus - 1) / num_gpus ))

# 分配任务并并行执行
for ((i=0; i<num_gpus; i++)); do
    (
        start=$((i * scenes_per_gpu))
        end=$((start + scenes_per_gpu))
        if (( end > total_pending )); then
            end=$total_pending
        fi

        for ((j=start; j<end; j++)); do
            scene=${pending_scenes[j]}
            echo "GPU ${gpus[i]} processing scene: $scene"

            CUDA_VISIBLE_DEVICES=${gpus[i]} python incremental_train_ControlNode.py \
                -s "${scene}" \
                -m "output/$(basename "${scene}")" \
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
                --max_depth -1 \
                --skip_render

            # 如果成功，记录到 complete.txt
            echo "$scene" >> "$complete_file"
            echo "GPU ${gpus[i]} finished scene: $scene"
            echo "-----------------------------"
        done
    ) &
done

# 等待所有 GPU 完成任务
wait
echo "All pending scenes processed."