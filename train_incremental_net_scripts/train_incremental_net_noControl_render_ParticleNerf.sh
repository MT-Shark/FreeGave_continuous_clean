#!/bin/bash

# 延迟 1.2 小时（1 小时 12 分钟）
# sleep 4320

densify_grad_threshold=0.0002
size_threshold=20
exp=net_woNodeWeight_zero_padding
output=output
scenes="pendulums spring robot robot-task cloth wheel"
scenes="robot-task cloth wheel"
# scenes=(pen1)
for scene in $scenes; do
    case "$scene" in
        pendulums|spring|robot)
            dt=0.0071942
            ;;
        *)
            dt=0.0035842
            ;;
    esac
    echo "Scene: $scene, dt: $dt"
    python incremental_train_noControlNode_4.2.py \
        -s ParticleNerf_datasets/${scene} \
        -m ${output}/ParticleNerf_datasets/${scene}_woControlNode_${exp}\
        --eval \
        --is_blender \
        --dt ${dt} \
        --incremental_warmup_time 0.1 \
        --incremental_max_time 0.9 \
        --deform_max_time 0.05 \
        --iter_basis 20 \
        --iter_gs 100 \
        --noStatic_mask \
        --cn_init zero \
        --cn_KNN 3 \
        --cn_hyperdim 16 \
        --cn_KNN_method xyz \
        --control_num 10000 \
        --warmup_iter_basis 75 \
        --warmup_iter_gs 150 \
        --densify_grad_threshold ${densify_grad_threshold} \
        --size_threshold ${size_threshold} \
        --zero_padding

    # python metrics_plot.py -m ${output}/${scene}_controlNode_${exp} --iter_basis 20 --iter_gs 100 --noStatic_mask
done