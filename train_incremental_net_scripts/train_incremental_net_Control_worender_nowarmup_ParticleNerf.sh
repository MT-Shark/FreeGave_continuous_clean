#!/bin/bash

# 延迟 1.2 小时（1 小时 12 分钟）
# sleep 4320

densify_grad_threshold=0.0002
size_threshold=20
exp=nowarmup_net_woNodeWeight_dbqInterpolate_xyzKNN_1knn_10000CN_zeropadding
output=output
scenes="pendulums spring robot robot-task cloth wheel"
# scenes="pendulums spring"
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
    python incremental_train_ControlNode_4.2_worender_nowarmup.py \
        -s ParticleNerf_datasets/${scene} \
        -m ${output}/ParticleNerf_datasets/${scene}_controlNode_${exp}\
        --eval \
        --is_blender \
        --dt ${dt} \
        --incremental_warmup_time 0 \
        --incremental_max_time 0.9 \
        --deform_max_time 0.05 \
        --iter_basis 20 \
        --iter_gs 100 \
        --noStatic_mask \
        --cn_init zero \
        --with_node_weight \
        --cn_KNN 1 \
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