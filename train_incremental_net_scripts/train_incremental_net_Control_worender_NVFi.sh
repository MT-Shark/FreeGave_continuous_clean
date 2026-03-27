#!/bin/bash

# 延迟 1.2 小时（1 小时 12 分钟）
# sleep 4320

densify_grad_threshold=0.0002
size_threshold=20
exp=nowarmup_net_woNodeWeight_dbqInterpolate_xyzKNN_1knn_10000CN_zeropadding
output=output
scenes="bat fallingball fan telescope shark whale"
# scenes="shark whale"
for scene in $scenes; do
    python incremental_train_ControlNode_worender_4.2.py \
        -s NVFi_datasets/InDoorObj/data/${scene} \
        -m ${output}/NVFi/${scene}_controlNode_${exp}\
        --eval \
        --is_blender \
        --dt 0.016666 \
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

    # python metrics_plot.py -m ${output}/NVFi/${scene}_controlNode_${exp} --iter_basis 20 --iter_gs 100 --noStatic_mask
done


scenes="chessboard darkroom dining factory"
# scenes="chessboard darkroom dining"
# scenes=(pen1)
for scene in $scenes; do
    python incremental_train_ControlNode_worender_4.2.py \
        -s NVFi_datasets/InDoorSeg/data/${scene} \
        -m ${output}/NVFi/${scene}_controlNode_${exp}\
        --eval \
        --is_blender \
        --incremental_warmup_time 0.1 \
        --incremental_max_time 0.9 \
        --deform_max_time 0.05 \
        --iter_basis 20 \
        --iter_gs 100 \
        --noStatic_mask \
        --cn_init zero \
        --with_node_weight \
        --cn_KNN 3 \
        --cn_hyperdim 16 \
        --cn_KNN_method xyz \
        --control_num 10000 \
        --warmup_iter_basis 75 \
        --warmup_iter_gs 150 \
        --densify_grad_threshold ${densify_grad_threshold} \
        --size_threshold ${size_threshold} \
        --zero_padding

    # python metrics_plot.py -m ${output}/NVFi/${scene}_controlNode_${exp} --iter_basis 20 --iter_gs 100 --noStatic_mask
done
