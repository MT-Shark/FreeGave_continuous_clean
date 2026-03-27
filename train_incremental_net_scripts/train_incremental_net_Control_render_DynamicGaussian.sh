#!/bin/bash

# 延迟 1.2 小时（1 小时 12 分钟）
# sleep 4320

densify_grad_threshold=0.0002
size_threshold=20
exp=net_woNodeWeight_zero_padding
output=output
static_warmup_iter=10000
scenes="basketball boxes football juggle softball tennis"
# scenes=(pen1)
for scene in $scenes; do
    python incremental_train_ControlNode_4.2.py \
        -s DynamicGaussian_datasets/${scene} \
        -m ${output}/DynamicGaussian/${scene}_controlNode_${exp}\
        --eval \
        --is_blender \
        --dt 0.006711 \
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
        --warmup_iter_basis 120 \
        --warmup_iter_gs 200 \
        --integrate_ksteps 3 \
        --densify_grad_threshold ${densify_grad_threshold} \
        --size_threshold ${size_threshold} \
        --static_warmup_iter ${static_warmup_iter} \
        --zero_padding \
        --DynamicGS_dataset

    # python metrics_plot.py -m ${output}/${scene}_controlNode_${exp} --iter_basis 20 --iter_gs 100 --noStatic_mask
done