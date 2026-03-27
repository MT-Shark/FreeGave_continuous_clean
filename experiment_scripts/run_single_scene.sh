#!/bin/bash
# 单元执行脚本：处理单个场景的增量训练
# 用法: ./run_single_scene.sh <gpu_id> <scene_path> [output_base_dir] [control_num]
#
# 参数:
#   gpu_id         - CUDA 设备编号
#   scene_path     - 场景数据文件夹的完整路径
#   output_base_dir- 输出根目录 (默认: output/mini_subset_exp)
#   control_num    - control node 数量 (默认: 10000)

set -euo pipefail

# 解析参数
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <gpu_id> <scene_path> [output_base_dir] [control_num]"
    echo "Example: $0 0 /path/to/PhysicsBenchmark/.../round_name/scene_name"
    exit 1
fi

GPU_ID="$1"
SCENE_PATH="$2"
OUTPUT_BASE="${3:-output/mini_subset_exp}"
CONTROL_NUM="${4:-10000}"

# 获取场景名和父级目录名（用于分类）
SCENE_NAME=$(basename "$SCENE_PATH")
ROUND_NAME=$(basename "$(dirname "$SCENE_PATH")")

# 输出目录
OUTPUT_DIR="${OUTPUT_BASE}/${ROUND_NAME}/${SCENE_NAME}"

# 脚本所在目录（用于定位项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "Scene:       $SCENE_NAME"
echo "Scene Path:  $SCENE_PATH"
echo "Output Dir:  $OUTPUT_DIR"
echo "GPU:         $GPU_ID"
echo "Control Num: $CONTROL_NUM"
echo "=============================================="

# 切换到项目目录
cd "$PROJECT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ========== 训练阶段 ==========
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting incremental training..."

if CUDA_VISIBLE_DEVICES="$GPU_ID" python incremental_train_ControlNode.py \
    -s "$SCENE_PATH" \
    -m "$OUTPUT_DIR" \
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
    --control_num "$CONTROL_NUM" \
    --warmup_iter_basis 120 \
    --warmup_iter_gs 200 \
    --cn_interpolate_method dqb \
    --predict_steps 10 \
    --densify_grad_threshold 0.0002 \
    --size_threshold 20 \
    --static_warmup_iter 3000 \
    --zero_padding \
    --max_depth -1 \
    --skip_render; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Incremental training completed successfully."
else
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Incremental training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "=============================================="
echo "Scene $SCENE_NAME finished."
