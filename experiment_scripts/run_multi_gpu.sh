#!/bin/bash
# 多进程调度脚本：并行处理多个场景
# 用法: ./run_multi_gpu.sh [config_dir] [control_num]
#
# 配置:
#   修改下面的 GPU_LIST 指定可用的 GPU
#   config_dir 包含 incomplete.txt, complete.txt, processing.txt, logs/

set -euo pipefail

# ===================== 配置区 =====================
# 指定可用的 GPU 列表（修改为你的显卡 id）
GPU_LIST=(0 1 2 3 4 5 6 7 8 9)

# 输出根目录（训练结果保存位置）
OUTPUT_BASE="output/mini_subset_exp"

# 默认 control node 数量
DEFAULT_CONTROL_NUM=7000
# ==================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${1:-$SCRIPT_DIR}"
CONTROL_NUM="${2:-$DEFAULT_CONTROL_NUM}"

# 文件路径
INCOMPLETE_TXT="$CONFIG_DIR/incomplete.txt"
COMPLETE_TXT="$CONFIG_DIR/complete.txt"
PROCESSING_TXT="$CONFIG_DIR/processing.txt"
LOG_DIR="$CONFIG_DIR/logs"
LOCK_FILE="$CONFIG_DIR/.task_lock"

# 检查必要文件
if [[ ! -f "$INCOMPLETE_TXT" ]]; then
    echo "[ERROR] incomplete.txt not found. Run init_scene_list.sh first."
    exit 1
fi

mkdir -p "$LOG_DIR"

NUM_GPUS=${#GPU_LIST[@]}
if [[ $NUM_GPUS -eq 0 ]]; then
    echo "[ERROR] GPU_LIST is empty."
    exit 1
fi

echo "=== Multi-GPU Experiment Runner ==="
echo "GPUs:         ${GPU_LIST[*]}"
echo "Config Dir:   $CONFIG_DIR"
echo "Output Base:  $OUTPUT_BASE"
echo "Control Num:  $CONTROL_NUM"
echo ""

# 获取任务的函数（使用文件锁保证原子操作）
# 返回：成功时输出场景路径，失败时返回非零
acquire_task() {
    local fd=200
    local scene=""
    
    # 使用 flock 获取独占锁
    (
        flock -x $fd
        
        # 读取 incomplete.txt 的第一行
        if [[ ! -s "$INCOMPLETE_TXT" ]]; then
            exit 1  # 没有更多任务
        fi
        
        scene=$(head -n 1 "$INCOMPLETE_TXT")
        
        if [[ -z "$scene" ]]; then
            exit 1
        fi
        
        # 从 incomplete.txt 移除第一行
        tail -n +2 "$INCOMPLETE_TXT" > "$INCOMPLETE_TXT.tmp" && mv "$INCOMPLETE_TXT.tmp" "$INCOMPLETE_TXT"
        
        # 添加到 processing.txt
        echo "$scene" >> "$PROCESSING_TXT"
        
        # 输出场景路径
        echo "$scene"
        
    ) 200>"$LOCK_FILE"
}

# 标记任务完成
mark_complete() {
    local scene="$1"
    local fd=200
    
    (
        flock -x $fd
        
        # 从 processing.txt 移除
        grep -v -F "$scene" "$PROCESSING_TXT" > "$PROCESSING_TXT.tmp" 2>/dev/null || true
        mv "$PROCESSING_TXT.tmp" "$PROCESSING_TXT"
        
        # 添加到 complete.txt
        echo "$scene" >> "$COMPLETE_TXT"
        
    ) 200>"$LOCK_FILE"
}

# 标记任务失败（重新放回 incomplete）
mark_failed() {
    local scene="$1"
    local fd=200
    
    (
        flock -x $fd
        
        # 从 processing.txt 移除
        grep -v -F "$scene" "$PROCESSING_TXT" > "$PROCESSING_TXT.tmp" 2>/dev/null || true
        mv "$PROCESSING_TXT.tmp" "$PROCESSING_TXT"
        
        # 重新添加到 incomplete.txt 末尾（可选：改为添加到 failed.txt）
        echo "$scene" >> "$CONFIG_DIR/failed.txt"
        
    ) 200>"$LOCK_FILE"
}

# Worker 函数：一个 worker 对应一个 GPU，持续获取并执行任务
worker() {
    local worker_id="$1"
    local gpu_id="$2"
    local log_file="$LOG_DIR/worker_${worker_id}_gpu${gpu_id}.log"
    
    echo "[Worker $worker_id] Started on GPU $gpu_id. Log: $log_file"
    
    # 重定向输出到日志文件
    exec > >(tee -a "$log_file") 2>&1
    
    echo "=============================================="
    echo "[Worker $worker_id] Started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[Worker $worker_id] GPU: $gpu_id"
    echo "=============================================="
    
    while true; do
        # 尝试获取任务
        scene=$(acquire_task) || {
            echo "[Worker $worker_id] No more tasks. Exiting."
            break
        }
        
        if [[ -z "$scene" ]]; then
            echo "[Worker $worker_id] Empty scene path. Exiting."
            break
        fi
        
        scene_name=$(basename "$scene")
        
        echo ""
        echo "[Worker $worker_id] [$(date '+%Y-%m-%d %H:%M:%S')] ========== START: $scene_name =========="
        echo "[Worker $worker_id] Scene Path: $scene"
        
        # 执行单元脚本
        if "$SCRIPT_DIR/run_single_scene.sh" "$gpu_id" "$scene" "$OUTPUT_BASE" "$CONTROL_NUM"; then
            echo "[Worker $worker_id] [$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $scene_name"
            mark_complete "$scene"
        else
            echo "[Worker $worker_id] [$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $scene_name"
            mark_failed "$scene"
        fi
        
        echo "[Worker $worker_id] [$(date '+%Y-%m-%d %H:%M:%S')] ========== END: $scene_name =========="
        echo ""
    done
    
    echo "[Worker $worker_id] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

# 主逻辑：启动多个 worker
echo "Starting ${NUM_GPUS} workers..."
echo ""

# 清空主日志
MAIN_LOG="$LOG_DIR/main.log"
> "$MAIN_LOG"

{
    echo "=== Experiment Started at $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "GPUs: ${GPU_LIST[*]}"
    echo "Control Num: $CONTROL_NUM"
    echo "Total incomplete scenes: $(wc -l < "$INCOMPLETE_TXT")"
    echo ""
} >> "$MAIN_LOG"

# 启动所有 workers
pids=()
for i in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$i]}"
    worker "$i" "$gpu" &
    pids+=($!)
    sleep 0.5  # 稍微错开启动时间
done

echo "All workers started. PIDs: ${pids[*]}"
echo "Waiting for completion..."

# 等待所有 workers 完成
for pid in "${pids[@]}"; do
    wait "$pid" || true
done

{
    echo ""
    echo "=== Experiment Finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "Completed scenes: $(wc -l < "$COMPLETE_TXT")"
    echo "Failed scenes: $(wc -l < "$CONFIG_DIR/failed.txt" 2>/dev/null || echo 0)"
    echo "Remaining incomplete: $(wc -l < "$INCOMPLETE_TXT")"
} >> "$MAIN_LOG"

echo ""
echo "=== All workers finished ==="
echo "Results:"
echo "  Completed: $(wc -l < "$COMPLETE_TXT")"
echo "  Failed:    $(wc -l < "$CONFIG_DIR/failed.txt" 2>/dev/null || echo 0)"
echo "  Remaining: $(wc -l < "$INCOMPLETE_TXT")"
echo ""
echo "Logs saved to: $LOG_DIR"
