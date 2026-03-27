#!/bin/bash
# 终止当前项目的实验相关进程（仅限 FreeGave 项目）
# 用法: ./kill_all.sh

# 获取项目根目录路径（用于精确匹配）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

echo "=== Killing experiment processes for project: $PROJECT_NAME ==="
echo "Project path: $PROJECT_DIR"
echo ""

# 1. 终止主调度脚本（通过完整路径匹配）
pkill -f "${PROJECT_DIR}/experiment_scripts/run_multi_gpu.sh" 2>/dev/null && echo "Killed: run_multi_gpu.sh"
pkill -f "${PROJECT_DIR}/experiment_scripts/run_single_scene.sh" 2>/dev/null && echo "Killed: run_single_scene.sh"

# 2. 终止 Python 进程（通过日志路径匹配 tee 进程，找到同一进程组的 python）
# 日志路径是唯一的，可以精确定位本项目的进程
for log_file in "${SCRIPT_DIR}"/logs/worker_*.log; do
    if [[ -f "$log_file" ]]; then
        # 找到 tee 进程的 PID
        tee_pids=$(pgrep -f "tee -a $log_file" 2>/dev/null)
        for tee_pid in $tee_pids; do
            # 获取 tee 进程的父进程（bash）的进程组，并杀死整个进程组
            ppid=$(ps -o ppid= -p "$tee_pid" 2>/dev/null | tr -d ' ')
            if [[ -n "$ppid" && "$ppid" != "1" ]]; then
                # 杀死进程组（负号表示进程组）
                pgid=$(ps -o pgid= -p "$ppid" 2>/dev/null | tr -d ' ')
                if [[ -n "$pgid" ]]; then
                    kill -- -"$pgid" 2>/dev/null && echo "Killed process group: $pgid (from $log_file)"
                fi
            fi
        done
    fi
done

# 3. 备用方案：直接匹配当前项目的训练入口
pkill -f "python.*incremental_train_ControlNode.py" 2>/dev/null && echo "Killed: incremental_train_ControlNode.py"
pkill -f "python.*incremental_train_ControlNode_warmup.py" 2>/dev/null && echo "Killed: incremental_train_ControlNode_warmup.py"

echo ""
echo "=== Remaining processes for this project (if any) ==="
ps aux | grep -E "${PROJECT_DIR}|incremental_train_ControlNode|run_multi_gpu.sh|run_single_scene.sh" | grep -v "grep" | grep -v "kill_all.sh" || echo "None"

echo ""

# 自动清理：把 processing.txt 中的场景放回 incomplete.txt
PROCESSING_TXT="$SCRIPT_DIR/processing.txt"
INCOMPLETE_TXT="$SCRIPT_DIR/incomplete.txt"

if [[ -s "$PROCESSING_TXT" ]]; then
    echo "=== Recovering scenes from processing.txt ==="
    cat "$PROCESSING_TXT"
    cat "$PROCESSING_TXT" >> "$INCOMPLETE_TXT"
    > "$PROCESSING_TXT"
    echo ""
    echo "Moved above scenes back to incomplete.txt"
fi

echo "Done."
