#!/bin/bash
# 查看实验进度状态
# 用法: ./status.sh [config_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${1:-$SCRIPT_DIR}"

echo "=== Experiment Status ==="
echo "Config Dir: $CONFIG_DIR"
echo ""

count_lines() {
    local file="$1"
    if [[ -f "$file" ]]; then
        wc -l < "$file"
    else
        echo "0"
    fi
}

total=$(count_lines "$CONFIG_DIR/all_scenes.txt")
incomplete=$(count_lines "$CONFIG_DIR/incomplete.txt")
complete=$(count_lines "$CONFIG_DIR/complete.txt")
processing=$(count_lines "$CONFIG_DIR/processing.txt")
failed=$(count_lines "$CONFIG_DIR/failed.txt")

echo "Total scenes:     $total"
echo "───────────────────────"
echo "Incomplete:       $incomplete"
echo "Processing:       $processing"
echo "Complete:         $complete"
echo "Failed:           $failed"
echo ""

if [[ $total -gt 0 ]]; then
    progress=$((complete * 100 / total))
    echo "Progress: $complete / $total ($progress%)"
fi

echo ""

# 显示当前正在处理的场景
if [[ -s "$CONFIG_DIR/processing.txt" ]]; then
    echo "Currently processing:"
    while IFS= read -r line; do
        echo "  - $(basename "$line")"
    done < "$CONFIG_DIR/processing.txt"
    echo ""
fi

# 显示最近完成的场景（最后5个）
if [[ -s "$CONFIG_DIR/complete.txt" ]]; then
    echo "Recently completed (last 5):"
    tail -n 5 "$CONFIG_DIR/complete.txt" | while IFS= read -r line; do
        echo "  - $(basename "$line")"
    done
    echo ""
fi

# 显示失败的场景
if [[ -s "$CONFIG_DIR/failed.txt" ]]; then
    echo "Failed scenes:"
    while IFS= read -r line; do
        echo "  - $(basename "$line")"
    done < "$CONFIG_DIR/failed.txt"
fi
