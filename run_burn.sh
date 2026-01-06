#!/bin/bash
#
# GPU Burn Monitor - 快速執行腳本
# 用法: ./run_burn.sh [duration] [power_limit]
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION=${1:-60}
POWER_LIMIT=${2:-""}

echo "=========================================="
echo "  GPU Burn Monitor"
echo "=========================================="
echo "Duration: ${DURATION}s"
[ -n "$POWER_LIMIT" ] && echo "Power Limit: ${POWER_LIMIT}W"
echo ""

# 檢查依賴
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 not found"
        exit 1
    fi
}

check_command python3
check_command nvidia-smi
check_command ipmitool
check_command gpu-burn

# 確認 pandas 已安裝
python3 -c "import pandas" 2>/dev/null || {
    echo "Installing pandas..."
    pip3 install pandas --break-system-packages 2>/dev/null || pip3 install pandas
}

# 組合參數
ARGS="-d $DURATION -tc"
[ -n "$POWER_LIMIT" ] && ARGS="$ARGS -pl $POWER_LIMIT"

# 執行
cd "$SCRIPT_DIR"
python3 gpu_burn_monitor.py $ARGS

echo ""
echo "Done! Check ./results/ for output files."
