#!/bin/bash
#
# B200 GPU Burn 完整測試腳本
#

GPU_BURN="./gpu_burn"
MONITOR="python3 gpu_burn_monitor.py"
DURATION=300
POWER_LIMITS=(200 400 600 800 1000)
OUTPUT_DIR="./results"

echo "========================================"
echo "  B200 GPU Burn Test Suite"
echo "========================================"
echo "Duration: ${DURATION}s per test"
echo "Power Limits: ${POWER_LIMITS[*]}"
echo ""

# 確認吸気温度
echo "[Step 1] 確認吸気温度"
ipmitool sensor list | grep -iE "inlet|ambient"
echo ""
read -p "Press Enter to continue..."

# Idle 測定
echo "[Step 2] Idle 測定"
$MONITOR -d 30 --no-tc -o $OUTPUT_DIR --gpu-burn $GPU_BURN 2>/dev/null || echo "Idle measurement skipped"
echo ""

# FP32 測試
echo "========================================"
echo "[Step 3] FP32 Tests (no tensor cores)"
echo "========================================"
for pl in "${POWER_LIMITS[@]}"; do
    echo ""
    echo ">>> FP32 PL=${pl}W"
    $MONITOR -d $DURATION -pl $pl --no-tc -o $OUTPUT_DIR --gpu-burn $GPU_BURN
    sleep 30  # 冷卻
done

echo ""
echo "FP32 tests complete. Check FAN optimal -> MAX now if needed."
read -p "Press Enter to continue to Tensor Core tests..."

# Tensor Core 測試
echo "========================================"
echo "[Step 4] Tensor Core Tests"
echo "========================================"
for pl in "${POWER_LIMITS[@]}"; do
    echo ""
    echo ">>> TC PL=${pl}W"
    $MONITOR -d $DURATION -pl $pl -tc -o $OUTPUT_DIR --gpu-burn $GPU_BURN
    sleep 30
done

echo ""
read -p "Run FP64 tests? (y/n) " run_fp64

if [[ "$run_fp64" == "y" ]]; then
    echo "========================================"
    echo "[Step 5] FP64 Tests (double precision)"
    echo "========================================"
    for pl in "${POWER_LIMITS[@]}"; do
        echo ""
        echo ">>> FP64 PL=${pl}W"
        $MONITOR -d $DURATION -pl $pl --doubles --no-tc -o $OUTPUT_DIR --gpu-burn $GPU_BURN
        sleep 30
    done
fi

# 最終吸気温度
echo ""
echo "[Final] 確認吸気温度"
ipmitool sensor list | grep -iE "inlet|ambient"

echo ""
echo "========================================"
echo "  All tests complete!"
echo "  Results in: $OUTPUT_DIR"
echo "========================================"
ls -la $OUTPUT_DIR/*.csv 2>/dev/null
