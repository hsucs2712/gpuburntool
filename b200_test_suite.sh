#!/bin/bash
#
# B200 GPU Burn Test Suite
#

GPU_BURN="./gpu_burn"
MONITOR="python3 gpu_burn_monitor.py"
DURATION=300

echo "========================================"
echo "  B200 GPU Burn Test Suite"
echo "========================================"

# ========================================
# FP32 測試
# ========================================
echo ""
echo "[FP32 Tests]"
sudo nvidia-smi -pl 200 && $MONITOR -d $DURATION -pl 200 --no-tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 400 && $MONITOR -d $DURATION -pl 400 --no-tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 600 && $MONITOR -d $DURATION -pl 600 --no-tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 800 && $MONITOR -d $DURATION -pl 800 --no-tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 1000 && $MONITOR -d $DURATION -pl 1000 --no-tc --gpu-burn $GPU_BURN

# ========================================
# Tensor Core 測試
# ========================================
echo ""
echo "[Tensor Core Tests]"
sudo nvidia-smi -pl 200 && $MONITOR -d $DURATION -pl 200 -tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 400 && $MONITOR -d $DURATION -pl 400 -tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 600 && $MONITOR -d $DURATION -pl 600 -tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 800 && $MONITOR -d $DURATION -pl 800 -tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 1000 && $MONITOR -d $DURATION -pl 1000 -tc --gpu-burn $GPU_BURN

# ========================================
# FP64 測試
# ========================================
echo ""
echo "[FP64 Tests]"
sudo nvidia-smi -pl 200 && $MONITOR -d $DURATION -pl 200 --doubles --no-tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 400 && $MONITOR -d $DURATION -pl 400 --doubles --no-tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 600 && $MONITOR -d $DURATION -pl 600 --doubles --no-tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 800 && $MONITOR -d $DURATION -pl 800 --doubles --no-tc --gpu-burn $GPU_BURN
sudo nvidia-smi -pl 1000 && $MONITOR -d $DURATION -pl 1000 --doubles --no-tc --gpu-burn $GPU_BURN

# ========================================
# 圖表產生
# ========================================
echo ""
echo "[Generating Charts]"
for f in results/*.csv; do
    $MONITOR --chart "$f"
done

echo ""
echo "========================================"
echo "  All tests complete!"
echo "  Results in: ./results/"
echo "========================================"
