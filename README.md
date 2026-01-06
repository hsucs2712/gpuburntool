# GPU Burn Monitor

監控 GPU burn 測試並記錄完整的系統數據，產生 CSV 和互動式圖表。

## 功能

- 🔥 執行 gpu-burn 壓力測試
- 📊 即時監控 GPU 溫度、功耗、TFLOPS
- 🌡️ 透過 IPMI 記錄系統溫度（Inlet/Exhaust）
- ⚡ 記錄 PSU、CPU、Memory 功耗
- 💨 監控風扇轉速
- 📈 產生互動式 HTML 圖表
- 📁 輸出 CSV 方便後續分析

## 系統需求

- Python 3.8+
- nvidia-smi (NVIDIA Driver)
- ipmitool
- gpu-burn
- pandas

```bash
# 安裝 Python 依賴
pip install pandas

# 確認工具可用
nvidia-smi
ipmitool sensor list
gpu-burn --help
```

## 使用方式

### 基本用法

```bash
# 60秒測試，使用 tensor cores
python gpu_burn_monitor.py -d 60 -tc

# 300秒測試，設定 400W power limit
python gpu_burn_monitor.py -d 300 -pl 400 -tc

# 120秒測試，不使用 tensor cores
python gpu_burn_monitor.py -d 120 --no-tc

# 使用 double precision
python gpu_burn_monitor.py -d 60 --doubles
```

### 完整參數

```
-d, --duration      測試時間（秒），預設 60
-pl, --power-limit  GPU Power Limit（瓦特）
-tc, --tensor-cores 使用 Tensor Cores（預設啟用）
--no-tc             不使用 Tensor Cores
--doubles           使用 double precision
-o, --output        輸出目錄，預設 ./results
-i, --interval      取樣間隔（秒），預設 1.0
--gpu-burn-path     gpu-burn 執行檔路徑
```

### 範例

```bash
# 多組 power limit 測試
for pl in 200 300 400; do
    python gpu_burn_monitor.py -d 300 -pl $pl -tc
done

# 長時間穩定性測試
python gpu_burn_monitor.py -d 3600 -pl 400 -tc -i 5
```

## 輸出檔案

測試完成後會在 `./results/` 目錄產生：

- `gpu_burn_YYYYMMDD_HHMMSS_pl400w_tc.csv` - 原始數據
- `gpu_burn_YYYYMMDD_HHMMSS_pl400w_tc.html` - 互動式圖表

### CSV 欄位說明

| 欄位 | 說明 |
|------|------|
| timestamp | ISO 格式時間戳 |
| elapsed_seconds | 測試經過時間（秒）|
| gpu_id | GPU 編號 |
| gpu_name | GPU 型號 |
| gpu_temp_c | GPU 溫度（°C）|
| gpu_power_w | GPU 功耗（W）|
| gpu_fan_speed_pct | GPU 風扇轉速（%）|
| gpu_memory_used_mb | GPU 記憶體使用量（MB）|
| gpu_memory_total_mb | GPU 記憶體總量（MB）|
| gpu_utilization_pct | GPU 使用率（%）|
| gpu_tflops | 運算效能（TFLOPS）|
| inlet_temp_c | 進風口溫度（°C）|
| exhaust_temp_c | 出風口溫度（°C）|
| cpu_temp_c | CPU 溫度（°C）|
| total_fan_power_w | 風扇總功耗（W）|
| total_psu_power_w | PSU 總功耗（W）|
| cpu_power_w | CPU 功耗（W）|
| memory_power_w | 記憶體功耗（W）|
| fan_speeds_rpm | 各風扇轉速（JSON）|

## 單獨產生圖表

```bash
python generate_charts.py results/gpu_burn_xxx.csv

# 指定輸出檔名
python generate_charts.py results/gpu_burn_xxx.csv my_report.html
```

## IPMI Sensor 名稱調整

如果你的伺服器 IPMI sensor 名稱不同，請修改 `gpu_burn_monitor.py` 中的 `self.ipmi_sensors` 字典：

```python
self.ipmi_sensors = {
    'inlet_temp': ['Inlet Temp', 'Ambient Temp', 'System Temp'],
    'exhaust_temp': ['Exhaust Temp', 'Outlet Temp'],
    # ... 依你的系統調整
}
```

查看你的 sensor 名稱：
```bash
ipmitool sensor list | grep -iE "temp|fan|power"
```

## License

MIT
