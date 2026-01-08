#!/usr/bin/env python3
"""
GPU Burn Monitor v2 - 監控 gpu_burn 測試並記錄系統數據
新增: 風扇轉速、系統溫度、記憶體使用率、DCMI整機功耗
輸出 CSV + 圖表
"""

import subprocess
import argparse
import csv
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, fields
import threading
import re

@dataclass
class SensorData:
    # 時間
    timestamp: str
    elapsed_seconds: float
    
    # GPU 資訊
    gpu_id: int
    gpu_name: str
    gpu_temp_c: float
    gpu_power_w: float
    gpu_fan_pct: float
    gpu_mem_used_mb: float
    gpu_mem_total_mb: float
    gpu_util_pct: float
    gpu_tflops: float
    
    # 系統記憶體
    sys_mem_total_gb: float
    sys_mem_used_gb: float
    sys_mem_used_pct: float
    
    # DCMI 整機功耗
    dcmi_power_w: float
    
    # 系統溫度 (IPMI)
    cpu1_temp_c: float
    cpu2_temp_c: float
    inlet_temp_c: float
    system_temp_c: float
    
    # HGX 溫度
    hgx_gpu_temp_c: float
    hgx_hbm_temp_c: float
    hgx_nvlink_temp_c: float
    hgx_inlet_temp_c: float
    
    # DIMM 溫度
    dimm_p1_temp_c: float
    dimm_p2_temp_c: float
    
    # 風扇轉速 (RPM) - 系統風扇平均
    fan_sys_avg_rpm: float
    fan_sys_min_rpm: float
    fan_sys_max_rpm: float
    
    # PSU 風扇平均
    fan_psu_avg_rpm: float
    
    # MB 風扇平均
    fan_mb_avg_rpm: float


class GPUBurnMonitor:
    def __init__(self, gpu_burn_path: str, duration: int, power_limit: int = None,
                 use_tc: bool = True, use_doubles: bool = False,
                 output_dir: str = "./results", interval: float = 1.0):
        
        self.gpu_burn_path = gpu_burn_path
        self.duration = duration
        self.power_limit = power_limit
        self.use_tc = use_tc
        self.use_doubles = use_doubles
        self.output_dir = Path(output_dir)
        self.interval = interval
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data: list = []
        self.process = None
        self.running = False
        self.start_time = None
        self.tflops: dict = {}
        
        # 快取 IPMI 資料 (減少呼叫頻率)
        self.ipmi_cache = {}
        self.ipmi_cache_time = 0
        self.ipmi_cache_interval = 2.0  # 每 2 秒更新一次 IPMI
        
    def get_gpu_count(self) -> int:
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10)
            return int(r.stdout.strip().split('\n')[0])
        except:
            return 1
    
    def get_gpu_data(self) -> list:
        try:
            r = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,temperature.gpu,power.draw,fan.speed,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            gpus = []
            for line in r.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                p = [x.strip() for x in line.split(',')]
                if len(p) >= 8:
                    gpus.append({
                        'id': int(p[0]),
                        'name': p[1],
                        'temp': float(p[2]) if p[2] not in ['[N/A]', 'N/A'] else 0,
                        'power': float(p[3]) if p[3] not in ['[N/A]', 'N/A'] else 0,
                        'fan': float(p[4]) if p[4] not in ['[N/A]', 'N/A'] else 0,
                        'mem_used': float(p[5]) if p[5] not in ['[N/A]', 'N/A'] else 0,
                        'mem_total': float(p[6]) if p[6] not in ['[N/A]', 'N/A'] else 0,
                        'util': float(p[7]) if p[7] not in ['[N/A]', 'N/A'] else 0,
                    })
            return gpus
        except Exception as e:
            print(f"nvidia-smi error: {e}")
            return []
    
    def get_system_memory(self) -> dict:
        """取得系統記憶體使用率"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1])  # KB
                        meminfo[key] = value
                
                total_kb = meminfo.get('MemTotal', 0)
                available_kb = meminfo.get('MemAvailable', 0)
                used_kb = total_kb - available_kb
                
                total_gb = total_kb / 1024 / 1024
                used_gb = used_kb / 1024 / 1024
                used_pct = (used_kb / total_kb * 100) if total_kb > 0 else 0
                
                return {
                    'total_gb': round(total_gb, 2),
                    'used_gb': round(used_gb, 2),
                    'used_pct': round(used_pct, 1)
                }
        except Exception as e:
            print(f"Memory read error: {e}")
            return {'total_gb': 0, 'used_gb': 0, 'used_pct': 0}
    
    def get_dcmi_power(self) -> float:
        """取得 DCMI 整機即時功耗"""
        try:
            r = subprocess.run(['sudo', 'ipmitool', 'dcmi', 'power', 'reading'],
                               capture_output=True, text=True, timeout=10)
            for line in r.stdout.split('\n'):
                if 'instantaneous' in line.lower():
                    match = re.search(r'(\d+)\s*[Ww]att', line)
                    if match:
                        return float(match.group(1))
        except Exception as e:
            pass
        return 0.0
    
    def get_all_ipmi_sensors(self) -> dict:
        """取得所有 IPMI sensor 並解析"""
        now = time.time()
        if now - self.ipmi_cache_time < self.ipmi_cache_interval and self.ipmi_cache:
            return self.ipmi_cache
        
        sensors = {
            'cpu1_temp': 0, 'cpu2_temp': 0,
            'inlet_temp': 0, 'system_temp': 0,
            'hgx_gpu_temp': 0, 'hgx_hbm_temp': 0,
            'hgx_nvlink_temp': 0, 'hgx_inlet_temp': 0,
            'dimm_p1_temp': 0, 'dimm_p2_temp': 0,
            'fan_sys': [], 'fan_psu': [], 'fan_mb': []
        }
        
        try:
            r = subprocess.run(['sudo', 'ipmitool', 'sensor', 'list'],
                               capture_output=True, text=True, timeout=15)
            
            for line in r.stdout.split('\n'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 2:
                    continue
                
                name = parts[0].lower()
                try:
                    value = float(parts[1])
                except:
                    continue
                
                # 溫度
                if name == 'cpu1 temp':
                    sensors['cpu1_temp'] = value
                elif name == 'cpu2 temp':
                    sensors['cpu2_temp'] = value
                elif name == 'inlet temp':
                    sensors['inlet_temp'] = value
                elif name == 'system temp':
                    sensors['system_temp'] = value
                elif name == 'hgx gpu temp':
                    sensors['hgx_gpu_temp'] = value
                elif name == 'hgx hbm temp':
                    sensors['hgx_hbm_temp'] = value
                elif name == 'hgx nvlink temp':
                    sensors['hgx_nvlink_temp'] = value
                elif name == 'hgx inlet temp':
                    sensors['hgx_inlet_temp'] = value
                elif 'p1_dimm' in name and 'temp' in name:
                    sensors['dimm_p1_temp'] = value
                elif 'p2_dimm' in name and 'temp' in name:
                    sensors['dimm_p2_temp'] = value
                
                # 風扇 RPM
                elif name.startswith('fan') and name[3:].isdigit():
                    # FAN1 ~ FAN15
                    sensors['fan_sys'].append(value)
                elif name.startswith('ps') and 'fan' in name:
                    # PS1 FAN ~ PS6 FAN
                    sensors['fan_psu'].append(value)
                elif name.startswith('mb fan'):
                    # MB FAN1 ~ MB FAN4
                    sensors['fan_mb'].append(value)
        
        except Exception as e:
            print(f"IPMI sensor error: {e}")
        
        self.ipmi_cache = sensors
        self.ipmi_cache_time = now
        return sensors
    
    def set_power_limit(self, limit: int):
        count = self.get_gpu_count()
        for i in range(count):
            try:
                subprocess.run(['sudo', 'nvidia-smi', '-i', str(i), '-pl', str(limit)],
                               capture_output=True, timeout=10)
                print(f"GPU {i}: Power limit set to {limit}W")
            except Exception as e:
                print(f"Warning: Could not set power limit for GPU {i}: {e}")
    
    def parse_tflops(self, line: str):
        # 格式: proc'd: 85 (18624 Gflop/s) - 43 (18937 Gflop/s) - ...
        if 'proc\'d:' in line or 'Gflop/s' in line or 'Tflop/s' in line:
            matches = re.findall(r'\((\d+\.?\d*)\s*(G|T)flop/s\)', line, re.IGNORECASE)
            for gpu_id, (val, unit) in enumerate(matches):
                val = float(val)
                if unit.upper() == 'G':
                    val /= 1000.0
                self.tflops[gpu_id] = val
    
    def read_output(self):
        while self.running and self.process and self.process.poll() is None:
            line = self.process.stdout.readline()
            if line:
                line = line.strip()
                self.parse_tflops(line)
                print(f"[gpu_burn] {line}")
    
    def collect(self) -> list:
        ts = datetime.now().isoformat()
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        gpus = self.get_gpu_data()
        mem = self.get_system_memory()
        dcmi = self.get_dcmi_power()
        ipmi = self.get_all_ipmi_sensors()
        
        # 計算風扇統計
        fan_sys = ipmi['fan_sys']
        fan_sys_avg = sum(fan_sys) / len(fan_sys) if fan_sys else 0
        fan_sys_min = min(fan_sys) if fan_sys else 0
        fan_sys_max = max(fan_sys) if fan_sys else 0
        
        fan_psu = ipmi['fan_psu']
        fan_psu_avg = sum(fan_psu) / len(fan_psu) if fan_psu else 0
        
        fan_mb = ipmi['fan_mb']
        fan_mb_avg = sum(fan_mb) / len(fan_mb) if fan_mb else 0
        
        samples = []
        for gpu in gpus:
            s = SensorData(
                timestamp=ts,
                elapsed_seconds=round(elapsed, 2),
                gpu_id=gpu['id'],
                gpu_name=gpu['name'],
                gpu_temp_c=gpu['temp'],
                gpu_power_w=gpu['power'],
                gpu_fan_pct=gpu['fan'],
                gpu_mem_used_mb=gpu['mem_used'],
                gpu_mem_total_mb=gpu['mem_total'],
                gpu_util_pct=gpu['util'],
                gpu_tflops=round(self.tflops.get(gpu['id'], 0), 3),
                
                sys_mem_total_gb=mem['total_gb'],
                sys_mem_used_gb=mem['used_gb'],
                sys_mem_used_pct=mem['used_pct'],
                
                dcmi_power_w=dcmi,
                
                cpu1_temp_c=ipmi['cpu1_temp'],
                cpu2_temp_c=ipmi['cpu2_temp'],
                inlet_temp_c=ipmi['inlet_temp'],
                system_temp_c=ipmi['system_temp'],
                
                hgx_gpu_temp_c=ipmi['hgx_gpu_temp'],
                hgx_hbm_temp_c=ipmi['hgx_hbm_temp'],
                hgx_nvlink_temp_c=ipmi['hgx_nvlink_temp'],
                hgx_inlet_temp_c=ipmi['hgx_inlet_temp'],
                
                dimm_p1_temp_c=ipmi['dimm_p1_temp'],
                dimm_p2_temp_c=ipmi['dimm_p2_temp'],
                
                fan_sys_avg_rpm=round(fan_sys_avg, 0),
                fan_sys_min_rpm=round(fan_sys_min, 0),
                fan_sys_max_rpm=round(fan_sys_max, 0),
                fan_psu_avg_rpm=round(fan_psu_avg, 0),
                fan_mb_avg_rpm=round(fan_mb_avg, 0),
            )
            samples.append(s)
        return samples
    
    def run(self) -> Path:
        if self.power_limit:
            self.set_power_limit(self.power_limit)
        
        # 檔名
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = ""
        if self.power_limit:
            suffix += f"_pl{self.power_limit}w"
        if self.use_tc:
            suffix += "_tc"
        if self.use_doubles:
            suffix += "_fp64"
        
        csv_path = self.output_dir / f"gpu_burn_{ts}{suffix}.csv"
        
        # 組合命令
        cmd = [self.gpu_burn_path]
        if self.use_tc:
            cmd.append('-tc')
        if self.use_doubles:
            cmd.append('-d')
        cmd.append(str(self.duration))
        
        print(f"Starting: {' '.join(cmd)}")
        print(f"Output: {csv_path}")
        print("-" * 60)
        
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1)
        self.running = True
        self.start_time = time.time()
        
        reader = threading.Thread(target=self.read_output)
        reader.start()
        
        try:
            while self.process.poll() is None:
                samples = self.collect()
                self.data.extend(samples)
                
                if samples:
                    s = samples[0]
                    print(f"\r[{s.elapsed_seconds:6.1f}s] GPU0: {s.gpu_temp_c:5.1f}°C {s.gpu_power_w:6.1f}W {s.gpu_tflops:6.2f}TF | "
                          f"DCMI:{s.dcmi_power_w:5.0f}W | Inlet:{s.inlet_temp_c:4.1f}°C | "
                          f"SysFan:{s.fan_sys_avg_rpm:5.0f}RPM | Mem:{s.sys_mem_used_pct:4.1f}%", end="", flush=True)
                
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\nInterrupted")
            if self.process:
                self.process.terminate()
        finally:
            self.running = False
            reader.join(timeout=5)
        
        # 儲存 CSV
        self.save_csv(csv_path)
        print(f"\n\nSaved: {csv_path}")
        
        return csv_path
    
    def save_csv(self, path: Path):
        if not self.data:
            return
        field_names = [f.name for f in fields(self.data[0])]
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=field_names)
            w.writeheader()
            for r in self.data:
                w.writerow(asdict(r))


def generate_charts(csv_path: str):
    """從 CSV 產生 PNG 圖表"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("請安裝: pip install pandas matplotlib")
        return
    
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    gpu_ids = sorted(df['gpu_id'].unique())
    colors = plt.cm.tab10.colors
    
    # 建立圖表目錄
    chart_dir = csv_path.parent / f"{csv_path.stem}_charts"
    chart_dir.mkdir(exist_ok=True)
    
    # 取系統層級資料 (只取一筆 GPU 的資料避免重複)
    sdf = df[df['gpu_id'] == gpu_ids[0]].copy()
    
    # 1. GPU 溫度
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_temp_c'], label=f'GPU {gid}', color=colors[i % 10])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('GPU Temperature')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '01_gpu_temp.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. GPU 功耗
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_power_w'], label=f'GPU {gid}', color=colors[i % 10])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title('GPU Power')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '02_gpu_power.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. GPU TFLOPS
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_tflops'], label=f'GPU {gid}', color=colors[i % 10])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('TFLOPS')
    ax.set_title('GPU Performance')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '03_gpu_tflops.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. DCMI 整機功耗
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sdf['elapsed_seconds'], sdf['dcmi_power_w'], color='red', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title('System Total Power (DCMI)')
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '04_dcmi_power.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. 系統溫度 (CPU, Inlet, System, HGX)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sdf['elapsed_seconds'], sdf['cpu1_temp_c'], label='CPU1', linewidth=1.5)
    ax.plot(sdf['elapsed_seconds'], sdf['cpu2_temp_c'], label='CPU2', linewidth=1.5)
    ax.plot(sdf['elapsed_seconds'], sdf['inlet_temp_c'], label='Inlet', linewidth=1.5)
    ax.plot(sdf['elapsed_seconds'], sdf['system_temp_c'], label='System', linewidth=1.5)
    ax.plot(sdf['elapsed_seconds'], sdf['hgx_inlet_temp_c'], label='HGX Inlet', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('System Temperature')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '05_sys_temp.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. HGX 溫度
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sdf['elapsed_seconds'], sdf['hgx_gpu_temp_c'], label='HGX GPU', linewidth=1.5)
    ax.plot(sdf['elapsed_seconds'], sdf['hgx_hbm_temp_c'], label='HGX HBM', linewidth=1.5)
    ax.plot(sdf['elapsed_seconds'], sdf['hgx_nvlink_temp_c'], label='HGX NVLink', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('HGX Temperature')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '06_hgx_temp.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 7. 風扇轉速
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sdf['elapsed_seconds'], sdf['fan_sys_avg_rpm'], label='System Fan (avg)', linewidth=1.5)
    ax.fill_between(sdf['elapsed_seconds'], sdf['fan_sys_min_rpm'], sdf['fan_sys_max_rpm'], 
                    alpha=0.3, label='System Fan (min-max)')
    ax.plot(sdf['elapsed_seconds'], sdf['fan_psu_avg_rpm'], label='PSU Fan (avg)', linewidth=1.5)
    ax.plot(sdf['elapsed_seconds'], sdf['fan_mb_avg_rpm'], label='MB Fan (avg)', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RPM')
    ax.set_title('Fan Speed')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '07_fan_speed.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. 系統記憶體使用率
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sdf['elapsed_seconds'], sdf['sys_mem_used_pct'], color='purple', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Memory Usage (%)')
    ax.set_title(f'System Memory Usage (Total: {sdf["sys_mem_total_gb"].iloc[0]:.0f} GB)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '08_sys_memory.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 9. DIMM 溫度
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sdf['elapsed_seconds'], sdf['dimm_p1_temp_c'], label='P1 DIMM', linewidth=1.5)
    ax.plot(sdf['elapsed_seconds'], sdf['dimm_p2_temp_c'], label='P2 DIMM', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('DIMM Temperature')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '09_dimm_temp.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 10. 總覽圖
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # GPU Temp
    ax = axes[0, 0]
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_temp_c'], color=colors[i % 10], linewidth=0.6)
    ax.set_title('GPU Temperature (°C)')
    ax.grid(True, alpha=0.3)
    
    # GPU Power
    ax = axes[0, 1]
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_power_w'], color=colors[i % 10], linewidth=0.6)
    ax.set_title('GPU Power (W)')
    ax.grid(True, alpha=0.3)
    
    # GPU TFLOPS
    ax = axes[0, 2]
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_tflops'], color=colors[i % 10], linewidth=0.6)
    ax.set_title('GPU TFLOPS')
    ax.grid(True, alpha=0.3)
    
    # DCMI Power
    ax = axes[1, 0]
    ax.plot(sdf['elapsed_seconds'], sdf['dcmi_power_w'], color='red', linewidth=1)
    ax.set_title('System Power - DCMI (W)')
    ax.grid(True, alpha=0.3)
    
    # System Temp
    ax = axes[1, 1]
    ax.plot(sdf['elapsed_seconds'], sdf['cpu1_temp_c'], label='CPU1', linewidth=1)
    ax.plot(sdf['elapsed_seconds'], sdf['inlet_temp_c'], label='Inlet', linewidth=1)
    ax.plot(sdf['elapsed_seconds'], sdf['system_temp_c'], label='System', linewidth=1)
    ax.set_title('System Temp (°C)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # HGX Temp
    ax = axes[1, 2]
    ax.plot(sdf['elapsed_seconds'], sdf['hgx_gpu_temp_c'], label='GPU', linewidth=1)
    ax.plot(sdf['elapsed_seconds'], sdf['hgx_hbm_temp_c'], label='HBM', linewidth=1)
    ax.plot(sdf['elapsed_seconds'], sdf['hgx_nvlink_temp_c'], label='NVLink', linewidth=1)
    ax.set_title('HGX Temp (°C)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Fan Speed
    ax = axes[2, 0]
    ax.plot(sdf['elapsed_seconds'], sdf['fan_sys_avg_rpm'], label='Sys', linewidth=1)
    ax.plot(sdf['elapsed_seconds'], sdf['fan_psu_avg_rpm'], label='PSU', linewidth=1)
    ax.plot(sdf['elapsed_seconds'], sdf['fan_mb_avg_rpm'], label='MB', linewidth=1)
    ax.set_title('Fan Speed (RPM)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Memory Usage
    ax = axes[2, 1]
    ax.plot(sdf['elapsed_seconds'], sdf['sys_mem_used_pct'], color='purple', linewidth=1)
    ax.set_title('System Memory (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # DIMM Temp
    ax = axes[2, 2]
    ax.plot(sdf['elapsed_seconds'], sdf['dimm_p1_temp_c'], label='P1', linewidth=1)
    ax.plot(sdf['elapsed_seconds'], sdf['dimm_p2_temp_c'], label='P2', linewidth=1)
    ax.set_title('DIMM Temp (°C)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'GPU Burn Test Overview: {csv_path.stem}', fontsize=14)
    plt.tight_layout()
    fig.savefig(chart_dir / '00_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved to: {chart_dir}/")
    
    # 印出統計
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    print(f"  Duration:          {df['elapsed_seconds'].max():.0f}s")
    print(f"  GPU Count:         {len(gpu_ids)}")
    print("-" * 60)
    print("  GPU:")
    print(f"    Max Temp:        {df['gpu_temp_c'].max():.1f}°C")
    print(f"    Avg Temp:        {df['gpu_temp_c'].mean():.1f}°C")
    print(f"    Max Power:       {df['gpu_power_w'].max():.1f}W (per GPU)")
    print(f"    Avg Power:       {df['gpu_power_w'].mean():.1f}W (per GPU)")
    total_gpu_power = df.groupby('elapsed_seconds')['gpu_power_w'].sum()
    print(f"    Total GPU Power: {total_gpu_power.mean():.0f}W (avg), {total_gpu_power.max():.0f}W (max)")
    if (df['gpu_tflops'] > 0).any():
        print(f"    Max TFLOPS:      {df['gpu_tflops'].max():.2f}")
        print(f"    Avg TFLOPS:      {df[df['gpu_tflops'] > 0]['gpu_tflops'].mean():.2f}")
    print("-" * 60)
    print("  System:")
    print(f"    DCMI Power:      {sdf['dcmi_power_w'].mean():.0f}W (avg), {sdf['dcmi_power_w'].max():.0f}W (max)")
    print(f"    CPU1 Temp:       {sdf['cpu1_temp_c'].mean():.1f}°C (avg), {sdf['cpu1_temp_c'].max():.1f}°C (max)")
    print(f"    Inlet Temp:      {sdf['inlet_temp_c'].mean():.1f}°C (avg)")
    print(f"    Memory Usage:    {sdf['sys_mem_used_pct'].mean():.1f}% (avg)")
    print("-" * 60)
    print("  HGX:")
    print(f"    GPU Temp:        {sdf['hgx_gpu_temp_c'].max():.1f}°C (max)")
    print(f"    HBM Temp:        {sdf['hgx_hbm_temp_c'].max():.1f}°C (max)")
    print(f"    NVLink Temp:     {sdf['hgx_nvlink_temp_c'].max():.1f}°C (max)")
    print("-" * 60)
    print("  Fans:")
    print(f"    System Fan:      {sdf['fan_sys_avg_rpm'].mean():.0f} RPM (avg)")
    print(f"    PSU Fan:         {sdf['fan_psu_avg_rpm'].mean():.0f} RPM (avg)")
    print(f"    MB Fan:          {sdf['fan_mb_avg_rpm'].mean():.0f} RPM (avg)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='GPU Burn Monitor v2 - with Fan, Temperature, Memory monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例:
  %(prog)s -d 300 -pl 400              # 300秒, 400W power limit
  %(prog)s -d 60 --no-tc               # 60秒, 不使用 tensor cores  
  %(prog)s --chart results/xxx.csv     # 從 CSV 產生圖表
        '''
    )
    
    parser.add_argument('-d', '--duration', type=int, default=60, help='測試時間 (秒)')
    parser.add_argument('-pl', '--power-limit', type=int, help='GPU Power Limit (W)')
    parser.add_argument('-tc', '--tensor-cores', action='store_true', default=True, help='使用 Tensor Cores')
    parser.add_argument('--no-tc', action='store_true', help='不使用 Tensor Cores')
    parser.add_argument('--doubles', action='store_true', help='使用 double precision')
    parser.add_argument('-o', '--output', default='./results', help='輸出目錄')
    parser.add_argument('-i', '--interval', type=float, default=1.0, help='取樣間隔 (秒)')
    parser.add_argument('--gpu-burn', default='./gpu_burn', help='gpu_burn 路徑')
    parser.add_argument('--chart', type=str, help='從 CSV 產生圖表 (不執行測試)')
    
    args = parser.parse_args()
    
    # 只產生圖表
    if args.chart:
        generate_charts(args.chart)
        return
    
    # 檢查 gpu_burn
    if not os.path.isfile(args.gpu_burn):
        print(f"Error: gpu_burn not found: {args.gpu_burn}")
        print("Use --gpu-burn to specify path")
        sys.exit(1)
    
    use_tc = args.tensor_cores and not args.no_tc
    
    monitor = GPUBurnMonitor(
        gpu_burn_path=args.gpu_burn,
        duration=args.duration,
        power_limit=args.power_limit,
        use_tc=use_tc,
        use_doubles=args.doubles,
        output_dir=args.output,
        interval=args.interval
    )
    
    csv_path = monitor.run()
    
    # 產生圖表
    print("\nGenerating charts...")
    generate_charts(csv_path)


if __name__ == '__main__':
    main()
