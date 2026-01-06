#!/usr/bin/env python3
"""
GPU Burn Monitor v2 - 監控 gpu_burn 測試並記錄系統數據
"""

import subprocess
import argparse
import csv
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from threading import Thread, Lock
import re
import select

@dataclass
class SensorData:
    timestamp: str
    elapsed_seconds: float
    gpu_id: int
    gpu_name: str
    gpu_temp_c: float
    gpu_power_w: float
    gpu_fan_pct: float
    gpu_mem_used_mb: float
    gpu_mem_total_mb: float
    gpu_util_pct: float
    gpu_tflops: float
    inlet_temp_c: float
    exhaust_temp_c: float
    cpu_temp_c: float
    fan_power_w: float
    psu_power_w: float
    cpu_power_w: float
    mem_power_w: float


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
        self.tflops_lock = Lock()
        self.gpu_count = 8  # 預設，會自動偵測
        
    def get_gpu_count(self) -> int:
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10)
            return int(r.stdout.strip().split('\n')[0])
        except:
            return 8
    
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
    
    def get_ipmi_sensor(self, patterns: list) -> float:
        try:
            r = subprocess.run(['ipmitool', 'sensor', 'list'],
                               capture_output=True, text=True, timeout=10)
            for line in r.stdout.split('\n'):
                for pat in patterns:
                    if pat.lower() in line.lower():
                        parts = line.split('|')
                        if len(parts) >= 2:
                            try:
                                val = float(parts[1].strip())
                                if val > 0:
                                    return val
                            except:
                                continue
        except:
            pass
        return 0.0
    
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
        """
        解析多種 gpu-burn 輸出格式:
        1. proc'd: 85 (18624 Gflop/s) - 43 (18937 Gflop/s) - ...
        2. GPU 0: xxxx Gflop/s
        3. 100.0%  proc'd: ...
        """
        # 格式 1: (數字 Gflop/s) 連續出現
        matches = re.findall(r'\((\d+\.?\d*)\s*(G|T)flop/s\)', line, re.IGNORECASE)
        if matches:
            with self.tflops_lock:
                for gpu_id, (val, unit) in enumerate(matches):
                    val = float(val)
                    if unit.upper() == 'G':
                        val /= 1000.0
                    self.tflops[gpu_id] = val
            return
        
        # 格式 2: GPU N: xxxx Gflop/s
        m = re.search(r'GPU\s*(\d+)[:\s]+(\d+\.?\d*)\s*(G|T)flop', line, re.IGNORECASE)
        if m:
            gpu_id = int(m.group(1))
            val = float(m.group(2))
            unit = m.group(3).upper()
            if unit == 'G':
                val /= 1000.0
            with self.tflops_lock:
                self.tflops[gpu_id] = val
    
    def read_output(self):
        """持續讀取 gpu_burn 輸出"""
        while self.running:
            if self.process is None or self.process.poll() is not None:
                break
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        self.parse_tflops(line)
                        # 只印重要訊息，避免畫面混亂
                        if 'error' in line.lower() or 'GPU' in line or 'proc\'d' in line.lower():
                            print(f"[gpu_burn] {line}")
            except:
                break
    
    def collect(self) -> list:
        ts = datetime.now().isoformat()
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        gpus = self.get_gpu_data()
        inlet = self.get_ipmi_sensor(['inlet', 'ambient', 'system temp'])
        exhaust = self.get_ipmi_sensor(['exhaust', 'outlet'])
        cpu_temp = self.get_ipmi_sensor(['cpu temp', 'cpu1 temp', 'cpu2 temp'])
        fan_pwr = self.get_ipmi_sensor(['fan pwr', 'fan power'])
        psu_pwr = self.get_ipmi_sensor(['psu', 'total power', 'pwr consumption', 'ps1 input', 'ps2 input'])
        cpu_pwr = self.get_ipmi_sensor(['cpu power', 'cpu1 power', 'cpu2 power'])
        mem_pwr = self.get_ipmi_sensor(['mem power', 'memory power', 'dimm power'])
        
        samples = []
        with self.tflops_lock:
            current_tflops = dict(self.tflops)
        
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
                gpu_tflops=round(current_tflops.get(gpu['id'], 0), 3),
                inlet_temp_c=inlet,
                exhaust_temp_c=exhaust,
                cpu_temp_c=cpu_temp,
                fan_power_w=fan_pwr,
                psu_power_w=psu_pwr,
                cpu_power_w=cpu_pwr,
                mem_power_w=mem_pwr,
            )
            samples.append(s)
        return samples
    
    def run(self) -> Path:
        self.gpu_count = self.get_gpu_count()
        print(f"Detected {self.gpu_count} GPUs")
        
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
        print("-" * 70)
        
        # 啟動 gpu_burn，合併 stdout 和 stderr
        self.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            bufsize=1,
            universal_newlines=True
        )
        self.running = True
        self.start_time = time.time()
        
        # 啟動讀取執行緒
        reader = Thread(target=self.read_output, daemon=True)
        reader.start()
        
        # 等待 gpu_burn 初始化
        time.sleep(3)
        
        try:
            while self.process.poll() is None:
                samples = self.collect()
                self.data.extend(samples)
                
                if samples:
                    s = samples[0]
                    # 計算總 TFLOPS
                    with self.tflops_lock:
                        total_tflops = sum(self.tflops.values())
                    
                    print(f"\r[{s.elapsed_seconds:6.1f}s] GPU0: {s.gpu_temp_c:5.1f}°C {s.gpu_power_w:6.1f}W | "
                          f"TFLOPS: {s.gpu_tflops:6.2f} (Total: {total_tflops:6.2f}) | "
                          f"Inlet:{s.inlet_temp_c:5.1f}°C PSU:{s.psu_power_w:6.0f}W", end="", flush=True)
                
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\nInterrupted")
            if self.process:
                self.process.terminate()
        finally:
            self.running = False
            reader.join(timeout=2)
        
        # 儲存 CSV
        self.save_csv(csv_path)
        print(f"\n\nSaved: {csv_path}")
        
        # 印出統計
        self.print_stats()
        
        return csv_path
    
    def save_csv(self, path: Path):
        if not self.data:
            return
        fields = list(asdict(self.data[0]).keys())
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.data:
                w.writerow(asdict(r))
    
    def print_stats(self):
        if not self.data:
            return
        
        temps = [d.gpu_temp_c for d in self.data]
        powers = [d.gpu_power_w for d in self.data]
        tflops_vals = [d.gpu_tflops for d in self.data if d.gpu_tflops > 0]
        
        print("\n" + "=" * 50)
        print("Statistics:")
        print("=" * 50)
        print(f"  Duration:       {self.data[-1].elapsed_seconds:.0f}s")
        print(f"  GPU Count:      {self.gpu_count}")
        print(f"  Max GPU Temp:   {max(temps):.1f}°C")
        print(f"  Avg GPU Temp:   {sum(temps)/len(temps):.1f}°C")
        print(f"  Max GPU Power:  {max(powers):.1f}W")
        print(f"  Avg GPU Power:  {sum(powers)/len(powers):.1f}W")
        if tflops_vals:
            print(f"  Max TFLOPS:     {max(tflops_vals):.2f}")
            print(f"  Avg TFLOPS:     {sum(tflops_vals)/len(tflops_vals):.2f}")
        print("=" * 50)


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
    
    chart_dir = csv_path.parent / f"{csv_path.stem}_charts"
    chart_dir.mkdir(exist_ok=True)
    
    # 1. GPU 溫度
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_temp_c'], label=f'GPU {gid}', color=colors[i % 10])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('GPU Temperature')
    ax.legend(loc='upper right')
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
    ax.legend(loc='upper right')
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
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.savefig(chart_dir / '03_gpu_tflops.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 總覽
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_temp_c'], color=colors[i % 10], linewidth=0.8)
    ax.set_title('GPU Temperature (°C)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_power_w'], color=colors[i % 10], linewidth=0.8)
    ax.set_title('GPU Power (W)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for i, gid in enumerate(gpu_ids):
        gdf = df[df['gpu_id'] == gid]
        ax.plot(gdf['elapsed_seconds'], gdf['gpu_tflops'], color=colors[i % 10], linewidth=0.8)
    ax.set_title('GPU TFLOPS')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    # 總功耗
    total_power = df.groupby('elapsed_seconds')['gpu_power_w'].sum()
    ax.plot(total_power.index, total_power.values, color='red', linewidth=1)
    ax.set_title('Total GPU Power (W)')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'GPU Burn Test: {csv_path.stem}', fontsize=14)
    plt.tight_layout()
    fig.savefig(chart_dir / '00_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved to: {chart_dir}/")


def main():
    parser = argparse.ArgumentParser(description='GPU Burn Monitor v2')
    
    parser.add_argument('-d', '--duration', type=int, default=60, help='測試時間 (秒)')
    parser.add_argument('-pl', '--power-limit', type=int, help='GPU Power Limit (W)')
    parser.add_argument('-tc', '--tensor-cores', action='store_true', default=True, help='使用 Tensor Cores')
    parser.add_argument('--no-tc', action='store_true', help='不使用 Tensor Cores')
    parser.add_argument('--doubles', action='store_true', help='使用 double precision')
    parser.add_argument('-o', '--output', default='./results', help='輸出目錄')
    parser.add_argument('-i', '--interval', type=float, default=1.0, help='取樣間隔 (秒)')
    parser.add_argument('--gpu-burn', default='./gpu_burn', help='gpu_burn 路徑')
    parser.add_argument('--chart', type=str, help='從 CSV 產生圖表')
    
    args = parser.parse_args()
    
    if args.chart:
        generate_charts(args.chart)
        return
    
    if not os.path.isfile(args.gpu_burn):
        print(f"Error: gpu_burn not found: {args.gpu_burn}")
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
    
    print("\nGenerating charts...")
    generate_charts(csv_path)


if __name__ == '__main__':
    main()
