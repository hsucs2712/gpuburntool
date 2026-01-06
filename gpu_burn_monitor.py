#!/usr/bin/env python3
"""
GPU Burn Monitor - 監控 GPU burn 測試並記錄系統數據
支援 NVIDIA GPU + Supermicro IPMI sensors
"""

import subprocess
import argparse
import csv
import json
import time
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading
import re

@dataclass
class SensorData:
    timestamp: str
    elapsed_seconds: float
    # GPU metrics (per GPU)
    gpu_id: int
    gpu_name: str
    gpu_temp_c: float
    gpu_power_w: float
    gpu_fan_speed_pct: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_utilization_pct: float
    gpu_tflops: float
    # System metrics (from IPMI)
    inlet_temp_c: Optional[float]
    exhaust_temp_c: Optional[float]
    cpu_temp_c: Optional[float]
    total_fan_power_w: Optional[float]
    total_psu_power_w: Optional[float]
    cpu_power_w: Optional[float]
    memory_power_w: Optional[float]
    # Fan speeds (RPM)
    fan_speeds_rpm: str  # JSON string of fan speeds

class GPUBurnMonitor:
    def __init__(self, 
                 gpu_burn_path: str = "gpu-burn",
                 duration: int = 60,
                 power_limit: Optional[int] = None,
                 use_tensor_cores: bool = True,
                 use_doubles: bool = False,
                 output_dir: str = "./results",
                 sample_interval: float = 1.0):
        
        self.gpu_burn_path = gpu_burn_path
        self.duration = duration
        self.power_limit = power_limit
        self.use_tensor_cores = use_tensor_cores
        self.use_doubles = use_doubles
        self.output_dir = Path(output_dir)
        self.sample_interval = sample_interval
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_records: List[SensorData] = []
        self.gpu_burn_process: Optional[subprocess.Popen] = None
        self.monitoring = False
        self.start_time: Optional[float] = None
        self.gpu_burn_output: List[str] = []
        self.current_tflops: Dict[int, float] = {}
        
        # IPMI sensor name mappings (adjust for your Supermicro system)
        self.ipmi_sensors = {
            'inlet_temp': ['Inlet Temp', 'Ambient Temp', 'System Temp'],
            'exhaust_temp': ['Exhaust Temp', 'Outlet Temp'],
            'cpu_temp': ['CPU Temp', 'CPU1 Temp', 'CPU2 Temp'],
            'fan_power': ['Fan Power', 'FAN PWR'],
            'psu_power': ['PSU1 PIN', 'PSU2 PIN', 'Total Power', 'PWR Consumption'],
            'cpu_power': ['CPU Power', 'CPU1 Power', 'CPU2 Power'],
            'memory_power': ['MEM Power', 'Memory Power', 'DIMM Power'],
        }
        
    def get_gpu_count(self) -> int:
        """取得 GPU 數量"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            return int(result.stdout.strip().split('\n')[0])
        except Exception as e:
            print(f"Warning: Could not get GPU count: {e}")
            return 1
    
    def get_nvidia_smi_data(self) -> List[Dict]:
        """從 nvidia-smi 取得 GPU 數據"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,temperature.gpu,power.draw,fan.speed,'
                'memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'temp': float(parts[2]) if parts[2] != '[N/A]' else 0,
                        'power': float(parts[3]) if parts[3] != '[N/A]' else 0,
                        'fan': float(parts[4]) if parts[4] != '[N/A]' else 0,
                        'mem_used': float(parts[5]) if parts[5] != '[N/A]' else 0,
                        'mem_total': float(parts[6]) if parts[6] != '[N/A]' else 0,
                        'util': float(parts[7]) if parts[7] != '[N/A]' else 0,
                    })
            return gpus
        except Exception as e:
            print(f"Warning: nvidia-smi failed: {e}")
            return []
    
    def get_ipmi_sensor(self, sensor_names: List[str]) -> Optional[float]:
        """從 IPMI 取得 sensor 數值"""
        try:
            result = subprocess.run(
                ['ipmitool', 'sensor', 'list'],
                capture_output=True, text=True, timeout=10
            )
            
            total = 0.0
            count = 0
            
            for line in result.stdout.split('\n'):
                for name in sensor_names:
                    if name.lower() in line.lower():
                        parts = line.split('|')
                        if len(parts) >= 2:
                            value_str = parts[1].strip()
                            try:
                                value = float(value_str)
                                if value > 0:  # 排除無效值
                                    total += value
                                    count += 1
                            except ValueError:
                                continue
            
            return total if count > 0 else None
        except Exception as e:
            return None
    
    def get_fan_speeds(self) -> Dict[str, float]:
        """取得所有風扇轉速"""
        fans = {}
        try:
            result = subprocess.run(
                ['ipmitool', 'sensor', 'list'],
                capture_output=True, text=True, timeout=10
            )
            
            for line in result.stdout.split('\n'):
                lower_line = line.lower()
                if 'fan' in lower_line and 'rpm' in lower_line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                            if value > 0:
                                fans[name] = value
                        except ValueError:
                            continue
        except Exception:
            pass
        return fans
    
    def parse_gpu_burn_output(self, line: str):
        """解析 gpu-burn 輸出取得 TFLOPS"""
        # 典型格式: "GPU 0: 32456 GF/s  proc'd: 123456 (123.4 Gflop/s)   errors: 0   temps: 65 C"
        # 或: "GPU 0(NVIDIA H100): OK - 1234.5 Gflop/s"
        
        # 匹配 TFLOPS/GFLOPS
        patterns = [
            r'GPU\s*(\d+).*?(\d+(?:\.\d+)?)\s*(?:TF|Tflop)',  # TFLOPS
            r'GPU\s*(\d+).*?(\d+(?:\.\d+)?)\s*(?:GF|Gflop)',  # GFLOPS
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                gpu_id = int(match.group(1))
                value = float(match.group(2))
                
                # 轉換為 TFLOPS
                if 'gf' in line.lower() or 'gflop' in line.lower():
                    value = value / 1000.0
                
                self.current_tflops[gpu_id] = value
                break
    
    def read_gpu_burn_output(self):
        """持續讀取 gpu-burn 輸出"""
        if self.gpu_burn_process is None:
            return
            
        while self.monitoring and self.gpu_burn_process.poll() is None:
            line = self.gpu_burn_process.stdout.readline()
            if line:
                line = line.strip()
                self.gpu_burn_output.append(line)
                self.parse_gpu_burn_output(line)
                print(f"[gpu-burn] {line}")
    
    def collect_sample(self) -> List[SensorData]:
        """收集一次完整的 sensor 數據"""
        timestamp = datetime.now().isoformat()
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # GPU 數據
        gpus = self.get_nvidia_smi_data()
        
        # IPMI 數據
        inlet_temp = self.get_ipmi_sensor(self.ipmi_sensors['inlet_temp'])
        exhaust_temp = self.get_ipmi_sensor(self.ipmi_sensors['exhaust_temp'])
        cpu_temp = self.get_ipmi_sensor(self.ipmi_sensors['cpu_temp'])
        fan_power = self.get_ipmi_sensor(self.ipmi_sensors['fan_power'])
        psu_power = self.get_ipmi_sensor(self.ipmi_sensors['psu_power'])
        cpu_power = self.get_ipmi_sensor(self.ipmi_sensors['cpu_power'])
        memory_power = self.get_ipmi_sensor(self.ipmi_sensors['memory_power'])
        
        # 風扇轉速
        fan_speeds = self.get_fan_speeds()
        
        samples = []
        for gpu in gpus:
            tflops = self.current_tflops.get(gpu['index'], 0.0)
            
            sample = SensorData(
                timestamp=timestamp,
                elapsed_seconds=round(elapsed, 2),
                gpu_id=gpu['index'],
                gpu_name=gpu['name'],
                gpu_temp_c=gpu['temp'],
                gpu_power_w=gpu['power'],
                gpu_fan_speed_pct=gpu['fan'],
                gpu_memory_used_mb=gpu['mem_used'],
                gpu_memory_total_mb=gpu['mem_total'],
                gpu_utilization_pct=gpu['util'],
                gpu_tflops=round(tflops, 3),
                inlet_temp_c=inlet_temp,
                exhaust_temp_c=exhaust_temp,
                cpu_temp_c=cpu_temp,
                total_fan_power_w=fan_power,
                total_psu_power_w=psu_power,
                cpu_power_w=cpu_power,
                memory_power_w=memory_power,
                fan_speeds_rpm=json.dumps(fan_speeds)
            )
            samples.append(sample)
        
        return samples
    
    def set_power_limit(self, limit: int):
        """設定 GPU Power Limit"""
        gpu_count = self.get_gpu_count()
        for i in range(gpu_count):
            try:
                subprocess.run([
                    'nvidia-smi', '-i', str(i), '-pl', str(limit)
                ], check=True, capture_output=True)
                print(f"GPU {i}: Power limit set to {limit}W")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not set power limit for GPU {i}: {e}")
    
    def build_gpu_burn_command(self) -> List[str]:
        """建構 gpu-burn 命令"""
        cmd = [self.gpu_burn_path]
        
        if self.use_tensor_cores:
            cmd.append('-tc')
        
        if self.use_doubles:
            cmd.append('-d')
        
        cmd.append(str(self.duration))
        
        return cmd
    
    def run(self) -> Path:
        """執行完整的 burn 測試與監控"""
        # 設定 power limit
        if self.power_limit:
            self.set_power_limit(self.power_limit)
        
        # 產生輸出檔名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        pl_str = f"_pl{self.power_limit}w" if self.power_limit else ""
        tc_str = "_tc" if self.use_tensor_cores else ""
        csv_path = self.output_dir / f"gpu_burn_{timestamp_str}{pl_str}{tc_str}.csv"
        
        # 啟動 gpu-burn
        cmd = self.build_gpu_burn_command()
        print(f"Starting: {' '.join(cmd)}")
        
        self.gpu_burn_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 啟動輸出讀取執行緒
        self.monitoring = True
        self.start_time = time.time()
        
        output_thread = threading.Thread(target=self.read_gpu_burn_output)
        output_thread.start()
        
        # 主監控迴圈
        try:
            while self.gpu_burn_process.poll() is None:
                samples = self.collect_sample()
                self.data_records.extend(samples)
                
                # 即時顯示摘要
                if samples:
                    s = samples[0]
                    print(f"[{s.elapsed_seconds:.1f}s] GPU0: {s.gpu_temp_c}°C, "
                          f"{s.gpu_power_w:.1f}W, {s.gpu_tflops:.2f} TFLOPS | "
                          f"Inlet: {s.inlet_temp_c}°C | PSU: {s.total_psu_power_w}W")
                
                time.sleep(self.sample_interval)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            if self.gpu_burn_process:
                self.gpu_burn_process.terminate()
        finally:
            self.monitoring = False
            output_thread.join(timeout=5)
        
        # 儲存 CSV
        self.save_csv(csv_path)
        print(f"\nData saved to: {csv_path}")
        
        return csv_path
    
    def save_csv(self, path: Path):
        """儲存數據為 CSV"""
        if not self.data_records:
            print("Warning: No data to save")
            return
        
        fieldnames = list(asdict(self.data_records[0]).keys())
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.data_records:
                writer.writerow(asdict(record))


def main():
    parser = argparse.ArgumentParser(
        description='GPU Burn Monitor - 監控 GPU burn 測試並記錄系統數據',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例:
  %(prog)s -d 300 -pl 400 -tc          # 300秒測試, 400W power limit, 使用 tensor cores
  %(prog)s -d 60 --no-tc               # 60秒測試, 不使用 tensor cores
  %(prog)s -d 120 -pl 200 -o ./my_results
        '''
    )
    
    parser.add_argument('-d', '--duration', type=int, default=60,
                        help='測試時間 (秒), 預設 60')
    parser.add_argument('-pl', '--power-limit', type=int, default=None,
                        help='GPU Power Limit (瓦特)')
    parser.add_argument('-tc', '--tensor-cores', action='store_true', default=True,
                        help='使用 Tensor Cores (預設啟用)')
    parser.add_argument('--no-tc', action='store_true',
                        help='不使用 Tensor Cores')
    parser.add_argument('--doubles', action='store_true',
                        help='使用 double precision')
    parser.add_argument('-o', '--output', type=str, default='./results',
                        help='輸出目錄, 預設 ./results')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='取樣間隔 (秒), 預設 1.0')
    parser.add_argument('--gpu-burn-path', type=str, default='gpu-burn',
                        help='gpu-burn 執行檔路徑')
    
    args = parser.parse_args()
    
    use_tc = args.tensor_cores and not args.no_tc
    
    monitor = GPUBurnMonitor(
        gpu_burn_path=args.gpu_burn_path,
        duration=args.duration,
        power_limit=args.power_limit,
        use_tensor_cores=use_tc,
        use_doubles=args.doubles,
        output_dir=args.output,
        sample_interval=args.interval
    )
    
    csv_path = monitor.run()
    
    # 產生圖表
    print("\n產生圖表中...")
    try:
        from generate_charts import generate_charts
        generate_charts(csv_path)
    except ImportError:
        print("請執行 python generate_charts.py <csv_file> 產生圖表")


if __name__ == '__main__':
    main()
