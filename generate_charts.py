#!/usr/bin/env python3
"""
GPU Burn Charts Generator - 從 CSV 產生互動式圖表
"""

import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime


def generate_charts(csv_path: str, output_path: str = None):
    """從 CSV 產生 HTML 圖表"""
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # 讀取數據
    df = pd.read_csv(csv_path)
    
    # 輸出路徑
    if output_path is None:
        output_path = csv_path.with_suffix('.html')
    else:
        output_path = Path(output_path)
    
    # 取得 GPU 列表
    gpu_ids = sorted(df['gpu_id'].unique())
    gpu_name = df['gpu_name'].iloc[0] if 'gpu_name' in df.columns else 'GPU'
    
    # 準備每個 GPU 的數據
    gpu_data = {}
    for gpu_id in gpu_ids:
        gpu_df = df[df['gpu_id'] == gpu_id].copy()
        gpu_data[int(gpu_id)] = {
            'elapsed': gpu_df['elapsed_seconds'].tolist(),
            'temp': gpu_df['gpu_temp_c'].tolist(),
            'power': gpu_df['gpu_power_w'].tolist(),
            'tflops': gpu_df['gpu_tflops'].tolist(),
            'fan': gpu_df['gpu_fan_speed_pct'].tolist(),
            'util': gpu_df['gpu_utilization_pct'].tolist(),
            'mem_used': gpu_df['gpu_memory_used_mb'].tolist(),
        }
    
    # 系統數據 (取第一個 GPU 的記錄，因為系統數據都一樣)
    sys_df = df[df['gpu_id'] == gpu_ids[0]].copy()
    system_data = {
        'elapsed': sys_df['elapsed_seconds'].tolist(),
        'inlet_temp': sys_df['inlet_temp_c'].fillna(0).tolist(),
        'exhaust_temp': sys_df['exhaust_temp_c'].fillna(0).tolist(),
        'psu_power': sys_df['total_psu_power_w'].fillna(0).tolist(),
        'cpu_power': sys_df['cpu_power_w'].fillna(0).tolist(),
        'memory_power': sys_df['memory_power_w'].fillna(0).tolist(),
        'fan_power': sys_df['total_fan_power_w'].fillna(0).tolist(),
    }
    
    # 統計摘要
    stats = {
        'duration': df['elapsed_seconds'].max(),
        'gpu_count': len(gpu_ids),
        'gpu_name': gpu_name,
        'max_gpu_temp': df['gpu_temp_c'].max(),
        'avg_gpu_temp': df['gpu_temp_c'].mean(),
        'max_gpu_power': df['gpu_power_w'].max(),
        'avg_gpu_power': df['gpu_power_w'].mean(),
        'total_gpu_power': df.groupby('elapsed_seconds')['gpu_power_w'].sum().mean(),
        'max_tflops': df['gpu_tflops'].max(),
        'avg_tflops': df[df['gpu_tflops'] > 0]['gpu_tflops'].mean() if (df['gpu_tflops'] > 0).any() else 0,
        'total_tflops': df.groupby('elapsed_seconds')['gpu_tflops'].sum().mean(),
        'max_psu_power': sys_df['total_psu_power_w'].max() if 'total_psu_power_w' in sys_df else 0,
        'avg_psu_power': sys_df['total_psu_power_w'].mean() if 'total_psu_power_w' in sys_df else 0,
        'max_inlet_temp': sys_df['inlet_temp_c'].max() if 'inlet_temp_c' in sys_df else 0,
    }
    
    # 產生 HTML
    html_content = generate_html(gpu_data, system_data, stats, csv_path.stem)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Charts saved to: {output_path}")
    return output_path


def generate_html(gpu_data: dict, system_data: dict, stats: dict, title: str) -> str:
    """產生完整的 HTML 圖表頁面"""
    
    # GPU 顏色
    gpu_colors = [
        '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', 
        '#ffeaa7', '#dfe6e9', '#fd79a8', '#a29bfe'
    ]
    
    html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Burn Report - {title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Noto+Sans+TC:wght@400;500;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --border-color: #2a2a3a;
            --text-primary: #e8e8f0;
            --text-secondary: #8888a0;
            --accent-red: #ff6b6b;
            --accent-cyan: #4ecdc4;
            --accent-blue: #45b7d1;
            --accent-green: #96ceb4;
            --accent-yellow: #ffeaa7;
            --accent-purple: #a29bfe;
            --glow-red: rgba(255, 107, 107, 0.3);
            --glow-cyan: rgba(78, 205, 196, 0.3);
        }}
        
        body {{
            font-family: 'Noto Sans TC', 'JetBrains Mono', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px;
            background-image: 
                radial-gradient(ellipse at 20% 20%, rgba(78, 205, 196, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(255, 107, 107, 0.05) 0%, transparent 50%);
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 40px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            position: relative;
            overflow: hidden;
        }}
        
        header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-red), var(--accent-cyan), var(--accent-blue));
        }}
        
        h1 {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -1px;
            margin-bottom: 8px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .stat-card:hover {{
            border-color: var(--accent-cyan);
            box-shadow: 0 0 20px var(--glow-cyan);
            transform: translateY(-2px);
        }}
        
        .stat-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-cyan);
            line-height: 1.2;
        }}
        
        .stat-value.temp {{
            color: var(--accent-red);
        }}
        
        .stat-value.power {{
            color: var(--accent-yellow);
        }}
        
        .stat-value.perf {{
            color: var(--accent-green);
        }}
        
        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .charts-section {{
            margin-bottom: 32px;
        }}
        
        .section-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 16px;
            padding-left: 16px;
            border-left: 4px solid var(--accent-cyan);
            color: var(--text-primary);
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 24px;
        }}
        
        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            transition: all 0.3s ease;
        }}
        
        .chart-card:hover {{
            border-color: var(--accent-cyan);
        }}
        
        .chart-card h3 {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .chart-card h3::before {{
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-cyan);
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        
        .chart-card.large .chart-container {{
            height: 400px;
        }}
        
        .gpu-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border-color);
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}
        
        footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            h1 {{
                font-size: 1.75rem;
            }}
            
            .stat-value {{
                font-size: 1.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔥 GPU BURN REPORT</h1>
            <p class="subtitle">{title} | {stats['gpu_count']}x {stats['gpu_name']} | {stats['duration']:.0f}s</p>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value temp">{stats['max_gpu_temp']:.1f}°C</div>
                <div class="stat-label">Max GPU Temp</div>
            </div>
            <div class="stat-card">
                <div class="stat-value temp">{stats['avg_gpu_temp']:.1f}°C</div>
                <div class="stat-label">Avg GPU Temp</div>
            </div>
            <div class="stat-card">
                <div class="stat-value power">{stats['max_gpu_power']:.0f}W</div>
                <div class="stat-label">Max GPU Power</div>
            </div>
            <div class="stat-card">
                <div class="stat-value power">{stats['total_gpu_power']:.0f}W</div>
                <div class="stat-label">Total GPU Power</div>
            </div>
            <div class="stat-card">
                <div class="stat-value perf">{stats['max_tflops']:.2f}</div>
                <div class="stat-label">Max TFLOPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value perf">{stats['total_tflops']:.2f}</div>
                <div class="stat-label">Total TFLOPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value power">{stats['max_psu_power']:.0f}W</div>
                <div class="stat-label">Max PSU Power</div>
            </div>
            <div class="stat-card">
                <div class="stat-value temp">{stats['max_inlet_temp']:.1f}°C</div>
                <div class="stat-label">Max Inlet Temp</div>
            </div>
        </div>
        
        <section class="charts-section">
            <h2 class="section-title">GPU Metrics</h2>
            <div class="charts-grid">
                <div class="chart-card large">
                    <h3>GPU Temperature (°C)</h3>
                    <div class="chart-container">
                        <canvas id="chartGpuTemp"></canvas>
                    </div>
                </div>
                <div class="chart-card large">
                    <h3>GPU Power (W)</h3>
                    <div class="chart-container">
                        <canvas id="chartGpuPower"></canvas>
                    </div>
                </div>
                <div class="chart-card large">
                    <h3>GPU Performance (TFLOPS)</h3>
                    <div class="chart-container">
                        <canvas id="chartGpuTflops"></canvas>
                    </div>
                </div>
                <div class="chart-card large">
                    <h3>GPU Fan Speed (%)</h3>
                    <div class="chart-container">
                        <canvas id="chartGpuFan"></canvas>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="charts-section">
            <h2 class="section-title">System Metrics</h2>
            <div class="charts-grid">
                <div class="chart-card large">
                    <h3>System Temperature (°C)</h3>
                    <div class="chart-container">
                        <canvas id="chartSysTemp"></canvas>
                    </div>
                </div>
                <div class="chart-card large">
                    <h3>System Power (W)</h3>
                    <div class="chart-container">
                        <canvas id="chartSysPower"></canvas>
                    </div>
                </div>
            </div>
        </section>
        
        <footer>
            Generated by GPU Burn Monitor | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </footer>
    </div>
    
    <script>
        const gpuData = {json.dumps(gpu_data)};
        const systemData = {json.dumps(system_data)};
        const gpuColors = {json.dumps(gpu_colors)};
        
        Chart.defaults.color = '#8888a0';
        Chart.defaults.borderColor = '#2a2a3a';
        Chart.defaults.font.family = "'JetBrains Mono', monospace";
        
        const commonOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{
                intersect: false,
                mode: 'index'
            }},
            plugins: {{
                legend: {{
                    position: 'bottom',
                    labels: {{
                        usePointStyle: true,
                        padding: 16
                    }}
                }}
            }},
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: 'Time (s)'
                    }},
                    grid: {{
                        color: 'rgba(42, 42, 58, 0.5)'
                    }}
                }},
                y: {{
                    grid: {{
                        color: 'rgba(42, 42, 58, 0.5)'
                    }}
                }}
            }}
        }};
        
        function createGpuChart(canvasId, dataKey, yLabel, unit = '') {{
            const ctx = document.getElementById(canvasId).getContext('2d');
            const datasets = Object.keys(gpuData).map((gpuId, index) => ({{
                label: `GPU ${{gpuId}}`,
                data: gpuData[gpuId][dataKey],
                borderColor: gpuColors[index % gpuColors.length],
                backgroundColor: gpuColors[index % gpuColors.length] + '20',
                borderWidth: 2,
                fill: false,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 4
            }}));
            
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: gpuData[Object.keys(gpuData)[0]].elapsed,
                    datasets: datasets
                }},
                options: {{
                    ...commonOptions,
                    scales: {{
                        ...commonOptions.scales,
                        y: {{
                            ...commonOptions.scales.y,
                            title: {{
                                display: true,
                                text: yLabel + (unit ? ` (${{unit}})` : '')
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // GPU Charts
        createGpuChart('chartGpuTemp', 'temp', 'Temperature', '°C');
        createGpuChart('chartGpuPower', 'power', 'Power', 'W');
        createGpuChart('chartGpuTflops', 'tflops', 'Performance', 'TFLOPS');
        createGpuChart('chartGpuFan', 'fan', 'Fan Speed', '%');
        
        // System Temperature Chart
        new Chart(document.getElementById('chartSysTemp').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: systemData.elapsed,
                datasets: [
                    {{
                        label: 'Inlet',
                        data: systemData.inlet_temp,
                        borderColor: '#4ecdc4',
                        backgroundColor: '#4ecdc420',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Exhaust',
                        data: systemData.exhaust_temp,
                        borderColor: '#ff6b6b',
                        backgroundColor: '#ff6b6b20',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                ...commonOptions,
                scales: {{
                    ...commonOptions.scales,
                    y: {{
                        ...commonOptions.scales.y,
                        title: {{
                            display: true,
                            text: 'Temperature (°C)'
                        }}
                    }}
                }}
            }}
        }});
        
        // System Power Chart
        new Chart(document.getElementById('chartSysPower').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: systemData.elapsed,
                datasets: [
                    {{
                        label: 'Total PSU',
                        data: systemData.psu_power,
                        borderColor: '#ffeaa7',
                        backgroundColor: '#ffeaa720',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }},
                    {{
                        label: 'CPU',
                        data: systemData.cpu_power,
                        borderColor: '#a29bfe',
                        backgroundColor: '#a29bfe20',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Memory',
                        data: systemData.memory_power,
                        borderColor: '#74b9ff',
                        backgroundColor: '#74b9ff20',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Fans',
                        data: systemData.fan_power,
                        borderColor: '#55efc4',
                        backgroundColor: '#55efc420',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                ...commonOptions,
                scales: {{
                    ...commonOptions.scales,
                    y: {{
                        ...commonOptions.scales.y,
                        title: {{
                            display: true,
                            text: 'Power (W)'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''
    
    return html


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_charts.py <csv_file> [output_html]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_charts(csv_file, output_file)
