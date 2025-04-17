import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

horizon = 24  # 24,6,3,1
lead_time = 10
prediction_length = lead_time * 24 / horizon
    
# 基础路径
base_path = f'/opt/dlami/nvme/model/inference/{horizon}/'

# 获取所有子文件夹（初始预报时间）
forecast_init_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
print(f"Found {len(forecast_init_dirs)} forecast initialization times")

# 存储所有预报数据
all_forecasts = []

# 遍历所有初始预报时间文件夹
for init_dir in tqdm(forecast_init_dirs, desc="Processing forecast times"):
    csv_path = os.path.join(base_path, init_dir, 'csv', 'rmse_surface_wind_speed.csv')
    
    # 检查文件是否存在
    if os.path.exists(csv_path):
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 检查CSV文件格式
            if len(df.columns) == 2:
                # 有些文件第一列可能没有列名，我们检查并处理
                if df.columns[0] == '':
                    df.columns = ['date', 'wind_speed']
                elif 'Unnamed' in df.columns[0]:
                    df.columns = ['date', 'wind_speed']
                
                # 添加初始预报时间列
                df['init_time'] = init_dir
                
                # 检查日期列的类型并相应处理
                first_date = str(df['date'].iloc[0])
                
                # 尝试将初始时间解析为日期
                try:
                    init_date = pd.to_datetime(init_dir, format='%Y%m%d%H')
                    
                    # 如果date列是字符串且格式为YYYYMMDDHH
                    if isinstance(first_date, str) and len(first_date) == 10:
                        df['init_date'] = init_date
                        df['forecast_date'] = pd.to_datetime(df['date'], format='%Y%m%d%H')
                        df['forecast_hour'] = (df['forecast_date'] - df['init_date']).dt.total_seconds() / 3600
                        df['forecast_hour'] = df['forecast_hour'].astype(int)
                    else:
                        print('数据格式错误')
                except:
                    print('无法解析初始时间')
                
                # 添加到所有预报数据中
                all_forecasts.append(df)
            else:
                print(f"Unexpected file format in {csv_path}: {df.columns}")
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            # 打印更详细的错误信息
            import traceback
            traceback.print_exc()
    else:
        print(f"File not found: {csv_path}")

# 合并所有预报数据
if all_forecasts:
    combined_df = pd.concat(all_forecasts, ignore_index=True)
    
    # 检查是否有超过10天的预报
    max_hours = combined_df['forecast_hour'].max() if 'forecast_hour' in combined_df.columns else prediction_length*horizon
    forecast_hours = range(horizon, min(max_hours + 1, prediction_length*horizon+1))  # 限制到最多10天
    
    # 只保留1-10天的预报
    combined_df = combined_df[combined_df['forecast_hour'].between(horizon, prediction_length*horizon)]
    
    # 计算每个预报天的平均RMSE
    avg_by_hour = combined_df.groupby('forecast_hour')['wind_speed'].agg(['mean', 'std', 'count'])
    
    print("\nAverage RMSE by forecast hour:")
    print(avg_by_hour)
    
    # 保存到CSV
    avg_by_hour.to_csv(f'average_rmse_by_forecast_hour_horizon_{horizon}.csv')
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        avg_by_hour.index, 
        avg_by_hour['mean'], 
        yerr=avg_by_hour['std'], 
        fmt='o-', 
        capsize=5, 
        ecolor='gray', 
        markersize=8
    )
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Forecast Hour')
    plt.ylabel('RMSE (Wind Speed)')
    plt.title('Average RMSE by Forecast Hour')
    plt.xticks(forecast_hours[::horizon])  # 只显示每隔几个标签
    plt.xlim(left=0)
    plt.ylim(bottom=0)  # 设置 y 轴从 0 开始，只设置下限，上限自动调整
    
    # 添加数值标签
    for day, mean, count in zip(avg_by_hour.index, avg_by_hour['mean'], avg_by_hour['count']):
        plt.text(day, mean + 0.05, f'{mean:.4f}\n(n={count})', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'average_rmse_by_forecast_hour_horizon_{horizon}.png', dpi=300)
    # plt.show()
    
    # 创建热图来显示所有初始时间的预报性能
    if len(forecast_init_dirs) > 1:
        # 针对热图创建透视表
        pivot_df = combined_df.pivot_table(
            index='init_time', 
            columns='forecast_hour', 
            values='wind_speed', 
            aggfunc='mean'
        )
        
        # 排序以便更清晰地查看
        pivot_df = pivot_df.sort_index()
        
        # 绘制热图
        plt.figure(figsize=(14, max(8, len(forecast_init_dirs) * 0.3)))
        heatmap = plt.imshow(pivot_df.values, aspect='auto', cmap='viridis')
        plt.colorbar(heatmap, label='RMSE (Wind Speed)')
        
        # 设置坐标轴
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        plt.xticks(range(len(pivot_df.columns)), [f'Day {d}' for d in pivot_df.columns])
        
        plt.title('RMSE by Initialization Time and Forecast Day')
        plt.xlabel('Forecast Day')
        plt.ylabel('Initialization Time')
        plt.tight_layout()
        plt.savefig(f'rmse_heatmap_by_init_time_horizon_{horizon}.png', dpi=300)
        # plt.show()
        
        # 保存透视表
        pivot_df.to_csv(f'rmse_by_init_time_and_forecast_hour_horizon_{horizon}.csv')
else:
    print("No valid forecast data found")