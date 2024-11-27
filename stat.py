import os
import warnings
import numpy as np
import torch
from torch.utils import data
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

from tensorboardX import SummaryWriter
import logging
import time
import argparse
from tqdm import tqdm

import torch
from torch.utils import data

from era5_data.config import cfg
from era5_data import utils, utils_data

warnings.simplefilter(action='ignore', category=FutureWarning)

class ComprehensiveStats:
    def __init__(self, num_channels):
        self.num_channels = num_channels
        # 基础统计
        self.sum = torch.zeros(num_channels)
        self.sum_sq = torch.zeros(num_channels)
        
        # 风速统计
        self.wind_stats = {
            'speed_sum': 0,
            'speed_sq_sum': 0,
            'max_speed': float('-inf'),
            'min_speed': float('inf'),
            'speed_values': [],  # 存储所有风速值用于分位数计算
        }
        
        # 季节性统计
        self.seasonal_stats = {
            'spring': [], 'summer': [], 
            'autumn': [], 'winter': []
        }
        
        # 空间统计
        self.spatial_stats = {
            'latitude_means': [],
            'longitude_means': []
        }
        
        # 极端值统计
        self.extremes = {
            'strong_wind_hours': 0  # >310m/s的小时数
        }
        
        self.count = 0
        
    def get_season(self, period):
        # 假设period是datetime对象
        month = pd.to_datetime(period[0], format='%Y%m%d%H').month
        if month in [3,4,5]:
            return 'spring'
        elif month in [6,7,8]:
            return 'summer'
        elif month in [9,10,11]:
            return 'autumn'
        else:
            return 'winter'
            
    def update(self, tensor, periods):
        # tensor shape: [batch, channels, height, width]
        U10 = tensor[:, 1, :, :]
        V10 = tensor[:, 2, :, :]
        # print('U10:', U10.mean(), 'V10:', V10.mean())
        
        # 1. 基础统计
        self.sum += tensor.mean(dim=[0, 2, 3]).cpu()
        self.sum_sq += (tensor**2).mean(dim=[0, 2, 3]).cpu()
        
        # 2. 风速统计
        wind_speed = torch.sqrt(U10**2 + V10**2)
        mean_wind_speed = wind_speed.mean().item()
        self.wind_stats['speed_sum'] += mean_wind_speed
        self.wind_stats['speed_sq_sum'] += mean_wind_speed**2
        self.wind_stats['max_speed'] = max(self.wind_stats['max_speed'], wind_speed.max().item())
        self.wind_stats['min_speed'] = min(self.wind_stats['min_speed'], wind_speed.min().item())
        self.wind_stats['speed_values'].extend(wind_speed.flatten().tolist())
        
        # 3. 季节性统计
        for i, period in enumerate(periods):
            season = self.get_season(period)
            self.seasonal_stats[season].append(wind_speed[i].mean().item())
        
        # 4. 空间统计
        self.spatial_stats['latitude_means'].append(wind_speed.mean(dim=2).cpu().numpy())  # 纬度平均
        self.spatial_stats['longitude_means'].append(wind_speed.mean(dim=1).cpu().numpy()) # 经度平均
        
        # 5. 极端值统计
        self.extremes['strong_wind_hours'] += (wind_speed > 310).sum().item()
        
        self.count += tensor.size(0) * tensor.size(2) * tensor.size(3)

    def get_comprehensive_stats(self):
        # 1. 基础统计
        means = self.sum / self.count
        std = torch.sqrt(self.sum_sq/self.count - (self.sum/self.count)**2)
        
        # 2. 风速统计
        wind_values = np.array(self.wind_stats['speed_values'])
        wind_percentiles = np.percentile(wind_values, [25, 50, 75, 90, 95, 99])
        
        # 3. 季节性统计
        seasonal_means = {
            season: np.mean(values) if values else 0 
            for season, values in self.seasonal_stats.items()
        }
        
        # 4. 空间统计
        lat_means = np.mean(self.spatial_stats['latitude_means'], axis=0)
        lon_means = np.mean(self.spatial_stats['longitude_means'], axis=0)
        
        # 5. 计算偏度和峰度
        skewness = stats.skew(wind_values)
        kurtosis = stats.kurtosis(wind_values)
        
        return {
            'basic_stats': {
                'channel_means': means,
                'channel_std': std,
            },
            'wind_stats': {
                'mean': np.mean(wind_values),
                'std': np.std(wind_values),
                'percentiles': {
                    '25th': wind_percentiles[0],
                    'median': wind_percentiles[1],
                    '75th': wind_percentiles[2],
                    '90th': wind_percentiles[3],
                    '95th': wind_percentiles[4],
                    '99th': wind_percentiles[5]
                },
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'seasonal_stats': seasonal_means,
            'spatial_stats': {
                'latitude_profile': lat_means,
                'longitude_profile': lon_means
            },
            'extreme_stats': {
                'strong_wind_percentage': 100 * self.extremes['strong_wind_hours'] / self.count
            }
        }

def plot_statistics(stats, year, output_dir):
    # 1. 季节性变化
    plt.figure(figsize=(10, 6))
    seasons = list(stats['seasonal_stats'].keys())
    values = list(stats['seasonal_stats'].values())
    plt.bar(seasons, values)
    plt.title(f'Seasonal Wind Speed Variation - {year}')
    plt.ylabel('Mean Wind Speed (m/s)')
    plt.savefig(os.path.join(output_dir, f'seasonal_variation_{year}.png'))
    plt.close()
    
    # 2. 空间分布
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(stats['spatial_stats']['latitude_profile'])
    plt.title('Latitude Profile')
    plt.subplot(1, 2, 2)
    plt.plot(stats['spatial_stats']['longitude_profile'])
    plt.title('Longitude Profile')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'spatial_distribution_{year}.png'))
    plt.close()

def analyze_year_data(dataloader, year, output_dir):
    stats = ComprehensiveStats(4)
    
    for data in tqdm(dataloader, desc=f'Analyzing {year}'):
        input_test, input_surface_test, target_test, target_surface_test, periods_test = data
        stats.update(target_surface_test, periods_test[1:2])
    
    results = stats.get_comprehensive_stats()
    
    # 保存统计结果
    output_file = os.path.join(output_dir, f'stats_{year}.txt')
    with open(output_file, 'w') as f:
        for category, values in results.items():
            f.write(f"\n{category.upper()}:\n")
            f.write(str(values))
            f.write('\n')
    
    # 绘制可视化图表
    # plot_statistics(results, year, output_dir)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_net', type=str, default="stat")
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    starts = time.time()

    PATH = cfg.PG_INPUT_PATH

    output_path = os.path.join(
        cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)

    writer_path = os.path.join(output_path, "writer")
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)

    writer = SummaryWriter(writer_path)

    logger_name = args.type_net + str(cfg.PG.HORIZON)
    utils.logger_info(logger_name, os.path.join(
        output_path, logger_name + '.log'))

    logger = logging.getLogger(logger_name)

    # test_dataset = utils_data.NetCDFDataset(nc_path=PATH,
    test_dataset = utils_data.PTDataset(pt_path=PATH,
                                        data_transform=None,
                                        training=False,
                                        validation=False,
                                        startDate=cfg.PG.TEST.START_TIME,
                                        endDate=cfg.PG.TEST.END_TIME,
                                        freq=cfg.PG.TEST.FREQUENCY,
                                        horizon=cfg.PG.HORIZON,
                                        device='cpu')  # device

    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=cfg.PG.TEST.BATCH_SIZE,
                                      drop_last=True, shuffle=False, num_workers=args.num_workers, prefetch_factor=2, pin_memory=True)  # default: num_workers=0, pin_memory=False
    
    # 创建输出目录
    output_dir = os.path.join(output_path, "statistics")
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析数据
    stats = analyze_year_data(test_dataloader, 2023, output_dir)  # TODO: manully change year
    
    # 打印主要结果
    print("\nKey Statistics Summary:")
    print("\nWind Speed Statistics:")
    for key, value in stats['wind_stats'].items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value:.3f}")
        else:
            print(f"{key}: {value:.3f}")
    
    print("\nSeasonal Analysis:")
    for season, value in stats['seasonal_stats'].items():
        print(f"{season}: {value:.3f}")
    
    print("\nExtreme Weather Events:")
    for event, percentage in stats['extreme_stats'].items():
        print(f"{event}: {percentage:.2f}%")