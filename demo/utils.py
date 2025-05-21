import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
import io
from PIL import Image

def visualize_map(data, title="", colormap="viridis", vmin=None, vmax=None):
    """
    创建地图可视化
    
    参数:
        data: numpy数组，要可视化的数据
        title: 图表标题
        colormap: 颜色映射名称
        vmin, vmax: 颜色范围的最小值和最大值
    
    返回:
        matplotlib图形对象
    """
    # 创建图形
    fig = plt.figure(figsize=(10, 6))
    
    # 创建地图投影
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    # 添加地图特征
    ax.add_feature(cfeature.COASTLINE, linewidth=0.75, edgecolor='black') # Adjusted style
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # 如果数据维度太大，进行下采样
    if data.shape[0] > 180 or data.shape[1] > 360:
        # 简单的下采样方法，实际应用中可能需要更复杂的方法
        step_lat = max(1, data.shape[0] // 180)
        step_lon = max(1, data.shape[1] // 360)
        data_sampled = data[::step_lat, ::step_lon]
    else:
        data_sampled = data
    
    # 计算经纬度网格（假设数据覆盖全球）
    lats = np.linspace(-90, 90, data_sampled.shape[0])
    lons = np.linspace(-180, 180, data_sampled.shape[1])
    lon, lat = np.meshgrid(lons, lats)
    
    # 如果没有提供vmin和vmax，自动计算
    if vmin is None:
        vmin = np.nanpercentile(data_sampled, 2)  # 使用百分位数避免异常值影响
    if vmax is None:
        vmax = np.nanpercentile(data_sampled, 98)
    
    # 绘制数据
    # Define contour levels
    num_levels = 15
    levels = np.linspace(vmin, vmax, num_levels)
    
    # Filled contours
    im = ax.contourf(lon, lat, data_sampled, levels=levels, transform=ccrs.PlateCarree(),
                     cmap=colormap, extend='both')
    
    # Line contours
    line_contours = ax.contour(lon, lat, data_sampled, levels=levels, colors='black', 
                               linewidths=0.5, transform=ccrs.PlateCarree())
    
    # Add contour labels (optional, can be made less intrusive or removed if too cluttered)
    ax.clabel(line_contours, inline=True, fontsize=6, fmt='%1.1f')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, 
                        ticks=np.linspace(vmin, vmax, num_levels // 2)) # Adjust ticks for clarity
    cbar.ax.tick_params(labelsize=8)
    
    # 设置标题
    ax.set_title(title)
    
    # 设置地图范围
    ax.set_global()
    
    return fig

def calculate_metrics(output, target, output_surface, target_surface):
    """
    计算预测结果的性能指标
    
    参数:
        output: 高空数据预测结果
        target: 高空数据真实值
        output_surface: 地表数据预测结果
        target_surface: 地表数据真实值
    
    返回:
        包含各种指标的字典
    """
    # 计算高空数据的RMSE
    rmse_z = torch.sqrt(torch.mean((output[0] - target[0])**2)).item()
    rmse_q = torch.sqrt(torch.mean((output[1] - target[1])**2)).item()
    rmse_t = torch.sqrt(torch.mean((output[2] - target[2])**2)).item()
    rmse_u = torch.sqrt(torch.mean((output[3] - target[3])**2)).item()
    rmse_v = torch.sqrt(torch.mean((output[4] - target[4])**2)).item()
    
    # 计算风速
    wind_speed_output = torch.sqrt(output[3]**2 + output[4]**2)
    wind_speed_target = torch.sqrt(target[3]**2 + target[4]**2)
    rmse_wind = torch.sqrt(torch.mean((wind_speed_output - wind_speed_target)**2)).item()
    
    # 计算地表数据的RMSE
    # Corrected indices: msl:0, u10:1, v10:2, t2m:3
    rmse_msl = torch.sqrt(torch.mean((output_surface[0] - target_surface[0])**2)).item()
    rmse_u10 = torch.sqrt(torch.mean((output_surface[1] - target_surface[1])**2)).item()
    rmse_v10 = torch.sqrt(torch.mean((output_surface[2] - target_surface[2])**2)).item()
    rmse_t2m = torch.sqrt(torch.mean((output_surface[3] - target_surface[3])**2)).item()
    
    # 计算地表风速 (using corrected u10 at index 1, v10 at index 2)
    wind10_output = torch.sqrt(output_surface[1]**2 + output_surface[2]**2)
    wind10_target = torch.sqrt(target_surface[1]**2 + target_surface[2]**2)
    rmse_wind10 = torch.sqrt(torch.mean((wind10_output - wind10_target)**2)).item()
    
    # 返回所有指标
    return {
        'rmse_z': rmse_z,
        'rmse_q': rmse_q,
        'rmse_t': rmse_t,
        'rmse_u': rmse_u,
        'rmse_v': rmse_v,
        'rmse_wind': rmse_wind,
        'rmse_u10': rmse_u10,
        'rmse_v10': rmse_v10,
        'rmse_t2m': rmse_t2m,
        'rmse_msl': rmse_msl,
        'rmse_wind10': rmse_wind10
    }

def load_sample_data():
    """
    加载示例数据（用于演示）
    
    返回:
        input_data: 输入高空数据
        input_surface: 输入地表数据
        target: 目标高空数据
        target_surface: 目标地表数据
    """
    # 创建随机数据用于演示
    # 实际应用中，这里应该从文件加载真实数据
    
    # 高空数据: [变量, 层级, 纬度, 经度]
    # 5个变量: z, q, t, u, v
    # 13个层级: 从1000hPa到50hPa
    # 简化的分辨率: 181x360 (实际ERA5是721x1440)
    input_data = np.random.randn(5, 13, 181, 360).astype(np.float32)
    target = np.random.randn(5, 13, 181, 360).astype(np.float32)
    
    # 地表数据: [变量, 纬度, 经度]
    # 4个变量: u10, v10, t2m, msl
    input_surface = np.random.randn(4, 181, 360).astype(np.float32)
    target_surface = np.random.randn(4, 181, 360).astype(np.float32)
    
    # 添加一些空间相关性，使数据看起来更真实
    for i in range(5):
        for j in range(13):
            # 创建平滑的随机场
            smooth_field = create_smooth_field(181, 360)
            input_data[i, j] = smooth_field + input_data[i, j] * 0.2
            target[i, j] = smooth_field + target[i, j] * 0.2
    
    for i in range(4):
        smooth_field = create_smooth_field(181, 360)
        input_surface[i] = smooth_field + input_surface[i] * 0.2
        target_surface[i] = smooth_field + target_surface[i] * 0.2
    
    return input_data, input_surface, target, target_surface

def create_smooth_field(height, width, feature_size=20):
    """创建平滑的随机场"""
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # 创建几个随机正弦波
    field = np.zeros((height, width))
    for _ in range(5):
        freq_x = np.random.uniform(1, 5)
        freq_y = np.random.uniform(1, 5)
        phase_x = np.random.uniform(0, 2*np.pi)
        phase_y = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(0.5, 2.0)
        field += amplitude * np.sin(2*np.pi*freq_x*X + phase_x) * np.sin(2*np.pi*freq_y*Y + phase_y)
    
    return field