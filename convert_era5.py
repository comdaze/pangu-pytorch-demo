import os
import math
import numpy as np
import pandas as pd

import xarray

from typing import Optional

from datetime import datetime, timedelta
import calendar

from tqdm import tqdm
import multiprocessing as mp

import torch

def get_last_day_of_month(date_string):
    # 将字符串转换为 datetime 对象
    date = datetime.strptime(date_string, '%Y%m%d')
    
    # 获取下个月的第一天
    if date.month == 12:
        next_month = datetime(date.year + 1, 1, 1)
    else:
        next_month = datetime(date.year, date.month + 1, 1)
    
    # 下个月第一天减去一天，就是本月最后一天
    last_day = next_month - timedelta(days=1)
    
    # 返回天数作为字符串
    return f'{last_day.day:02d}'

def process_month(select_month):
    select_month_end = get_last_day_of_month(select_month+'01')
    print(select_month, select_month_end)
    
    # if os.path.exists(f'surface/surface_{select_month}.nc'):
    #     continue
    
    msl_ds = xarray.open_dataset(f'nsf-ncar-era5/e5.oper.an.sfc/{select_month}/e5.oper.an.sfc.128_151_msl.ll025sc.{select_month}0100_{select_month}{select_month_end}23.nc')
    u10_ds = xarray.open_dataset(f'nsf-ncar-era5/e5.oper.an.sfc/{select_month}/e5.oper.an.sfc.128_165_10u.ll025sc.{select_month}0100_{select_month}{select_month_end}23.nc')
    v10_ds = xarray.open_dataset(f'nsf-ncar-era5/e5.oper.an.sfc/{select_month}/e5.oper.an.sfc.128_166_10v.ll025sc.{select_month}0100_{select_month}{select_month_end}23.nc')
    t2m_ds = xarray.open_dataset(f'nsf-ncar-era5/e5.oper.an.sfc/{select_month}/e5.oper.an.sfc.128_167_2t.ll025sc.{select_month}0100_{select_month}{select_month_end}23.nc')

    surface_ds = xarray.merge([msl_ds.rename({'MSL': 'msl'}), u10_ds.rename({'VAR_10U': 'u10'}), v10_ds.rename({'VAR_10V': 'v10'}), t2m_ds.rename({'VAR_2T': 't2m'})])
    # surface_ds.to_netcdf(f'surface/surface_{select_month}.nc')
    
    for d in range(1, int(select_month_end)+1):
        if d < 10:
            d = '0'+str(d)
        else:
            d = str(d)
        select_date = select_month+d
        for h in range(24):
            if h < 10:
                h = '0'+str(h)
            else:
                h = str(h)
            select_hour = select_date+h
            select_hour_datetime = pd.to_datetime(select_hour, format='%Y%m%d%H')
            select_surface_ds = surface_ds.sel(time=select_hour_datetime)
            surface_np = select_surface_ds[['msl', 'u10', 'v10', 't2m']].to_array().values
            # np.save(f'surface/surface_{select_hour}.npy', surface_np)
            surface_tensor = torch.from_numpy(surface_np)
            torch.save(surface_tensor, f'surface/surface_{select_hour}.pt')

def process_date(select_date):
    select_month = select_date[:6]
    select_month_end = get_last_day_of_month(select_date)
    print(select_date, select_month, select_month_end)

    z_ds = xarray.open_dataset(f'nsf-ncar-era5/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_129_z.ll025sc.{select_date}00_{select_date}23.nc')
    q_ds = xarray.open_dataset(f'nsf-ncar-era5/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_133_q.ll025sc.{select_date}00_{select_date}23.nc')
    t_ds = xarray.open_dataset(f'nsf-ncar-era5/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_130_t.ll025sc.{select_date}00_{select_date}23.nc')
    u_ds = xarray.open_dataset(f'nsf-ncar-era5/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_131_u.ll025uv.{select_date}00_{select_date}23.nc')
    v_ds = xarray.open_dataset(f'nsf-ncar-era5/e5.oper.an.pl/{select_month}/e5.oper.an.pl.128_132_v.ll025uv.{select_date}00_{select_date}23.nc')

    z_ds = z_ds.sel(level=pressure_levels)
    q_ds = q_ds.sel(level=pressure_levels)
    t_ds = t_ds.sel(level=pressure_levels)
    u_ds = u_ds.sel(level=pressure_levels)
    v_ds = v_ds.sel(level=pressure_levels)

    upper_ds = xarray.merge([z_ds.rename({'Z': 'z'}), q_ds.rename({'Q': 'q'}), t_ds.rename({'T': 't'}), u_ds.rename({'U': 'u'}), v_ds.rename({'V': 'v'})])
    # upper_ds.to_netcdf(f'upper/upper_{select_date}.nc')
    
    for h in range(24):
        if h < 10:
            h = '0'+str(h)
        else:
            h = str(h)
        select_hour = select_date+h
        select_hour_datetime = pd.to_datetime(select_hour, format='%Y%m%d%H')
        select_upper_ds = upper_ds.sel(time=select_hour_datetime)
        upper_np = select_upper_ds[['z', 'q', 't', 'u', 'v']].to_array().values
        # np.save(f'upper/upper_{select_hour}.npy', upper_np)
        upper_tensor = torch.from_numpy(upper_np)
        torch.save(upper_tensor, f'upper/upper_{select_hour}.pt')

pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
startDate= '20180101'
endDate = '20180131'  # '20191231'
select_dates = list(pd.date_range(start=startDate, end=endDate, freq='1D'))
select_dates = [date.strftime('%Y%m%d') for date in select_dates]
# select_months = set([select_date[:6] for select_date in select_dates])
select_months = list(pd.date_range(start=startDate, end=endDate, freq='1ME'))
select_months = [date.strftime('%Y%m') for date in select_months]

os.system('mkdir -p surface')
os.system('mkdir -p upper')

# 设置进程数，可以根据你的CPU核心数进行调整
num_processes = 1  # mp.cpu_count()  # 使用所有可用的CPU核心

# # 使用进程池并行处理
# with mp.Pool(num_processes) as pool:
#     list(tqdm(pool.imap(process_month, select_months), total=len(select_months)))

# 使用进程池并行处理
with mp.Pool(num_processes) as pool:
    list(tqdm(pool.imap(process_date, select_dates), total=len(select_dates)))

