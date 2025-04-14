import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datetime import datetime, timedelta
from tqdm import tqdm
import onnxruntime as ort
import onnx
import numpy as np
from torch.utils import data
from torch import nn
import torch
import gc

from era5_data.config import cfg
from era5_data import score
from era5_data import utils, utils_data

# The directory of your input and output data
PATH = cfg.PG_INPUT_PATH
output_data_dir = cfg.PG_OUT_PATH

h = cfg.PG.HORIZON
output_data_dir = os.path.join(output_data_dir, str(h))
utils.mkdirs(output_data_dir)

# Prepare for the test data
# test_dataset = utils_data.NetCDFDataset(nc_path=PATH,
test_dataset = utils_data.PTDataset(pt_path=PATH,
                                    data_transform=None,
                                    training=False,
                                    validation=False,
                                    startDate=cfg.PG.TEST.START_TIME,
                                    endDate=cfg.PG.TEST.END_TIME,
                                    freq=cfg.PG.TEST.FREQUENCY,
                                    horizon=h,
                                    device='cpu')
dataset_length = len(test_dataset)
print("dataset_length", dataset_length)

test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=cfg.PG.TEST.BATCH_SIZE,
                                  drop_last=True, shuffle=False, num_workers=1, pin_memory=False)  # default: num_workers=0

# # Load pretrained model
# model_24 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_24)
# model_6 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_6)
# model_3 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_3)
# model_1 = onnx.load(cfg.PG.BENCHMARK.PRETRAIN_1)

# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = cfg.GLOBAL.NUM_THREADS

# # Set the behavier of cuda provider
# cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}
# providers = [('CUDAExecutionProvider', cuda_provider_options)]

# 为每个模型配置不同的GPU
providers_24 = [('CUDAExecutionProvider', {'device_id': 0})]
providers_6 = [('CUDAExecutionProvider', {'device_id': 1})]
providers_3 = [('CUDAExecutionProvider', {'device_id': 2})]
providers_1 = [('CUDAExecutionProvider', {'device_id': 3})]

# Initialize onnxruntime session for Pangu-Weather Models
ort_session_24 = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_24, sess_options=options, providers=providers_24)
ort_session_6 = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_6, sess_options=options, providers=providers_6)
ort_session_3 = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_3, sess_options=options, providers=providers_3)
ort_session_1 = ort.InferenceSession(cfg.PG.BENCHMARK.PRETRAIN_1, sess_options=options, providers=providers_1)

# Dic to save rmses
rmse_upper_z, rmse_upper_q, rmse_upper_t, rmse_upper_u, rmse_upper_v = dict(
), dict(), dict(), dict(), dict()
rmse_surface = dict()

acc_upper_z, acc_upper_q, acc_upper_t, acc_upper_u, acc_upper_v = dict(
), dict(), dict(), dict(), dict()
acc_surface = dict()

# Loss function
criterion = nn.L1Loss(reduction='none')

# Load constants and teleconnection indices
aux_constants = utils_data.loadAllConstants(
    device='cpu')  # 'weather_statistics','weather_statistics_last','constant_maps','tele_indices','variable_weights'
upper_weights, surface_weights, upper_loss_weight, surface_loss_weight = aux_constants['variable_weights']

test_loss = 0.0

def save_prediction(output, output_surface, hour, prediction_time, save_dir):
    """保存预测结果到磁盘"""
    # 创建以日期命名的子目录
    date_dir = os.path.join(save_dir, prediction_time.strftime('%Y%m%d'))
    os.makedirs(date_dir, exist_ok=True)
    
    # 保存文件名格式：YYYYMMDD_HH.npz
    save_path = os.path.join(date_dir, f"{prediction_time.strftime('%Y%m%d_%H')}.npz")
    
    # 将tensor转移到CPU并转换为numpy数组
    output_cpu = output.cpu().numpy()
    output_surface_cpu = output_surface.cpu().numpy()
    
    # 保存为压缩的npz文件
    np.savez_compressed(
        save_path,
        output=output_cpu,
        output_surface=output_surface_cpu,
        hour=hour,
        timestamp=prediction_time.strftime('%Y%m%d%H')
    )
    
    # 释放内存
    del output_cpu, output_surface_cpu
    gc.collect()
    
    return save_path
  
def load_prediction(save_path, device='cuda'):
    """从磁盘加载预测结果"""
    data = np.load(save_path)
    output = torch.from_numpy(data['output']).to(device)
    output_surface = torch.from_numpy(data['output_surface']).to(device)
    return output, output_surface

def inference_step(ort_session, input, input_surface):
    # 确保输入数据在CPU上
    if isinstance(input, torch.Tensor):
        input = input.cpu()
    if isinstance(input_surface, torch.Tensor):
        input_surface = input_surface.cpu()
        
    input, input_surface = input.numpy().astype(np.float32).squeeze(), input_surface.numpy().astype(np.float32).squeeze()

    # 执行推理
    output, output_surface = ort_session.run(
        None, {'input': input, 'input_surface': input_surface})
    
    return torch.from_numpy(output).type(torch.float32), torch.from_numpy(output_surface).type(torch.float32)

def inference_25_hours(input_time, input, input_surface, save_dir):
    # 存储每个时间点的预测结果
    prediction_files = {}  # 存储文件路径而不是实际数据

    # 1. 使用24h模型预测第24小时
    # print('h=24')
    output_24, output_surface_24 = inference_step(ort_session_24, input, input_surface)
    prediction_time = input_time + timedelta(hours=24)
    save_path = save_prediction(output_24, output_surface_24, 24, prediction_time, save_dir)
    prediction_files[24] = save_path
    # 释放GPU内存
    del output_24, output_surface_24
    torch.cuda.empty_cache()

    # 2. 使用6h模型预测30,36,42,48小时
    for h in [30, 36, 42, 48]:
        # print(f"h={h}")
        # 找到最近的已知预测结果作为输入
        latest_h = max([x for x in prediction_files.keys() if x <= h - 6])
        new_input, new_input_surface = load_prediction(prediction_files[latest_h])
        output_6, output_surface_6 = inference_step(ort_session_6, new_input, new_input_surface)
        prediction_time = input_time + timedelta(hours=h)
        save_path = save_prediction(output_6, output_surface_6, h, prediction_time, save_dir)
        prediction_files[h] = save_path
        # 释放GPU内存
        del new_input, new_input_surface, output_6, output_surface_6
        torch.cuda.empty_cache()

    # 3. 使用3h模型预测27,33,39,45小时
    for h in [27, 33, 39, 45]:
        # print(f"h={h}")
        latest_h = max([x for x in prediction_files.keys() if x <= h - 3])
        new_input, new_input_surface = load_prediction(prediction_files[latest_h])
        output_3, output_surface_3 = inference_step(ort_session_3, new_input, new_input_surface)
        prediction_time = input_time + timedelta(hours=h)
        save_path = save_prediction(output_3, output_surface_3, h, prediction_time, save_dir)
        prediction_files[h] = save_path
        # 释放GPU内存
        del new_input, new_input_surface, output_3, output_surface_3
        torch.cuda.empty_cache()

    # 4. 使用1h模型填充剩余时间点
    remaining_hours = [h for h in range(24, 49) if h not in prediction_files]
    for h in remaining_hours:
        # print(f"h={h}")
        latest_h = max([x for x in prediction_files.keys() if x <= h - 1])
        new_input, new_input_surface = load_prediction(prediction_files[latest_h])
        output_1, output_surface_1 = inference_step(ort_session_1, new_input, new_input_surface)
        prediction_time = input_time + timedelta(hours=h)
        save_path = save_prediction(output_1, output_surface_1, h, prediction_time, save_dir)
        prediction_files[h] = save_path
        # 释放GPU内存
        del new_input, new_input_surface, output_1, output_surface_1
        torch.cuda.empty_cache()

    return prediction_files

batch_id = 0
# 每天00:00 12:00预报h小时候的天气（single frame output）
# for id, data in tqdm(enumerate(test_dataloader, 0)):
for data in tqdm(test_dataloader):
    # Store initial input for different models
    input, input_surface, target, target_surface, periods = data
    # start time
    input_time = datetime.strptime(periods[0][batch_id], '%Y%m%d%H')
    print(f"input_time: {input_time}")
    save_dir = os.path.join(output_data_dir, 'predictions', periods[0][batch_id])
    
    prediction_files = inference_25_hours(input_time, input, input_surface, save_dir)
    
    # for k,v in prediction_files.items():
    #   print(k, v)