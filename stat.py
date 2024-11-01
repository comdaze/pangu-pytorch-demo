import os

import warnings
# 忽略所有FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

from tensorboardX import SummaryWriter
import logging
import time
import argparse
from tqdm import tqdm

import torch
from torch.utils import data

from era5_data.config import cfg
from era5_data import utils, utils_data


class RunningStats:
    def __init__(self, num_channels):
        self.sum = torch.zeros(num_channels)
        self.wind_speed_sum = 0
        self.count = 0

    def update(self, tensor):
        # tensor shape: [batch, channels, height, width]
        # 提取U10和V10
        U10 = tensor[:, 2, :, :]  # [1, 721, 1440]
        V10 = tensor[:, 3, :, :]  # [1, 721, 1440]
        # 计算风速 sqrt(U10² + V10²)
        wind_speed = torch.sqrt(U10**2 + V10**2)  # [1, 721, 1440]
        # 计算风速的平均值
        mean_wind_speed = wind_speed.mean()
        self.wind_speed_sum += mean_wind_speed
        self.sum += tensor.mean(dim=[0, 2, 3]).cpu()
        self.count += 1

    def get_means(self):
        return self.sum / self.count, self.wind_speed_sum / self.count
    
"""
Fully finetune the pretrained model
"""
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


    stats = RunningStats(4)
    batch_id = 0
    # for id, data in enumerate(test_loader, 0):
    for data in tqdm(test_dataloader, desc='Testing'):
        # Store initial input for different models
        # print(f"predict on {id}")
        input_test, input_surface_test, target_test, target_surface_test, periods_test = data
        # print('target_surface_test.shape:', target_surface_test.shape)
        stats.update(target_surface_test)
        
    means = stats.get_means()
    print('surface means:', means)
