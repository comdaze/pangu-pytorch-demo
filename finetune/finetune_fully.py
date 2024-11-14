import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import warnings
# 忽略所有FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

from tensorboardX import SummaryWriter
import logging
import time
import argparse

import torch
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from models.pangu_sample import test, train, monitor_system
from models.pangu_model import PanguModel
from era5_data.config import cfg
from era5_data.utils_dist import get_dist_info, init_dist
from era5_data import utils, utils_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
"""
Fully finetune the pretrained model
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_net', type=str, default="finetune_fully")
    parser.add_argument('--load_pretrained', type=str2bool, default=False)
    parser.add_argument('--load_my_best', type=str2bool, default=True)
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dist', type=str2bool, default=True)
    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--visualize', type=str2bool, default=False)
    parser.add_argument('--only_use_wind_speed_loss', type=str2bool, default=False)
    parser.add_argument('--use_deepspeed', type=str2bool, default=False)

    args = parser.parse_args()
    starts = time.time()

    PATH = cfg.PG_INPUT_PATH

    # opt = {"gpu_ids": [0]}
    # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # # gpu_list = str(opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')

    torch.set_num_threads(cfg.GLOBAL.NUM_THREADS)

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if args.use_deepspeed:
        import deepspeed
        deepspeed.init_distributed(dist_backend='nccl')
    elif args.dist:
        init_dist(args.launcher, backend='nccl')
    rank, world_size = get_dist_info()
    print("The rank and world size is", rank, world_size)
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    local_rank = rank % gpu_count
    print('local_rank:', local_rank)
    device = torch.device('cuda:' + str(local_rank))
    print(f"Predicting on {device}")

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

    # train_dataset = utils_data.NetCDFDataset(nc_path=PATH,
    train_dataset = utils_data.PTDataset(pt_path=PATH,
                                         data_transform=None,
                                         training=True,
                                         validation=False,
                                         startDate=cfg.PG.TRAIN.START_TIME,
                                         endDate=cfg.PG.TRAIN.END_TIME,
                                         freq=cfg.PG.TRAIN.FREQUENCY,
                                         horizon=cfg.PG.HORIZON,
                                         device='cpu')  # device
    if args.dist:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, drop_last=True)

        train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=cfg.PG.TRAIN.BATCH_SIZE//world_size,  # replace len(opt['gpu_ids']) with world_size
                                           num_workers=args.num_workers, prefetch_factor=2, pin_memory=True, sampler=train_sampler)  # default: num_workers=0, pin_memory=False
    else:
        train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=cfg.PG.TRAIN.BATCH_SIZE,
                                           drop_last=True, shuffle=True, num_workers=args.num_workers, prefetch_factor=2, pin_memory=True)  # default: num_workers=0, pin_memory=False

    dataset_length = len(train_dataloader)
    if rank == 0:
        print("dataset_length", dataset_length)

    # val_dataset = utils_data.NetCDFDataset(nc_path=PATH,
    val_dataset = utils_data.PTDataset(pt_path=PATH,
                                       data_transform=None,
                                       training=False,
                                       validation=True,
                                       startDate=cfg.PG.VAL.START_TIME,
                                       endDate=cfg.PG.VAL.END_TIME,
                                       freq=cfg.PG.VAL.FREQUENCY,
                                       horizon=cfg.PG.HORIZON,
                                       device='cpu')  # device

    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=cfg.PG.VAL.BATCH_SIZE,
                                     drop_last=True, shuffle=False, num_workers=args.num_workers, prefetch_factor=2, pin_memory=True)  # default: num_workers=0, pin_memory=False

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

    model = PanguModel(device=device).to(device)

    if cfg.PG.HORIZON == 1:
        checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_1_torch, weights_only=True)
    elif cfg.PG.HORIZON == 3:
        checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_3_torch, weights_only=True)
    elif cfg.PG.HORIZON == 6:
        checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_6_torch, weights_only=True)
    elif cfg.PG.HORIZON == 24:
        checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, weights_only=True)
    else:
        print('cfg.PG.HORIZON:', cfg.PG.HORIZON, 'NO CHECKPOINT FOUND')
    model.load_state_dict(checkpoint['model'])
    
    # Fully finetune
    for param in model.parameters():
        param.requires_grad = True
        
    if args.use_deepspeed:
        # 定义参数组
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        # DeepSpeed初始化
        ds_config = "ds_config.json"  # 配置文件路径
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=parameters,
            config=ds_config
        )
        start_epoch = 1
        
        # # TODO: 加载之前的检查点（如果需要）
        # if args.load_pretrained:
        #     _, client_sd = model.load_checkpoint(output_path, tag="train_57")
        #     start_epoch = client_sd['epoch'] + 1
        # else:
        #     start_epoch = 1
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
        )), lr=cfg.PG.TRAIN.LR, weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[25, 50], gamma=0.5)
        start_epoch = 1
        
        if args.load_pretrained:
            cpk = torch.load(os.path.join(output_path, "models/train_57.pth"), weights_only=True)
            cpk['model'] = {k.replace("module.", ""): v for k, v in cpk['model'].items()}
            model.load_state_dict(cpk['model'])
            optimizer.load_state_dict(cpk['optimizer'])
            lr_scheduler.load_state_dict(cpk['lr_scheduler'])
            start_epoch = cpk["epoch"]+1
            del cpk
            torch.cuda.empty_cache()
        
        model = DDP(model)  # Use DistributedDataParallel
        
    if rank == 0:
        msg = '\n'
        msg += utils.torch_summarize(model, show_weights=False)
        logger.info(msg)

    # weather_statistics = utils.LoadStatic_pretrain()
    # if rank == 0:
    #     print("weather statistics are loaded!")

    if not args.only_test:
        model = train(model, train_loader=train_dataloader,
                    val_loader=val_dataloader,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    res_path=output_path,
                    device=device,
                    writer=writer, logger=logger, start_epoch=start_epoch, rank=rank, visualize=args.visualize, only_use_wind_speed_loss=args.only_use_wind_speed_loss, use_deepspeed=args.use_deepspeed)

    if rank == 0:
        print('args:', args)
        if args.load_my_best:
            print('load_my_best')
            if args.use_deepspeed:
                best_model = PanguModel(device=device).to(device)
                best_model.load_state_dict(torch.load(os.path.join(output_path, "models/best_model.pth"), weights_only=True))
            else:
                best_model = torch.load(os.path.join(
                    output_path, "models/best_model.pth"), weights_only=False, map_location=device)  # 'cuda:0'
        else:
            best_model = model

        logger.info("Begin testing...")

        test(test_loader=test_dataloader,
             model=best_model,
             device=device,
             res_path=output_path,
             visualize=args.visualize,
             only_use_wind_speed_loss=args.only_use_wind_speed_loss)

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 finetune_lastLayer_ddp.py --dist True
