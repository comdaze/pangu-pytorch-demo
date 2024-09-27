import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tensorboardX import SummaryWriter
import copy
import logging
import time
import argparse

import torch
from torch import nn
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from peft import LoraConfig, get_peft_model

from models.pangu_model import PanguModel
from models.pangu_sample import test, train
from era5_data.config import cfg
from era5_data.utils_dist import get_dist_info, init_dist
from era5_data import utils, utils_data


"""
Finetune the model using parameter-efficient finetune (lora)
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_net', type=str, default="loratuner_normout")
    parser.add_argument('--load_pretrained', type=bool, default=False)
    parser.add_argument('--load_my_best', type=bool, default=True)
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    # parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--dist', default=True)
    args = parser.parse_args()
    starts = time.time()

    PATH = cfg.PG_INPUT_PATH

    # opt = {"gpu_ids": [0]}
    # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # # gpu_list = str(opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')

    # print(f"Predicting on {device}")
    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if args.dist:
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
                                         horizon=cfg.PG.HORIZON)
    if args.dist:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, drop_last=True)

        train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=cfg.PG.TRAIN.BATCH_SIZE//world_size,  # replace len(opt['gpu_ids']) with world_size
                                           num_workers=8, pin_memory=False, sampler=train_sampler)  # default: num_workers=0
    else:
        train_dataloader = data.DataLoader(dataset=train_dataset,
                                           batch_size=cfg.PG.TRAIN.BATCH_SIZE,
                                           drop_last=True, shuffle=True, num_workers=8, pin_memory=False)  # default: num_workers=0

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
                                       horizon=cfg.PG.HORIZON)

    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=cfg.PG.VAL.BATCH_SIZE,
                                     drop_last=True, shuffle=False, num_workers=8, pin_memory=False)  # default: num_workers=0

    # test_dataset = utils_data.NetCDFDataset(nc_path=PATH,
    test_dataset = utils_data.PTDataset(pt_path=PATH,
                                        data_transform=None,
                                        training=False,
                                        validation=False,
                                        startDate=cfg.PG.TEST.START_TIME,
                                        endDate=cfg.PG.TEST.END_TIME,
                                        freq=cfg.PG.TEST.FREQUENCY,
                                        horizon=cfg.PG.HORIZON)

    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=cfg.PG.TEST.BATCH_SIZE,
                                      drop_last=True, shuffle=False, num_workers=8, pin_memory=False)  # default: num_workers=0

    model = PanguModel(device=device).to(device)
    
    if cfg.PG.HORIZON == 1:
        checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_1_torch)
    elif cfg.PG.HORIZON == 3:
        checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_3_torch)
    elif cfg.PG.HORIZON == 6:
        checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_6_torch)
    elif cfg.PG.HORIZON == 24:
        checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch)
    else:
        print('cfg.PG.HORIZON:', cfg.PG.HORIZON, 'NO CHECKPOINT FOUND')
    model.load_state_dict(checkpoint['model'])

    print([(n, type(m)) for n, m in model.named_modules()])
    target_modules = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            target_modules.append(n)
            print(f"appended {n}")
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        modules_to_save=["_output_layer.conv_surface", "_output_layer.conv"]
    )

    # we keep a copy of the original model for later
    module_copy = copy.deepcopy(model)

    peft_model = get_peft_model(model, config)
    optimizer = torch.optim.Adam(peft_model.parameters(
    ), lr=cfg.PG.TRAIN.LR, weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50], gamma=0.5)
    start_epoch = 1
    if args.load_pretrained:
        cpk = torch.load(os.path.join(output_path, "models/train_30.pth"))
        peft_model.load_state_dict(cpk['model'])
        optimizer.load_state_dict(cpk['optimizer'])
        lr_scheduler.load_state_dict(cpk['lr_scheduler'])
        start_epoch = cpk["epoch"]

    peft_model = DDP(peft_model)  # Use DistributedDataParallel
    
    peft_model = train(peft_model, train_loader=train_dataloader,
                       val_loader=val_dataloader,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler,
                       res_path=output_path,
                       device=device,
                       writer=writer, logger=logger, start_epoch=start_epoch, rank=rank)

    if rank == 0:
        for name, param in peft_model.base_model.named_parameters():
            if "lora" not in name:
                continue

            print(
                f"New parameter {name:<13} | {param.numel():>5} parameters | updated")

        params_before = dict(module_copy.named_parameters())
        for name, param in peft_model.base_model.named_parameters():
            if "lora" in name:
                continue

            name_before = name.partition(".")[-1].replace("original_", "").replace("module.", "").replace(
                "modules_to_save.default.", "")
            if name_before in params_before:  # TODO in case name_before not in params_before, e.g., 'downsample.linear.base_layer.weight'
                param_before = params_before[name_before]
                if torch.allclose(param, param_before):
                    print(
                        f"Parameter {name_before:<13} | {param.numel():>7} parameters | not updated")
                else:
                    print(
                        f"Parameter {name_before:<13} | {param.numel():>7} parameters | updated")

        output_path = os.path.join(output_path, "test")
        utils.mkdirs(output_path)

        test(test_loader=test_dataloader,
             model=peft_model,
             device=device,
             res_path=output_path)

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 finetune_lastLayer_ddp.py --dist True
