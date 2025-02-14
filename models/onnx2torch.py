"""
Pseudocode for converting the onnx weights to torch weights
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import torch

import onnx.numpy_helper as np_helper
import onnx

from models.pangu_model import PanguModel
from era5_data.config import cfg

# Match between onnx key and torch key
PATH = './'
lookUpTable = os.path.join(PATH, 'keys_all.csv')
lookUpTable = pd.read_csv(lookUpTable)
# Load onnx file of pretrained pangu model
# onnx_model_path = cfg.PG.BENCHMARK.PRETRAIN_24
# horizon = 24
# onnx_model_path = cfg.PG.BENCHMARK.PRETRAIN_6
# horizon = 6
# onnx_model_path = cfg.PG.BENCHMARK.PRETRAIN_3
# horizon = 3
onnx_model_path = cfg.PG.BENCHMARK.PRETRAIN_1
horizon = 1

onnx_model = onnx.load(onnx_model_path)

graph = onnx_model.graph
INTIALIZERS = graph.initializer
onnx_weights = {}
for initializer in INTIALIZERS:
     W = np_helper.to_array(initializer)
     onnx_weights[initializer.name] = W
print('onnx_weights:', onnx_weights.keys())

os.makedirs(os.path.join(PATH, 'aux_data'), exist_ok=True)
for node in graph.node:
    # print(f"Node name: {node.name}")
    # print(f"Node inputs: {node.input}")
    # print(f"Node outputs: {node.output}")
    # for attr in node.attribute:
        # print(f"Attribute name: {attr.name}")
        # if attr.type == onnx.AttributeProto.FLOAT:
        #     print(f"Attribute value (float): {attr.f}")
        # elif attr.type == onnx.AttributeProto.INTS:
        #     print(f"Attribute value (ints): {attr.ints}")
        # elif attr.type == onnx.AttributeProto.TENSOR:
        #     tensor = onnx.numpy_helper.to_array(attr.t)
        #     print(f"Attribute value (tensor): {tensor}")
    if node.name == '/b1/Constant_11':
        for attr in node.attribute:
            if attr.name == 'value':
                surface_mean = onnx.numpy_helper.to_array(attr.t)
                np.save(os.path.join(PATH, 'aux_data/surface_mean.npy'), surface_mean)
    elif node.name == '/b1/Constant_12':
        for attr in node.attribute:
            if attr.name == 'value':
                surface_std = onnx.numpy_helper.to_array(attr.t)
                np.save(os.path.join(PATH, 'aux_data/surface_std.npy'), surface_std)
    elif node.name == '/b1/Constant_9':
        for attr in node.attribute:
            if attr.name == 'value':
                upper_mean = onnx.numpy_helper.to_array(attr.t)
                np.save(os.path.join(PATH, 'aux_data/upper_mean.npy'), upper_mean)
    elif node.name == '/b1/Constant_10':
        for attr in node.attribute:
            if attr.name == 'value':
                upper_std = onnx.numpy_helper.to_array(attr.t)
                np.save(os.path.join(PATH, 'aux_data/upper_std.npy'), upper_std)
    elif node.name == '/b1/Constant_44':
        for attr in node.attribute:
            if attr.name == 'value':
                maps = onnx.numpy_helper.to_array(attr.t)
                np.save(os.path.join(PATH, f'aux_data/constantMask{horizon}.npy'), maps)
    elif node.name == '/b1/Constant_17':
        for attr in node.attribute:
            if attr.name == 'value':
                const_h = onnx.numpy_helper.to_array(attr.t)
                np.save(os.path.join(PATH, 'aux_data/Constant_17_output_0.npy'), const_h)

def compare_npy(file1, file2):
    # 加载 .npy 文件
    array1 = np.load(file1)
    array2 = np.load(file2)

    # 逐元素比较
    if np.array_equal(array1, array2):
        print("两个 .npy 文件完全一致")
        return True
    else:
        print("两个 .npy 文件不一致")
        print(array1.shape, array2.shape)
        print('array1:', array1)
        print('array2:', array2)
        return False

compare_npy(os.path.join(PATH, 'aux_data/surface_mean.npy'), '/opt/dlami/nvme/aux_data/surface_mean.npy')
compare_npy(os.path.join(PATH, 'aux_data/surface_std.npy'), '/opt/dlami/nvme/aux_data/surface_std.npy')
compare_npy(os.path.join(PATH, 'aux_data/upper_mean.npy'), '/opt/dlami/nvme/aux_data/upper_mean.npy')
compare_npy(os.path.join(PATH, 'aux_data/upper_std.npy'), '/opt/dlami/nvme/aux_data/upper_std.npy')
compare_npy(os.path.join(PATH, f'aux_data/constantMask{horizon}.npy'), '/opt/dlami/nvme/aux_data/constantMaks3.npy')
compare_npy(os.path.join(PATH, 'aux_data/Constant_17_output_0.npy'), '/opt/dlami/nvme/aux_data/Constant_17_output_0.npy')

# Pangu model in pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PanguModel(device=device).to(device)

torch_keys = lookUpTable["torch_name"]
print('torch_keys:', len(torch_keys))

count = 0
print("Load pretrain with bias")
# Load the onnx weight to pangu model layer by layer
for name, param in model.named_parameters():
    if param.requires_grad:
    #    print(count, name)
    #    print(torch_keys[count])

       row = lookUpTable[lookUpTable['torch_name'] == name]
       if row.empty:
           print("no record torch key ", name)
       onnx_name = row['onnx_name'].values[0]
       if isinstance(onnx_name, str):
          w  = torch.tensor(onnx_weights[onnx_name])
        #   print("shapes", param.data.shape, w.shape)
          if len(param.data.shape) == 1:
              assert param.data.shape == w.shape
              param.data = w.clone().to(device)
              param.requires_grad = False
              count += 1
          elif len(param.data.shape) == 2:
              assert param.data.shape == w.T.shape
              param.data = w.T.clone().to(device)
              param.requires_grad = False
              count += 1
          elif len(param.data.shape) == 3:
              assert param.data.shape == w.shape
              param.data = w.clone().to(device)
              param.requires_grad = False
              count += 1
          elif len(param.data.shape) == 4:
              print('len(param.data.shape) == 4')
              assert param.data.shape == w.shape
              param.data = w.clone().to(device)
              param.requires_grad = False
              count += 1
          elif len(param.data.shape) == 5:
              assert param.data.shape == w.shape
              param.data = w.clone().to(device)
              param.requires_grad = False
              count += 1
print('count:', count)
# Save the torch weights
# output_path = '/opt/dlami/nvme/'
# torch.save(model,os.path.join(output_path,"onnx2torch.pth"))
torch.save({'model': model.state_dict()},
           onnx_model_path[:-5]+'_torch.pth')
