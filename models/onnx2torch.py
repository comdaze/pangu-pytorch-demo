"""
Pseudocode for converting the onnx weights to torch weights
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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
# onnx_model_path = cfg.PG.BENCHMARK.PRETRAIN_6
# onnx_model_path = cfg.PG.BENCHMARK.PRETRAIN_3
onnx_model_path = cfg.PG.BENCHMARK.PRETRAIN_1
model_24 = onnx.load(onnx_model_path)

graph = model_24.graph
INTIALIZERS = graph.initializer
onnx_weights = {}
for initializer in INTIALIZERS:
     W = np_helper.to_array(initializer)
     onnx_weights[initializer.name] = W
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
