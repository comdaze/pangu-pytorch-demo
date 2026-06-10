"""
Convert the official Pangu-Weather ONNX models (1h/3h/6h) to torch .pth
checkpoints matching PanguModel, and extract each horizon's constant mask.

Mirrors models/onnx2torch.py but without the brittle reference comparisons,
and only for the horizons that still need converting.

Run:  LD_LIBRARY_PATH=/opt/conda/lib python models/convert_horizons.py
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

import numpy as np
import pandas as pd
import torch
import onnx
import onnx.numpy_helper as np_helper

from models.pangu_model import PanguModel
from era5_data.config import cfg

REPO = os.path.dirname(current_dir)
AUX = os.path.join(cfg.PG_INPUT_PATH, "aux_data")
lookUpTable = pd.read_csv(os.path.join(REPO, "keys_all.csv"))

HORIZONS = [
    (cfg.PG.BENCHMARK.PRETRAIN_1, 1),
    (cfg.PG.BENCHMARK.PRETRAIN_3, 3),
    (cfg.PG.BENCHMARK.PRETRAIN_6, 6),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for onnx_path, horizon in HORIZONS:
    out_path = onnx_path[:-5] + "_torch.pth"
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists")
        continue
    print(f"[convert] {onnx_path} -> {out_path} (horizon={horizon}h)")
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    onnx_weights = {}
    for initializer in graph.initializer:
        onnx_weights[initializer.name] = np_helper.to_array(initializer)

    # extract this horizon's constant mask (node /b1/Constant_44)
    for node in graph.node:
        if node.name == "/b1/Constant_44":
            for attr in node.attribute:
                if attr.name == "value":
                    maps = onnx.numpy_helper.to_array(attr.t)
                    np.save(os.path.join(AUX, f"constantMask{horizon}.npy"), maps)
                    print(f"   saved constantMask{horizon}.npy {maps.shape}")

    model = PanguModel(device=device).to(device)
    count = 0
    for name, param in model.named_parameters():
        row = lookUpTable[lookUpTable["torch_name"] == name]
        if row.empty:
            print("   no record torch key", name)
            continue
        onnx_name = row["onnx_name"].values[0]
        if isinstance(onnx_name, str):
            w = torch.tensor(onnx_weights[onnx_name])
            if len(param.data.shape) == 2:
                assert param.data.shape == w.T.shape
                param.data = w.T.clone().to(device)
            else:
                assert param.data.shape == w.shape
                param.data = w.clone().to(device)
            param.requires_grad = False
            count += 1
    print(f"   mapped {count} tensors")
    torch.save({"model": model.state_dict()}, out_path)
    print(f"   saved {out_path}")

print("done")
