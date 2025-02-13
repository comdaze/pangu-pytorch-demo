# pip install onnx2pytorch
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch

import onnx.numpy_helper
import onnx
from onnx2pytorch import ConvertModel

from models.pangu_model import PanguModel


pytorch_model = PanguModel()
print('pytorch_model:', pytorch_model)

# # 设置模型为评估模式
# pytorch_model.eval()

# # 创建示例输入

# dummy_input = torch.randn(1, 5, 13, 721, 1440)
# dummy_input_surface = torch.randn(1, 4, 721, 1440)  # 假设 input_surface 的形状
# dummy_statistics = (torch.randn(4), torch.randn(4), torch.randn(13, 1, 1, 5), torch.randn(13, 1, 1, 5))              # 假设 statistics 的形状
# dummy_maps = torch.randn(1, 3, 724, 1440)           # 假设 maps 的形状
# dummy_const_h = torch.randn(1, 1, 1, 13, 721, 1440)                  # 假设 const_h 的形状

# # 将所有输入打包为一个元组
# dummy_input_all = (dummy_input, dummy_input_surface, dummy_statistics, dummy_maps, dummy_const_h)

# # 定义导出的 ONNX 文件路径
# onnx_file_path = "PanguModel.onnx"

# # 将模型导出为 ONNX 格式
# torch.onnx.export(
#     pytorch_model,                  # 要导出的模型
#     dummy_input_all,            # 示例输入
#     onnx_file_path,         # 导出的 ONNX 文件路径
#     export_params=True,     # 是否导出模型参数
#     opset_version=11,       # ONNX 算子集版本（建议使用 11 或更高版本）
#     do_constant_folding=True,  # 是否进行常量折叠优化
#     input_names=["input"],  # 输入节点的名称
#     output_names=["output"],  # 输出节点的名称
#     dynamic_axes={          # 如果需要支持动态输入/输出，可以指定动态轴
#         "input": {0: "batch_size"},  # 输入的第 0 维是动态的（batch_size）
#         "output": {0: "batch_size"},  # 输出的第 0 维是动态的（batch_size）
#     }
# )

# print(f"模型已成功导出为 {onnx_file_path}")

# # pip install onnx-simplifier
# # from onnxsim import simplify
# onnx_model_naive = onnx.load('PanguModel.onnx')
# # onnx_model_naive, check = simplify(onnx_model_naive)
# converted_model_naive = ConvertModel(onnx_model_naive)
# onnx_state_dict_naive = converted_model_naive.state_dict()
# print('onnx_state_dict_naive:', len(onnx_state_dict_naive.keys()), onnx_state_dict_naive.keys())


# 加载 ONNX 模型
onnx_model_path = '/opt/dlami/nvme/pretrained_model/pangu_weather_1.onnx'  # 1, 3, 6, 24
onnx_model = onnx.load(onnx_model_path)

# 转换为 PyTorch 模型
converted_model = ConvertModel(onnx_model)
# print('converted_model:', converted_model)

# 获取ONNX模型的状态字典
onnx_state_dict = converted_model.state_dict()
print('onnx_state_dict:', len(onnx_state_dict.keys()), onnx_state_dict.keys())

# 获取PyTorch模型的状态字典
pytorch_state_dict = pytorch_model.state_dict()
print('pytorch_state_dict:', len(pytorch_state_dict.keys()), pytorch_state_dict.keys())

# 创建一个新的状态字典来存储映射后的参数
new_state_dict = {}
torch_names = []
matching_names = []
matching_shapes = []
for name, param in pytorch_state_dict.items():
    torch_names.append(name)
    if name in onnx_state_dict:
        # 如果名称完全匹配,直接使用
        print('Exactly matched:', name)
        new_state_dict[name] = onnx_state_dict[name]
    else:
        # 尝试通过形状匹配
        matching_name = None
        matching_param = None
        for onnx_name, onnx_param in onnx_state_dict.items():
            if param.shape == onnx_param.shape and onnx_name not in matching_names:
                if name.endswith('bias'):
                    if onnx_name.endswith('bias'):
                        matching_param = onnx_param
                        matching_name = onnx_name
                        matching_names.append(matching_name)
                        matching_shapes.append(onnx_param.shape)
                        break
                    elif name.endswith('earth_specific_bias'):
                        matching_param = onnx_param
                        matching_name = onnx_name
                        matching_names.append(matching_name)
                        matching_shapes.append(onnx_param.shape)
                        break
                elif name.endswith('weight'):
                    if onnx_name.endswith('weight'):
                        matching_param = onnx_param
                        matching_name = onnx_name
                        matching_names.append(matching_name)
                        matching_shapes.append(onnx_param.shape)
                        break
                else:
                    matching_param = onnx_param
                    matching_name = onnx_name
                    matching_names.append(matching_name)
                    matching_shapes.append(onnx_param.shape)
                    break

        if matching_param is not None:
            new_state_dict[name] = matching_param
            # print('Shape matched:', name, 'matching_name:', matching_name)
        else:
            print(f"Warning: No matching parameter found for {name}")
            new_state_dict[name] = param  # 保留原始参数

import pandas as pd
pd.DataFrame({'torch_name': torch_names, 'onnx_name': matching_names, 'shape': matching_shapes}).to_csv('my_keys_all.csv', index=False)

# 加载新的状态字典到PyTorch模型
pytorch_model.load_state_dict(new_state_dict, strict=True)

torch.save({'model': pytorch_model.state_dict()},
           onnx_model_path[:-5]+'_torch.pth')
