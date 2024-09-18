# pip install onnx2pytorch

import sys
sys.path.append("/home/ec2-user/pangu-pytorch")
from models.pangu_model import PanguModel

from onnx2pytorch import ConvertModel
import onnx
import onnx.numpy_helper
import torch


pytorch_model = PanguModel()
# print('pytorch_model:', pytorch_model)

# 加载 ONNX 模型
onnx_model_path = 'pretrained_model/pangu_weather_1.onnx'  # 1, 3, 6, 24
onnx_model = onnx.load(onnx_model_path)

# 转换为 PyTorch 模型
converted_model = ConvertModel(onnx_model)
# print('converted_model:', converted_model)

# 获取ONNX模型的状态字典
onnx_state_dict = converted_model.state_dict()
# print('onnx_state_dict:', onnx_state_dict.keys())

# 获取PyTorch模型的状态字典
pytorch_state_dict = pytorch_model.state_dict()
# print('pytorch_state_dict:', pytorch_state_dict.keys())

# 创建一个新的状态字典来存储映射后的参数
new_state_dict = {}
matching_names = []
for name, param in pytorch_state_dict.items():
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
                matching_param = onnx_param
                matching_name = onnx_name
                matching_names.append(matching_name)
                break
        
        if matching_param is not None:
            new_state_dict[name] = matching_param
            # print('Shape matched:', name, 'matching_name:', matching_name)
        else:
            print(f"Warning: No matching parameter found for {name}")
            new_state_dict[name] = param  # 保留原始参数

# 加载新的状态字典到PyTorch模型
pytorch_model.load_state_dict(new_state_dict, strict=True)

torch.save({'model': pytorch_model.state_dict()}, onnx_model_path[:-5]+'_torch.pth')