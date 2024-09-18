# pip install onnx2pytorch

from onnx2pytorch import ConvertModel
import onnx
import torch

# 加载 ONNX 模型
onnx_model_path = 'pretrained_model/pangu_weather_1.onnx'  # 1, 3, 6, 24
onnx_model = onnx.load(onnx_model_path)

# 转换为 PyTorch 模型
pytorch_model = ConvertModel(onnx_model)

torch.save(pytorch_model, onnx_model_path[:-5]+'_torch.pth')