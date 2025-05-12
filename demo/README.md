# PanGu Weather Forecast GUI Demo

这是一个基于Streamlit的PanGu-Weather模型可视化演示应用。该应用允许用户以交互方式探索PanGu-Weather模型的天气预报能力。

## 功能特点

- 选择不同的预测时长模型（1小时、3小时、6小时、24小时）
- 可视化高空和地表天气预报结果
- 支持多种气象变量的可视化（温度、风速、气压等）
- 计算和展示预测性能指标（RMSE等）
- 导出预测结果和可视化图表

## 安装与运行

### 前提条件

- Python 3.8+
- PanGu-Weather模型及其依赖项

### 安装步骤

1. 确保已安装PanGu-Weather项目的所有依赖项
2. 安装Demo所需的额外依赖项：

```bash
cd /path/to/pangu-pytorch-demo/demo
pip install -r requirements.txt
```

### 运行应用

```bash
cd /path/to/pangu-pytorch-demo/demo
streamlit run app.py
```

应用将在本地启动，并自动在默认浏览器中打开。

## 使用指南

1. **选择数据**：在侧边栏中指定数据路径
2. **选择模型**：选择预测时长（1小时、3小时、6小时或24小时）
3. **设置时间**：选择预测的起始日期和时间
4. **选择变量**：选择要可视化的气象变量和层级
5. **运行预测**：点击"运行预测"按钮开始预测
6. **查看结果**：预测完成后，可以查看预测结果和实际数据的对比
7. **查看指标**：展开"性能指标"部分查看预测性能
8. **导出结果**：在"结果导出"部分下载预测图像或数据

## 注意事项

- 当前演示版本使用模拟数据。要使用真实数据，请确保已下载并配置好ERA5数据集
- 确保已正确设置`era5_data/config.py`中的数据路径
- 对于大型数据集，预测过程可能需要较长时间

## 自定义与扩展

如需自定义或扩展此Demo，可以修改以下文件：

- `app.py`：主应用逻辑和用户界面
- `utils.py`：辅助函数，包括可视化和数据处理

## 参考资料

- Bi et al. (2023) - Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast
- [PanGu-Weather GitHub仓库](https://github.com/198808xc/Pangu-Weather)