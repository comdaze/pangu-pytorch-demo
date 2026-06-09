# Demo 架构与代码对照

本文档说明 `demo/` 这一层的结构，以及它如何**复用**原始 Pangu-PyTorch 代码。

> 一句话：`demo/` 是新增的**应用层**（交互界面 + 真实推理 / 微调 / 降尺度出力链路），
> 底层调用原始 `models/` 与 `era5_data/` 的模型和数据管线；原始 `finetune/`、
> `inference/` 脚本保持原样、仍可独立使用。

---

## 一、论文背景

本 demo 呈现论文《Bridging the Weather Forecasting Gap: Region-Aware and
Variable-Specific Adaptation of Weather Foundation Models》(VAAWM) 的核心贡献，
并在真实 Pangu-Weather + ERA5 上落地：

- **变量特定权重 α**：聚焦风速 → 对应 `cfg.PG.TRAIN.*_WEIGHTS` / `--only_use_wind_speed_loss`
- **区域特定掩码 β**：聚焦目标区域 → 对应 `aux_data/custom_mask.npy`（`custom_mask.ipynb`）
- **混合推理**：目标变量取微调模型、其余取原始模型 → 缓解自回归漂移（论文 §3.4）

---

## 二、`demo/` 新增文件

| 文件 | 作用 |
|---|---|
| `app.py` | Streamlit 交互界面（8 页，全中文） |
| `paper_data.py` | 论文表 1-4 的真实 RMSE、区域定义、变量权重 |
| `regions.py` | 区域掩码构造（与 `custom_mask.ipynb` 逻辑一致）+ 地图绘制 |
| `inference_engine.py` | 真实推理、混合推理、掩码加权 RMSE、多步自回归滚动、留出集评估 |
| `wind_power.py` | 风电链路 stage 2（降尺度接口/占位插值）+ stage 3（轮毂外推 + 功率曲线） |
| `finetune_vaawm.py` | 忠实小规模 VAAWM 微调（仓库原损失：归一化风速 L1 + 掩码，train/val 早停） |
| `finetune_vaawm_paper.py` | 更贴近论文的 VAAWM（全变量 MSE + 软 β + α，避免灾难性遗忘） |
| `fetch_era5_upper.py` | 从公开 ARCO-ERA5 抓取额外 upper 时刻 |
| `run_pipeline.py` | 2019 多月：下载 → 微调 → 留出集评估（无人值守） |
| `run_pipeline_pre2019.py` | 论文 Pre-2019 协议：训 2016-2017 / 验 2019 / 测 2018 |
| `app_mock_legacy.py` | 原始的随机数 mock demo 备份（不再使用） |
| `utils.py` | 原 demo 的可视化/指标工具（部分仍被 legacy 引用） |

---

## 三、demo 复用的原始 Pangu 代码（核心，被直接调用）

| 原始模块 | demo 中的用途 |
|---|---|
| `models/pangu_model.py` → `PanguModel` | 构建模型、加载 `.pth` 权重、推理/微调 |
| `models/pangu_sample.py` → `get_wind_speed` | 风速损失计算 |
| `models/layers.py` | 被 `PanguModel` 依赖（PatchEmbedding / EarthSpecificLayer 等） |
| `era5_data/config.py` → `cfg` | 全局配置、数据路径、权重路径、HORIZON、损失权重 |
| `era5_data/utils_data.py` | `NetCDFDataset`、`normData`/`normBackData`、`loadAllConstants`、统计量与掩码加载 |
| `era5_data/score.py` → `weighted_rmse_torch_channels` | 纬度加权 RMSE（支持区域掩码） |
| `era5_data/ordered_easydict.py` | 被 config 依赖 |

调用关系示意：

```
demo/app.py
  └─ demo/inference_engine.py
        ├─ models.pangu_model.PanguModel        (原始)
        ├─ models.pangu_sample.get_wind_speed   (原始)
        ├─ era5_data.utils_data (NetCDFDataset, normData, loadAllConstants)  (原始)
        ├─ era5_data.score.weighted_rmse_torch_channels                      (原始)
        └─ era5_data.config.cfg                                              (原始)
  └─ demo/wind_power.py        (stage2 降尺度占位 + stage3 功率曲线，纯新增)
  └─ demo/regions.py / paper_data.py   (纯新增)
```

---

## 四、原始代码：保留、未改动、仍可独立使用

这些**不被 demo 直接调用**，但原始工作流照常可用：

- `finetune/finetune_fully.py`、`finetune/lora_tune.py`：原始全参/LoRA 训练入口（DDP / DeepSpeed）。
  demo 出于单卡/可控考虑，另写了自己的训练循环（`finetune_vaawm*.py`），未改动这两个文件。
- `inference/*.py`：原始推理/评测脚本（`inference_singleOutput.py`、`inference_multiOutput*.py`、
  `inference_*mix24.py`、`calculate_avg_rmse*.py`、`test_main.py`、`test_lora.py` 等），未动。
- `models/onnx2torch.py`：ONNX → torch 权重转换。
- `models/pangu_model_deepspeed.py`：DeepSpeed 版模型。
- `convert_era5.py`、`stat.py`：原始数据转换/统计。
- `era5_data/utils.py`：可视化与日志工具。
- `era5_data/utils_dist.py`：分布式工具。

---

## 五、数据与权重位置（不在仓库内，已被 .gitignore 排除）

均位于 `/opt/dlami/nvme`（由 `cfg.GLOBAL.PATH` 自动探测）：

- `pretrained_model/pangu_weather_24_torch.pth`：官方预训练权重（horizon=24）
- `aux_data/`：统计量、常量掩码、`custom_mask.npy`（新疆）、`const_h`
- `upper/upper_YYYYMMDD.nc`、`surface/surface_YYYYMM.nc`：ERA5 样本（HuggingFace + ARCO-ERA5）
- `model/finetune_vaawm/24/*.pth`：各次微调产出的检查点
  - `vaawm_finetuned.pth`（5 天）、`vaawm_multimonth.pth`（2019 多月）、`vaawm_paper.pth`、
    `vaawm_pre2019.pth`（论文协议，训练中）

---

## 六、运行

```bash
# 启动交互式 demo（需 GPU 才能跑第 ⑥⑦⑧ 页的真实推理）
cd demo
LD_LIBRARY_PATH=/opt/conda/lib PYTHONPATH=.. \
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

> 注：`LD_LIBRARY_PATH=/opt/conda/lib` 用于让 pip 安装的 matplotlib/torch 链接到
> conda 的新版 `libstdc++`（系统自带版本缺少 `CXXABI_1.3.15`）。
