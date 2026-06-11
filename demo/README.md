# 风眼 · 风电功率预报智能助手 (Demo)

一个面向风电场的对话式功率预报演示：Claude 风格聊天界面，由 Bedrock Claude
以 **function calling** 驱动，按需运行端到端预报管线并流式输出专业气象图与分析。

前端 React (assistant-ui)，后端 FastAPI，单服务统一托管并带登录鉴权。

## 端到端链路

```
ERA5 初始场（最接近当天季节的可用再分析场）
  → Pangu-Weather 推理
       · 逐日(24h)：混合推理 Hybrid（目标风场取 VAAWM 微调，余取 zero-shot 基座，论文§3.4）
       · 逐 6/3/1 小时：官方 zero-shot Pangu-{6,3,1}h
  → CorrDiff 生成式降尺度（regression + diffusion，SageMaker 训练，25km→5km）
  → 按风场海拔/轮毂高度选气压层(10m/1000/925/850hPa) 取 U/V
  → 风机功率曲线(IEC) → 出力 / 容量因子 / 发电量
```

时间分辨率、风场、时长均由 LLM 从多轮对话上下文中推断（工具参数
`farm_query` / `horizon_days` / `step_hours`）。

## 架构与文件

后端 / 管线
- `api.py` — FastAPI 服务：Bedrock Claude 工具调用编排、流式进度+图、托管前端静态、HTTP Basic Auth、`/healthz`。
- `forecast_pipeline.py` — 预报管线（混合/zero-shot 自回归、CorrDiff 降尺度、选层、功率换算、出图）。
- `inference_engine.py` — Pangu 模型加载（zeroshot / vaawm / vaawm_pre2019 / 各时效 zs）、aux 常量、ERA5 初始场。
- `corrdiff_infer.py` — CorrDiff 两阶段推理封装。
- `wind_power.py` — 选层、轮毂外推、IEC 功率曲线、降尺度接口。
- `wind_farms.py` — 新疆/浙江风电场清单（mock）。
- `llm.py` — Bedrock Claude (Converse) 封装。
- `paper_data.py` / `regions.py` — 区域掩码与论文相关数据。

前端
- `web/` — Vite + React + assistant-ui（Claude 主题）。详见 `web/README.md`。

训练 / 数据（独立脚本，非服务运行所需）
- `run_pipeline_pre2019.py` — 论文 Pre-2019 协议 VAAWM 微调全流程。
- `finetune_vaawm.py` / `finetune_vaawm_paper.py` — VAAWM 微调实验。
- `fetch_era5_upper.py` / `run_pipeline.py` — ERA5 数据获取与流程脚本。

运维
- `ops/` — S3 备份/恢复脚本、systemd 单元、部署文档（见 `ops/DEPLOY.md`）。

## 本地运行

后端（GPU 推理；`LD_LIBRARY_PATH` 修复 matplotlib 的 libstdc++）：

```bash
cd demo
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/conda/lib \
  FENGYAN_USER=admin FENGYAN_PASS=<your-pass> \
  uvicorn api:app --host 0.0.0.0 --port 8000
```

前端开发态（/api 代理到 8000）：

```bash
cd demo/web && npm install && npm run dev    # http://localhost:5173
```

生产态（单服务托管前端 + API）：

```bash
cd demo/web && npm run build                 # 产物 dist/ 由 api.py 自动托管
# 浏览器访问 http://<host>:8000 ，用上面的用户名/密码登录
```

## 部署（长期运行 + ALB + 登录）

见 `ops/DEPLOY.md`：systemd 自启/重启、实例临时盘被清时从 S3 自动恢复、
ALB(:80)→实例(:8000，健康检查 `/healthz`)、HTTP Basic Auth 登录。

## 参考

- Bi et al. (2022) Pangu-Weather；NVIDIA PhysicsNeMo CorrDiff；
  VAAWM（变量-区域自适应加权 + 混合推理）见仓库内论文 PDF。
