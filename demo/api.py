"""
FastAPI backend for the assistant-ui (Claude-style) front-end.

Wraps the existing Python pipeline:
  - Bedrock Claude streaming (llm.stream_chat)
  - wind-farm forecast pipeline (forecast_pipeline.run_forecast)
  - professional matplotlib figures -> inline base64 PNG (markdown images)

Exposes a single streaming endpoint `/api/chat` that emits plain UTF-8 text
chunks (markdown). The assistant-ui local-runtime adapter on the front-end
accumulates these chunks into the assistant message.

Run (GPU inference, matplotlib libstdc++ fix):
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/conda/lib \
      uvicorn api:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wind_farms as wf
import forecast_pipeline as fp
import llm

app = FastAPI(title="风眼 · 风电功率预报 API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

SYSTEM = f"""你是「风眼」——一个面向风电场的专业气象与功率预报智能助手。语气专业、简洁、可靠，使用中文。

技术链路（解释时要专业、准确）：
1. ERA5 再分析作为初始场（0.25°，约25km）；
2. 华为 Pangu-Weather 大模型做全球预报（演示用24h模型自回归逐日步进）；
3. CorrDiff 生成式扩散模型做区域降尺度到 km 级（已接入 SageMaker 训练的回归+扩散两阶段模型）；
4. 按风场海拔+轮毂高度，从 10m/1000/925/850hPa 中选取最接近轮毂高度的层取 U/V 风；
5. 用风机功率曲线（IEC，切入/额定/切出）把风速换算为出力与容量因子。

可用风电场（仅这些是已知的）：
{wf.farms_brief()}

规则：
- 若用户询问某风电场的功率/出力/风速预报，确认风场与时长后，简要说明你将运行的预报流程（2-4句，专业）。不要编造未运行的数值。
- 闲聊或概念问题，正常专业作答。
- 不要杜撰不在清单中的风电场；若用户问的风场不存在，礼貌说明并列出可选风场。
"""


def detect_intent(text):
    name, info = wf.find_farm(text)
    if not info:
        return None, None, None
    fc_kw = any(k in text for k in ["功率", "出力", "发电", "风速", "预报", "预测", "功率曲线", "电量"])
    m = re.search(r"(\d+)\s*(天|日|day)", text)
    horizon = int(m.group(1)) if m else 7
    horizon = max(1, min(horizon, 10))
    if fc_kw or m:
        return name, info, horizon
    return None, None, None


def analysis_prompt(result):
    f = result["farm"]
    lines = [f"{t.strftime('%m-%d')}: 风速{w:.1f}m/s, 容量因子{c*100:.0f}%, 出力{p:.0f}MW"
             for t, w, c, p in zip(result["times"], result["hub_ws"], result["cf"], result["power_mw"])]
    return (
        f"风电场：{f['id']}，装机{f['capacity_mw']:.0f}MW，{f['turbines']}台{f['turbine_model']}，"
        f"轮毂{f['hub_height_m']}m，海拔{f['elevation_m']}m，地形：{f['terrain']}。\n"
        f"选用气压层：{fp._lname(result['level'])}（按海拔自动选取）。\n"
        f"初始场日期：{result['init_date']}，预报时长：{result['horizon_days']}天。\n"
        f"逐日预报：\n" + "\n".join(lines) + "\n"
        f"预报期平均容量因子 {result['mean_cf']*100:.0f}%，累计发电量约 {result['total_energy_mwh']:.0f} MWh。\n\n"
        "请作为风电功率预报专家，对以上结果给出专业分析（中文，分点，简洁）："
        "包括风况与天气形势研判、出力高/低值时段与爬坡(ramp)风险、容量因子评价、"
        "以及对电力调度/检修安排/电力市场交易的建议。不要重复罗列每日数字。"
    )


def _fig_md(fig, caption):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"\n\n##### {caption}\n\n![]({'data:image/png;base64,' + b64})\n"


def forecast_figures_md(result):
    """Yield markdown chunks for the full figure suite + summary table."""
    farm = result["farm"]
    lname = fp._lname(result["level"])
    yield _fig_md(fp.fig_region_map(farm), "① 站点位置")

    # ② level selection table (markdown)
    rows = ["| 层 | 高度(m ASL) | 与轮毂差(m) | 选中 |", "|---|---|---|---|"]
    for lv, (h, d) in result["level_details"].items():
        nm = {"10m": "10m", "1000": "1000hPa", "925": "925hPa", "850": "850hPa"}[lv]
        rows.append(f"| {nm} | {round(h)} | {round(d)} | {'✓' if lv == result['level'] else ''} |")
    yield "\n\n##### ② 选层依据（按海拔自动选取气压层）\n\n" + "\n".join(rows) + "\n"

    method = result.get("downscale_method", "bilinear (placeholder)")
    yield f"\n\n##### ③ 降尺度风场（{lname}，{method}）\n"
    for k, snap in sorted(result["field_snaps"].items()):
        yield _fig_md(fp.fig_wind_field(snap, farm, k, result["level"]), f"lead +{k*24}h")

    yield _fig_md(fp.fig_timeseries(result), "④ 轮毂高度风速 & 逐日出力")
    yield _fig_md(fp.fig_power_curve(result), "⑤ 功率曲线与预报落点")

    # ⑥ summary table
    hdr = ["| 日期 | 轮毂风速(m/s) | 容量因子 | 平均出力(MW) | 日发电量(MWh) |",
           "|---|---|---|---|---|"]
    for t, w, c, p, e in zip(result["times"], result["hub_ws"], result["cf"],
                             result["power_mw"], result["daily_energy_mwh"]):
        hdr.append(f"| {t.strftime('%m-%d')} | {w:.1f} | {c*100:.0f}% | {p:.1f} | {e:.0f} |")
    yield "\n\n##### ⑥ 逐日预报汇总\n\n" + "\n".join(hdr) + "\n"


class ChatRequest(BaseModel):
    messages: list  # [{role, content}]


def chat_stream(messages):
    user_text = ""
    for m in reversed(messages):
        if m["role"] == "user":
            user_text = m["content"]
            break

    farm_name, info, horizon = detect_intent(user_text)
    if info is None:
        for delta in llm.stream_chat(messages, SYSTEM):
            yield delta
        return

    # forecast flow: intro -> figures -> analysis
    intro_user = (f"用户请求：{farm_name} 未来{horizon}天的功率预报。"
                  "请用2-4句话专业说明你将运行的预报链路（不要给数值）。")
    for delta in llm.stream_chat([{"role": "user", "content": intro_user}], SYSTEM,
                                 max_tokens=400):
        yield delta

    yield "\n\n---\n"
    result = fp.run_forecast(info, horizon_days=horizon)
    for chunk in forecast_figures_md(result):
        yield chunk

    yield "\n\n##### 🧭 专家分析\n\n"
    for delta in llm.stream_chat([{"role": "user", "content": analysis_prompt(result)}],
                                 SYSTEM, max_tokens=1200):
        yield delta


@app.get("/api/farms")
def farms():
    return list(wf.WIND_FARMS.values())


@app.post("/api/chat")
def chat(req: ChatRequest):
    return StreamingResponse(chat_stream(req.messages),
                             media_type="text/plain; charset=utf-8")
