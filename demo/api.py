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
import json
import os
import queue
import re
import sys
import threading

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
2. 华为 Pangu-Weather 大模型做全球预报（24h 模型自回归逐日步进），采用论文的混合推理（Hybrid Inference）策略：每个自回归步，目标风场变量（10m 风 u10/v10、高空 u/v）取自 VAAWM 微调模型，其余变量（z/q/t、msl/t2m）取自基座 zero-shot 模型，兼顾局地风速精度与全局物理一致性、抑制长程漂移；逐日为混合推理；更细的逐小时/6小时推理使用官方 zero-shot 模型（已就绪 1/3/6/24h 四个时效）；
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
    step_h = result.get("step_hours", 24)
    tfmt = "%m-%d" if step_h == 24 else "%m-%d %Hh"
    res_txt = "逐日" if step_h == 24 else f"逐{step_h}小时"
    lines = [f"{t.strftime(tfmt)}: 风速{w:.1f}m/s, 容量因子{c*100:.0f}%, 出力{p:.0f}MW"
             for t, w, c, p in zip(result["times"], result["hub_ws"], result["cf"], result["power_mw"])]
    return (
        f"风电场：{f['id']}，装机{f['capacity_mw']:.0f}MW，{f['turbines']}台{f['turbine_model']}，"
        f"轮毂{f['hub_height_m']}m，海拔{f['elevation_m']}m，地形：{f['terrain']}。\n"
        f"预报模型：Pangu {result.get('pangu_model','—')}；降尺度 {result.get('downscale_method','—')}。\n"
        f"时间分辨率：{res_txt}（步长 {step_h}h）。\n"
        f"选用气压层：{fp._lname(result['level'])}（按海拔自动选取）。\n"
        f"初始场日期：{result['init_date']}，预报时长：{result['horizon_days']}天。\n"
        f"{res_txt}预报：\n" + "\n".join(lines) + "\n"
        f"预报期平均容量因子 {result['mean_cf']*100:.0f}%，累计发电量约 {result['total_energy_mwh']:.0f} MWh。\n\n"
        "请作为风电功率预报专家，对以上结果给出专业分析（中文，分点，简洁）："
        "包括风况与天气形势研判、出力高/低值时段与爬坡(ramp)风险、容量因子评价、"
        "以及对电力调度/检修安排/电力市场交易的建议。不要重复罗列每个时刻的数字。"
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
    for lead_h, snap in sorted(result["field_snaps"].items()):
        yield _fig_md(fp.fig_wind_field(snap, farm, lead_h, result["level"]), f"lead +{lead_h}h")

    step_h = result.get("step_hours", 24)
    res_txt = "逐日" if step_h == 24 else f"逐{step_h}小时"
    yield _fig_md(fp.fig_timeseries(result), f"④ 轮毂高度风速 & {res_txt}出力")
    yield _fig_md(fp.fig_power_curve(result), "⑤ 功率曲线与预报落点")

    # ⑥ summary table (adapts to the rollout granularity)
    tfmt = "%m-%d" if step_h == 24 else "%m-%d %Hh"
    date_label = "日期" if step_h == 24 else "时刻"
    energy_label = "日发电量(MWh)" if step_h == 24 else f"{step_h}h发电量(MWh)"
    hdr = [f"| {date_label} | 轮毂风速(m/s) | 容量因子 | 平均出力(MW) | {energy_label} |",
           "|---|---|---|---|---|"]
    for t, w, c, p, e in zip(result["times"], result["hub_ws"], result["cf"],
                             result["power_mw"], result["daily_energy_mwh"]):
        hdr.append(f"| {t.strftime(tfmt)} | {w:.1f} | {c*100:.0f}% | {p:.1f} | {e:.0f} |")
    yield f"\n\n##### ⑥ {res_txt}预报汇总\n\n" + "\n".join(hdr) + "\n"


class ChatRequest(BaseModel):
    messages: list  # [{role, content}]


FORECAST_TOOL = {
    "toolSpec": {
        "name": "run_wind_power_forecast",
        "description": (
            "运行端到端风电功率预报管线：ERA5 初始场 → Pangu-Weather 混合推理（逐日 24h，"
            "目标风场取 VAAWM 微调、其余取 zero-shot 基座）→ CorrDiff 降尺度 → 按海拔选气压层 "
            "→ 功率曲线，产出逐日风速/容量因子/出力/发电量及多张专业气象图。"
            "当用户想要某风电场的功率、出力、发电量、风速或功率曲线预报时调用此工具。"
            "请结合多轮对话上下文推断风场名称与预报时长。"
        ),
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "farm_query": {
                    "type": "string",
                    "description": "风电场名称或关键词，例如：十二间房、达坂城、小草湖、三塘湖、括苍山、大陈岛、苍南。",
                },
                "horizon_days": {
                    "type": "integer",
                    "description": "预报时长（天），范围 1-10。若用户用小时表述（如 8 小时）请折算并向上取整到天，最少 1 天。默认 7。",
                },
                "step_hours": {
                    "type": "integer",
                    "enum": [1, 3, 6, 24],
                    "description": (
                        "时间分辨率（步长，小时）。24=逐日（默认，采用 VAAWM 混合推理，论文证明 1-5 天最优）；"
                        "6/3/1=逐 6/3/1 小时（使用官方 zero-shot 模型，无微调）。"
                        "当用户要求‘逐小时/逐6小时/更细分辨率/小时级’时设为 6 或 1；否则用 24。"
                    ),
                },
            },
            "required": ["farm_query"],
        }},
    }
}

_IMG_RE = re.compile(r"!\[\]\(data:image/[^)]+\)")


def _sanitize(text):
    """Strip large base64 image data URIs from history before sending to the LLM."""
    return _IMG_RE.sub("［气象图］", text or "")


def _to_bedrock(messages):
    return [{"role": m["role"], "content": [{"text": _sanitize(m["content"])}]}
            for m in messages]


def _run_forecast_stream(info, horizon, box, step_hours=24):
    """Yield markdown (live progress + model chain + figures); store result in box."""
    yield "\n\n##### ⏳ 运行进度\n\n"
    q: "queue.Queue" = queue.Queue()

    def prog(stage, frac):
        q.put(f"- `{int(frac * 100):>3d}%`  {stage}\n")

    def worker():
        try:
            box["result"] = fp.run_forecast(info, horizon_days=horizon,
                                            step_hours=step_hours, progress=prog)
        except Exception as e:  # noqa: BLE001
            box["error"] = repr(e)
        finally:
            q.put(None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    t.join()

    if "error" in box:
        yield f"\n> ⚠️ 预报管线出错：{box['error']}\n"
        return
    result = box["result"]
    yield "- `100%`  生成气象图与图表…\n"
    yield (f"\n> **模型链路**：Pangu = {result.get('pangu_model','—')}；"
           f"降尺度 = {result.get('downscale_method','—')}\n")
    for chunk in forecast_figures_md(result):
        yield chunk


def _stream_text(resp):
    for ev in resp["stream"]:
        if "contentBlockDelta" in ev:
            d = ev["contentBlockDelta"]["delta"]
            if "text" in d:
                yield d["text"]


def chat_stream(messages):
    """Tool-calling chat: the LLM decides (from multi-turn context) whether to call
    the forecast tool, extracting farm + horizon itself. Fixes brittle keyword
    matching and supports follow-up turns like just naming the farm."""
    conv = _to_bedrock(messages)
    cl = llm.client()
    resp = cl.converse_stream(
        modelId=llm.MODEL_ID,
        messages=conv,
        system=[{"text": SYSTEM}],
        toolConfig={"tools": [FORECAST_TOOL]},
        inferenceConfig={"maxTokens": 1200, "temperature": 0.4},
    )

    pre_text = ""
    tool_use = None
    tool_input_json = ""
    for ev in resp["stream"]:
        if "contentBlockStart" in ev:
            st = ev["contentBlockStart"]["start"]
            if "toolUse" in st:
                tool_use = {"toolUseId": st["toolUse"]["toolUseId"],
                            "name": st["toolUse"]["name"]}
                tool_input_json = ""
        elif "contentBlockDelta" in ev:
            d = ev["contentBlockDelta"]["delta"]
            if "text" in d:
                pre_text += d["text"]
                yield d["text"]
            elif "toolUse" in d:
                tool_input_json += d["toolUse"].get("input", "")

    if tool_use is None:
        return  # plain chat answer already streamed

    try:
        args = json.loads(tool_input_json or "{}")
    except Exception:
        args = {}
    farm_query = str(args.get("farm_query", ""))
    horizon = max(1, min(int(args.get("horizon_days", 7) or 7), 10))
    step_hours = int(args.get("step_hours", 24) or 24)
    if step_hours not in (1, 3, 6, 24):
        step_hours = 24
    farm_name, info = wf.find_farm(farm_query)

    assistant_turn = ([{"text": pre_text}] if pre_text.strip() else []) + [
        {"toolUse": {"toolUseId": tool_use["toolUseId"], "name": tool_use["name"],
                     "input": args}}]
    conv.append({"role": "assistant", "content": assistant_turn})

    if info is None:
        conv.append({"role": "user", "content": [{"toolResult": {
            "toolUseId": tool_use["toolUseId"],
            "content": [{"text": f"未匹配到风场“{farm_query}”。可选风场：\n{wf.farms_brief()}"}],
            "status": "error"}}]})
        yield from _stream_text(cl.converse_stream(
            modelId=llm.MODEL_ID, messages=conv, system=[{"text": SYSTEM}],
            toolConfig={"tools": [FORECAST_TOOL]},
            inferenceConfig={"maxTokens": 600, "temperature": 0.4}))
        return

    # run the pipeline (stream live progress + figures to the UI)
    box = {}
    for chunk in _run_forecast_stream(info, horizon, box, step_hours=step_hours):
        yield chunk
    if "result" not in box:
        return
    result = box["result"]

    # feed a compact numeric summary back as the tool result -> expert analysis
    conv.append({"role": "user", "content": [{"toolResult": {
        "toolUseId": tool_use["toolUseId"],
        "content": [{"text": analysis_prompt(result)}]}}]})

    yield "\n\n##### 🧭 专家分析\n\n"
    yield from _stream_text(cl.converse_stream(
        modelId=llm.MODEL_ID, messages=conv, system=[{"text": SYSTEM}],
        toolConfig={"tools": [FORECAST_TOOL]},
        inferenceConfig={"maxTokens": 1200, "temperature": 0.4}))


@app.get("/api/farms")
def farms():
    return list(wf.WIND_FARMS.values())


@app.post("/api/chat")
def chat(req: ChatRequest):
    return StreamingResponse(chat_stream(req.messages),
                             media_type="text/plain; charset=utf-8")
