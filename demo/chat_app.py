"""
风眼 · 风电功率预报智能助手 (chat demo)

Conversational front-end powered by Bedrock Claude (Opus 4.1). When the user
asks about a wind farm's power forecast, it runs the real pipeline:

  ERA5 初始场 → Pangu (逐24h自回归) → CorrDiff 降尺度(占位) → 按海拔选气压层 U/V
  → 功率曲线 → 出力

and renders professional meteorological figures, with Claude providing expert
narration and analysis.

NOTE: launched with CUDA_VISIBLE_DEVICES="" so the pipeline runs on CPU and never
disturbs the GPU (pre2019 training).
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wind_farms as wf
import forecast_pipeline as fp
import llm

st.set_page_config(page_title="风眼 · 风电功率预报助手", layout="wide",
                   initial_sidebar_state="expanded")

SYSTEM = f"""你是「风眼」——一个面向风电场的专业气象与功率预报智能助手。你的语气专业、简洁、可靠，使用中文。

你背后的技术链路（向用户解释时要专业、准确）：
1. ERA5 再分析作为初始场（0.25°，约25km）；
2. 华为 Pangu-Weather 大模型做全球预报（理想为1/3/6/24h四模型逐小时聚合，当前演示用24h模型自回归逐日步进）；
3. CorrDiff 生成式扩散模型做区域降尺度到 km 级（当前为插值占位，接口已就绪）；
4. 按风场海拔+轮毂高度，从 10m/1000/925/850hPa 中选取最接近轮毂高度的层取 U/V 风；
5. 用风机功率曲线（IEC，切入/额定/切出）把风速换算为出力与容量因子。

可用风电场（仅这些是已知的）：
{wf.farms_brief()}

规则：
- 若用户询问某风电场的功率/出力/风速预报，确认风场与预报时长后，简要说明你将运行的预报流程（2-4句，专业）。不要编造未运行的数值。
- 若用户只是闲聊或问概念，正常专业作答。
- 不要杜撰不在以上清单中的风电场；若用户问的风场不存在，礼貌说明并列出可选风场。
"""


def detect_intent(text):
    """Return (farm_name, info, horizon_days) if this is a forecast request, else (None,None,None)."""
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


def render_forecast_figs(result):
    """Render the professional figure suite from a forecast result dict."""
    farm = result["farm"]
    lname = fp._lname(result["level"])

    st.markdown("##### ① 站点位置")
    st.pyplot(fp.fig_region_map(farm))

    st.markdown("##### ② 选层依据（按海拔自动选取气压层）")
    rows = []
    for lv, (h, d) in result["level_details"].items():
        nm = {"10m": "10m", "1000": "1000hPa", "925": "925hPa", "850": "850hPa"}[lv]
        rows.append({"层": nm, "高度(m ASL)": round(h), "与轮毂差(m)": round(d),
                     "选中": "✓" if lv == result["level"] else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown(f"##### ③ 降尺度风场（{lname}，CorrDiff占位）")
    cols = st.columns(len(result["field_snaps"]))
    for col, (k, snap) in zip(cols, sorted(result["field_snaps"].items())):
        with col:
            st.pyplot(fp.fig_wind_field(snap, farm, k, result["level"]))

    st.markdown("##### ④ 轮毂高度风速 & 逐日出力")
    st.pyplot(fp.fig_timeseries(result))

    st.markdown("##### ⑤ 功率曲线与预报落点")
    st.pyplot(fp.fig_power_curve(result))

    # summary table
    df = pd.DataFrame({
        "日期": [t.strftime("%m-%d") for t in result["times"]],
        "轮毂风速(m/s)": [round(x, 1) for x in result["hub_ws"]],
        "容量因子": [f"{x*100:.0f}%" for x in result["cf"]],
        "平均出力(MW)": [round(x, 1) for x in result["power_mw"]],
        "日发电量(MWh)": [round(x, 0) for x in result["daily_energy_mwh"]],
    })
    st.markdown("##### ⑥ 逐日预报汇总")
    st.dataframe(df, hide_index=True, use_container_width=True)


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


# ----------------------------------- UI -----------------------------------
st.title("🌬️ 风眼 · 风电功率预报智能助手")
st.caption("AI 大模型预报 → CorrDiff 降尺度 → 功率曲线出力预测 ｜ 由 Claude (Bedrock) 驱动对话")

with st.sidebar:
    st.subheader("可查询的风电场")
    for r in ["新疆 (Xinjiang)", "浙江 (Zhejiang)"]:
        st.markdown(f"**{r.split(' ')[0]}**")
        for n, f in wf.WIND_FARMS.items():
            if f["region"] == r:
                st.markdown(f"- {n}（{f['capacity_mw']:.0f}MW）")
    st.markdown("---")
    st.markdown("**示例提问**")
    st.code("新疆十二间房风电场未来7天的功率曲线", language=None)
    st.code("浙江括苍山风电场未来5天出力预测", language=None)
    if st.button("清空对话"):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

# replay history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("fc") is not None:
            render_forecast_figs(msg["fc"])

prompt = st.chat_input("询问某个风电场的功率预报，或聊聊风电与气象…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    farm_name, info, horizon = detect_intent(prompt)
    chat_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]

    with st.chat_message("assistant"):
        if info is None:
            # plain professional chat
            reply = st.write_stream(llm.stream_chat(chat_msgs, SYSTEM))
            st.session_state.history.append({"role": "assistant", "content": reply})
        else:
            # 1) intro / plan (streamed)
            intro_user = (f"用户请求：{farm_name} 未来{horizon}天的功率预报。"
                          "请用2-4句话专业说明你将运行的预报链路（不要给数值）。")
            intro = st.write_stream(llm.stream_chat(
                [{"role": "user", "content": intro_user}], SYSTEM, max_tokens=400))

            # 2) run the real pipeline (cached), streaming stage progress
            cache = st.session_state.setdefault("fc_cache", {})
            key = (info["id"], horizon)
            if key in cache:
                result = cache[key]
                st.info(f"（已缓存）初始场 {result['init_date']} ｜ 计算设备 {result['device']}")
            else:
                status = st.status("正在运行端到端预报管线…", expanded=True)
                def prog(stage, frac):
                    status.update(label=f"[{frac*100:.0f}%] {stage}")
                    status.write(f"• {stage}")
                result = fp.run_forecast(info, horizon_days=horizon, progress=prog)
                status.update(label="预报管线完成 ✓", state="complete", expanded=False)
                cache[key] = result

            # 3) figures
            render_forecast_figs(result)

            # 4) expert analysis (streamed)
            st.markdown("##### 🧭 专家分析")
            analysis = st.write_stream(llm.stream_chat(
                [{"role": "user", "content": analysis_prompt(result)}], SYSTEM, max_tokens=1200))

            full = intro + "\n\n（已生成站点图、降尺度风场、出力时序、功率曲线与逐日汇总）\n\n" + \
                "##### 🧭 专家分析\n" + analysis
            st.session_state.history.append({"role": "assistant", "content": full, "fc": result})
