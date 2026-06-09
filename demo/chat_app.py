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

st.set_page_config(page_title="风眼 · 风电功率预报助手", layout="centered",
                   initial_sidebar_state="collapsed")

ASSISTANT_AVATAR = "✨"
USER_AVATAR = "🧑"

# ---- Claude / assistant-ui style theme ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

:root { --claude-coral:#D97757; --claude-bg:#FAF9F5; --claude-panel:#F0EEE6;
        --claude-ink:#2B2A27; --claude-muted:#73726C; --claude-border:#E6E2D8; }

.stApp { background: var(--claude-bg); }

/* centered, ~1/3-of-viewport reading column (adaptive) */
.block-container { max-width:34vw !important; min-width:440px; padding-top:2.2rem; padding-bottom:7rem; }

/* headings -> serif (Claude vibe) */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Newsreader', Georgia, 'Times New Roman', serif !important;
    letter-spacing: .2px; color: var(--claude-ink); }
h1 { font-weight: 600 !important; }

/* app title */
.block-container h1:first-of-type { font-size: 2.1rem; }

/* chat message rows */
[data-testid="stChatMessage"] {
    background: transparent; border: none; padding: .35rem .2rem; gap: .9rem; }

/* user message -> soft rounded panel (Claude user bubble) */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--claude-panel); border: 1px solid var(--claude-border);
    border-radius: 16px; padding: .6rem 1rem; }

/* avatars */
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--claude-coral) !important; color: #fff !important;
    border: none !important; font-size: 1rem; }
[data-testid="stChatMessageAvatarUser"] {
    background: #EDE9DE !important; border: 1px solid var(--claude-border) !important; }

/* body text */
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
    font-size: 1.02rem; line-height: 1.7; color: var(--claude-ink); }

/* section sub-headers inside answers */
[data-testid="stChatMessage"] h5 {
    font-family: 'Newsreader', Georgia, serif !important; color: var(--claude-coral);
    font-weight: 600; margin-top: 1rem; }

/* chat input -> rounded, coral focus */
[data-testid="stChatInput"] {
    border-radius: 18px !important; border: 1px solid var(--claude-border) !important;
    background: #fff !important; box-shadow: 0 2px 10px rgba(0,0,0,.04); }
[data-testid="stChatInput"]:focus-within {
    border-color: var(--claude-coral) !important; box-shadow: 0 0 0 2px rgba(217,119,87,.18); }

/* sidebar */
[data-testid="stSidebar"] { background: #F2F0E9; border-right: 1px solid var(--claude-border); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--claude-ink); }

/* dataframes & status soften */
[data-testid="stDataFrame"] { border-radius: 10px; }
.stButton button { border-radius: 10px; }

/* hide Streamlit chrome for a cleaner app */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* remove the sidebar entirely */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }
</style>""", unsafe_allow_html=True)

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

    method = result.get("downscale_method", "bilinear (placeholder)")
    st.markdown(f"##### ③ 降尺度风场（{lname}，{method}）")
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
import chat_store as cs

if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = cs.new_id()

EXAMPLES = [
    "新疆十二间房风电场未来7天的功率曲线",
    "浙江括苍山风电场未来5天出力预测",
    "新疆达坂城风电场未来3天发电量",
    "浙江大陈岛海上风电场未来7天功率预报",
]


def load_session(sid):
    s = cs.get(sid)
    if s:
        st.session_state.history = [dict(m) for m in s["messages"]]
        st.session_state.session_id = sid


# ---- extra CSS for the Claude-style landing hero ----
st.markdown("""
<style>
.hero-title { text-align:center; font-family:'Newsreader',Georgia,serif;
    font-size:2.1rem; font-weight:600; color:#2B2A27; margin:0 0 1.4rem; }
.hero-title .spark { color:#D97757; margin-right:.3rem; }
.chips-label { text-align:center; color:#9a978d; font-size:.85rem; margin:.9rem 0 .4rem; }
/* landing input box -> Claude composer card */
[data-testid="stForm"] { background:#fff; border:1px solid #E6E2D8; border-radius:22px;
    padding:.7rem 1rem .8rem; box-shadow:0 6px 24px rgba(0,0,0,.06); }
[data-testid="stForm"] [data-baseweb="textarea"],
[data-testid="stForm"] [data-baseweb="base-input"] { border:none !important; background:transparent !important; }
[data-testid="stForm"] textarea { font-size:1.08rem !important; line-height:1.6 !important;
    border:none !important; box-shadow:none !important; background:transparent !important;
    color:#2B2A27 !important; }
[data-testid="stForm"] textarea::placeholder { color:#b3afa4 !important; }
/* send button (primary, coral) */
[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    background:#D97757 !important; color:#fff !important; border:none !important;
    border-radius:12px !important; font-size:.95rem !important; padding:.45rem 1rem !important; }
/* bottom chat input -> taller */
[data-testid="stChatInput"] textarea { min-height:58px !important; font-size:1.05rem !important;
    line-height:1.6 !important; }
[data-testid="stChatInput"] { padding:.3rem .4rem !important; }
/* example pills */
.stButton button { border-radius:999px !important; border:1px solid #E2DED4 !important;
    background:#fff !important; color:#4a4842 !important; font-size:.86rem !important;
    padding:.28rem .9rem !important; }
.stButton button:hover { border-color:#D97757 !important; color:#D97757 !important; }
/* pin the example-pill row BELOW the bottom chat input */
.chipbar { height:0; }
/* raise the fixed bottom input to leave room for pills underneath */
[data-testid="stBottom"] { bottom: 3.2rem !important; }
[data-testid="stElementContainer"]:has(.chipbar) + [data-testid="stHorizontalBlock"] {
    position: fixed; bottom: 0.55rem; left: 50%; transform: translateX(-50%);
    width: min(34vw, 92vw); min-width: 440px; z-index: 90; background: var(--claude-bg);
    padding: .15rem 0; }
.block-container { padding-bottom: 10.5rem !important; }
/* top bar */
.topbar-logo { font-family:'Newsreader',Georgia,serif; font-size:1.15rem; font-weight:600;
    color:#2B2A27; } .topbar-logo .spark{color:#D97757;}
</style>
""", unsafe_allow_html=True)

# ---- slim top bar ----
tb = st.columns([0.66, 0.17, 0.17])
tb[0].markdown("<div class='topbar-logo'><span class='spark'>✦</span> 风眼 "
               "<span style='color:#9a978d;font-size:.8rem'>风电功率预报助手</span></div>",
               unsafe_allow_html=True)
with tb[1]:
    if st.button("➕ 新对话", use_container_width=True):
        cs.upsert(st.session_state.session_id, st.session_state.history)
        st.session_state.history = []
        st.session_state.session_id = cs.new_id()
        st.session_state.pop("fc_cache", None)
        st.rerun()
with tb[2]:
    with st.popover("📚 历史", use_container_width=True):
        sessions = cs.list_sessions()
        if not sessions:
            st.caption("暂无历史对话")
        for s in sessions[:cs.MAX_SESSIONS]:
            mark = "• " if s["id"] == st.session_state.session_id else ""
            if st.button(f"{mark}{s['title']}", key="sess_" + s["id"], use_container_width=True):
                cs.upsert(st.session_state.session_id, st.session_state.history)
                load_session(s["id"])
                st.session_state.pop("fc_cache", None)
                st.rerun()

pending = st.session_state.pop("pending_prompt", None)
is_landing = (not st.session_state.history) and (pending is None)

# replay history
for msg in st.session_state.history:
    with st.chat_message(msg["role"],
                         avatar=ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR):
        st.markdown(msg["content"])
        if msg.get("fc") is not None:
            render_forecast_figs(msg["fc"])

CHIPS = [("🌬️ 新疆·十二间房 7天", EXAMPLES[0]),
         ("⛰️ 浙江·括苍山 5天", EXAMPLES[1]),
         ("💨 新疆·达坂城 3天", EXAMPLES[2]),
         ("🌊 浙江·大陈岛 7天", EXAMPLES[3])]

prompt = None
if is_landing:
    # ---- Claude-style centered landing: greeting -> input box -> chips ----
    st.markdown("<div style='height:9vh'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'><span class='spark'>✦</span>今天想预报哪个风电场？</div>",
                unsafe_allow_html=True)
    with st.form("landing", clear_on_submit=True, border=True):
        q = st.text_area("q", label_visibility="collapsed", height=110,
                         placeholder="向「风眼」提问，例如：新疆十二间房风电场未来7天的功率曲线")
        bc = st.columns([0.72, 0.28])
        bc[0].markdown("<span style='color:#b3afa4;font-size:.8rem'>Ctrl + Enter 发送</span>",
                       unsafe_allow_html=True)
        sent = bc[1].form_submit_button("发送 ↑", use_container_width=True, type="primary")
    if sent and q.strip():
        st.session_state.pending_prompt = q.strip()
        st.rerun()
    cc = st.columns(4)
    for i, (label, full) in enumerate(CHIPS):
        if cc[i].button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_prompt = full
            st.rerun()
else:
    typed = st.chat_input("向「风眼」提问，例如：新疆十二间房风电场未来7天的功率曲线")
    prompt = typed or pending
    # example pills pinned just above the bottom input
    st.markdown("<div class='chipbar'></div>", unsafe_allow_html=True)
    cc = st.columns(4)
    for i, (label, full) in enumerate(CHIPS):
        if cc[i].button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_prompt = full
            st.rerun()

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    farm_name, info, horizon = detect_intent(prompt)
    chat_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if info is None:
            reply = st.write_stream(llm.stream_chat(chat_msgs, SYSTEM))
            st.session_state.history.append({"role": "assistant", "content": reply})
        else:
            intro_user = (f"用户请求：{farm_name} 未来{horizon}天的功率预报。"
                          "请用2-4句话专业说明你将运行的预报链路（不要给数值）。")
            intro = st.write_stream(llm.stream_chat(
                [{"role": "user", "content": intro_user}], SYSTEM, max_tokens=400))

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

            render_forecast_figs(result)

            st.markdown("##### 🧭 专家分析")
            analysis = st.write_stream(llm.stream_chat(
                [{"role": "user", "content": analysis_prompt(result)}], SYSTEM, max_tokens=1200))

            full = intro + "\n\n（已生成站点图、降尺度风场、出力时序、功率曲线与逐日汇总）\n\n" + \
                "##### 🧭 专家分析\n" + analysis
            st.session_state.history.append({"role": "assistant", "content": full, "fc": result})

    cs.upsert(st.session_state.session_id, st.session_state.history)
    st.rerun()
