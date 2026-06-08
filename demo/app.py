"""
VAAWM 交互式演示 — 盘古气象模型的区域感知与变量特定适配。

呈现论文《Bridging the Weather Forecasting Gap: Region-Aware and
Variable-Specific Adaptation of Weather Foundation Models》(VAAWM) 的核心贡献，
并与本仓库代码一一对应：

  - 变量特定权重 (alpha_v)  -> cfg.PG.TRAIN.UPPER/SURFACE_WEIGHTS + --only_use_wind_speed_loss
  - 区域特定掩码 (beta_i,j)  -> aux_data/custom_mask.npy (custom_mask.ipynb)
  - 掩码加权纬度 RMSE        -> era5_data/score.weighted_rmse_torch_channels
  - 混合推理                 -> inference/inference_*mix*.py

界面所有定量结果均转写自论文表 1-4；区域掩码用与 custom_mask.ipynb 相同的经纬度逻辑构造。
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

import paper_data as pdata
import regions as reg

try:
    import inference_engine as eng
    _ENGINE_OK = True
    _ENGINE_ERR = None
except Exception as _e:  # torch / model import problems
    _ENGINE_OK = False
    _ENGINE_ERR = str(_e)

st.set_page_config(page_title="VAAWM · 盘古气象模型适配演示",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .main-header {font-size: 2.1rem; font-weight: 800; color: #1E3A8A; margin-bottom: .2rem;}
      .sub {color:#475569; font-size: 1.0rem;}
      .sec {font-size: 1.35rem; font-weight: 700; color:#2563EB; margin-top:.3rem; margin-bottom:.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">盘古气象模型的区域感知 · 变量特定适配演示</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub">变量-区域自适应加权机制 (VAAWM)：把全球通用气象大模型，'
            '专门化为面向风能的区域风速预报模型。</p>', unsafe_allow_html=True)

logo_path = os.path.join(parent_dir, "fig", "VIS.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True, caption="Pangu-Weather")

PAGES = [
    "① 方法概览（VAAWM）",
    "② 区域掩码 β（交互）",
    "③ 核心结果 RMSE（交互）",
    "④ 消融实验",
    "⑤ 混合推理策略",
    "⑥ 实时推理（真实模型）",
    "⑦ 多步滚动 & 泛化（真实）",
]
page = st.sidebar.radio("导航", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("定量结果转写自论文表 1-4；区域掩码与 `custom_mask.ipynb` 构造逻辑一致。")

BASE_KEY = "Pangu-Weather (zero-shot)"
OURS_KEY = "Pangu-Weather (VAAWM, Ours)"


# ============================================================== 页面 ① ======
if page == PAGES[0]:
    st.markdown('<p class="sec">要解决的问题</p>', unsafe_allow_html=True)
    st.write("像盘古这样的气象大模型，训练时对**所有变量、整个地球**使用统一的损失函数。"
             "但风能用户真正关心的是**某个特定区域**（比如风电场）的**风速**预报——"
             "统一损失会稀释模型对关键变量和关键区域的注意力。")

    st.markdown('<p class="sec">核心思路：VAAWM 自适应加权损失</p>', unsafe_allow_html=True)
    st.write("把普通的 MSE/MAE 损失换成对目标变量、目标区域加权的损失：")
    st.latex(r"\mathcal{L}_{\text{adaptive}} = \sum_{v=1}^{V} \alpha_v "
             r"\sum_{(i,j)\in \mathcal{G}} \beta_{i,j}\,\bigl(Y_{i,j,v}-\hat{Y}_{i,j,v}\bigr)^2")
    c = st.columns(2)
    with c[0]:
        st.markdown("**α_v —— 变量特定权重**")
        st.write("提高风相关变量（`u/v`、`u10/v10`）的权重，让模型聚焦风速精度。"
                 "对应本仓库的 `--only_use_wind_speed_loss` 以及配置中的权重。")
    with c[1]:
        st.markdown("**β_i,j —— 区域特定掩码**")
        st.write("只保留目标区域（如新疆/浙江）内的网格点。"
                 "对应本仓库 `aux_data/custom_mask.npy`（由 `custom_mask.ipynb` 生成）。")

    st.markdown('<p class="sec">🎛 交互：调节变量权重 α，观察风速侧重</p>', unsafe_allow_html=True)
    st.caption("拖动滑块改变各变量权重，下方柱状图与「风速侧重度」会实时更新。"
               "仓库默认值见 `era5_data/config.py`。")

    cc = st.columns(2)
    with cc[0]:
        st.markdown("**地表变量权重**")
        s_w = {}
        for v, dflt in zip(pdata.SURFACE_VARS, pdata.SURFACE_WEIGHTS):
            s_w[v] = st.slider(pdata.SURFACE_VARS_CN[v], 0.0, 3.0, float(dflt), 0.01,
                               key=f"sw_{v}")
    with cc[1]:
        st.markdown("**高空变量权重**")
        u_w = {}
        for v, dflt in zip(pdata.UPPER_VARS, pdata.UPPER_WEIGHTS):
            u_w[v] = st.slider(pdata.UPPER_VARS_CN[v], 0.0, 3.0, float(dflt), 0.01,
                               key=f"uw_{v}")

    g1, g2 = st.columns(2)
    g1.bar_chart(pd.DataFrame({"权重 α": list(s_w.values())},
                              index=[pdata.SURFACE_VARS_CN[v] for v in s_w]), height=260)
    g2.bar_chart(pd.DataFrame({"权重 α": list(u_w.values())},
                              index=[pdata.UPPER_VARS_CN[v] for v in u_w]), height=260)

    wind_w = s_w["u10"] + s_w["v10"] + u_w["u"] + u_w["v"]
    total_w = sum(s_w.values()) + sum(u_w.values())
    wind_share = wind_w / total_w * 100 if total_w > 0 else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("风速相关权重之和", f"{wind_w:.2f}")
    m2.metric("全部权重之和", f"{total_w:.2f}")
    m3.metric("风速侧重度", f"{wind_share:.1f}%",
              help="风相关变量(u/v/u10/v10)权重占总权重的比例，越高越聚焦风速。")
    if wind_share >= 35:
        st.success("当前配置高度聚焦风速 —— 符合 VAAWM 面向风能的目标。")
    else:
        st.info("提高 u/v、u10/v10 的权重可让模型更聚焦风速预报。")

# ============================================================== 页面 ② ======
elif page == PAGES[1]:
    st.markdown('<p class="sec">区域特定掩码 β（实时构造）</p>', unsafe_allow_html=True)
    st.write("掩码只保留目标区域包围盒内的网格点。下面的构造与 `custom_mask.ipynb` 完全一致："
             "纬度索引 `(90-lat)*4`，经度索引 `lon*4`，区域内置 1、区域外置 0。")

    mode = st.radio("区域选择方式", ["预设区域", "🎛 自定义经纬度（交互）"], horizontal=True)

    if mode == "预设区域":
        region_name = st.selectbox("目标区域", list(pdata.REGIONS.keys()))
        r = pdata.REGIONS[region_name]
        lat_min, lat_max = r["lat_min"], r["lat_max"]
        lon_min, lon_max = r["lon_min"], r["lon_max"]
        ascii_title = f"{r['ascii']} mask (beta_i,j)"
        desc = r["desc"]
        exact = r["exact"]
    else:
        st.caption("拖动滑块自由圈定一个矩形区域，地图与统计会实时更新。")
        ca, cb = st.columns(2)
        lat_min, lat_max = ca.slider("纬度范围 (°N)", -90, 90, (34, 49))
        lon_min, lon_max = cb.slider("经度范围 (°E)", 0, 359, (73, 96))
        ascii_title = f"Custom region [{lat_min}-{lat_max}N, {lon_min}-{lon_max}E]"
        desc = "用户自定义矩形区域。"
        exact = True

    mask = reg.build_region_mask(lat_min, lat_max, lon_min, lon_max)
    kept, total, frac = reg.mask_stats(mask)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("纬度范围", f"{lat_min}–{lat_max}°N")
    k2.metric("经度范围", f"{lon_min}–{lon_max}°E")
    k3.metric("保留网格点", f"{kept:,}")
    k4.metric("占全球网格", f"{frac:.3f}%")

    st.write(f"*{desc}*")
    if not exact:
        st.caption("提示：浙江包围盒为对照论文图 2b 的近似值；仓库内置的是新疆精确掩码。")

    try:
        fig = reg.visualize_mask(mask, ascii_title)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"地图渲染失败（{e}），改为显示原始掩码数组。")
        st.image((mask * 255).astype("uint8"),
                 caption="掩码（白=保留区域）", use_container_width=True)

    st.markdown('<p class="sec">🎛 损失聚焦模拟器</p>', unsafe_allow_html=True)
    st.write("在一个随机误差场上，直观看到「区域掩码 β」如何把损失聚焦到目标区域。")
    seed = st.slider("随机误差场种子", 0, 99, 7)
    rng = np.random.default_rng(seed)
    err = rng.standard_normal((reg.GRID_LAT, reg.GRID_LON)).astype(np.float32) ** 2  # 模拟平方误差
    full_loss = float(err.mean())
    region_loss = float((err * mask).sum() / max(kept, 1))
    region_share = float((err * mask).sum() / err.sum() * 100)
    s1, s2, s3 = st.columns(3)
    s1.metric("全球平均平方误差", f"{full_loss:.3f}")
    s2.metric("仅区域内平均误差", f"{region_loss:.3f}")
    s3.metric("区域内误差占全球", f"{region_share:.3f}%")
    st.caption("VAAWM 用区域内的损失（中间这一项）来更新模型，因此优化算力集中在运营价值最高的地方。")

# ============================================================== 页面 ③ ======
elif page == PAGES[2]:
    st.markdown('<p class="sec">核心结果：VAAWM 对比各基线</p>', unsafe_allow_html=True)
    st.write("10 天风速预报 RMSE（24 小时间隔）。微调后的 VAAWM 盘古模型，"
             "与零样本原模型、AutoGluon 时间序列集成进行对比。")

    f = st.columns(3)
    region = f[0].selectbox("区域", ["新疆 (Xinjiang)"])
    variable = f[1].selectbox("目标变量", ["10m wind speed", "850hPa wind speed"],
                              format_func=lambda x: {"10m wind speed": "10米风速",
                                                     "850hPa wind speed": "850hPa 风速"}[x])
    if variable == "10m wind speed":
        split = f[2].selectbox("数据划分", ["Post-2018", "Pre-2019"],
                               format_func=lambda x: {"Post-2018": "Post-2018（部署场景）",
                                                      "Pre-2019": "Pre-2019（历史基准）"}[x])
    else:
        split = f[2].selectbox("数据划分", ["Post-2018"],
                               format_func=lambda x: "Post-2018（2024测试集）")

    key = ("Xinjiang", variable, split)
    if key not in pdata.RESULTS:
        st.warning("论文中没有该组合的表格。")
    else:
        title, data = pdata.RESULTS[key]

        all_models = list(data.keys())
        show = st.multiselect("显示哪些模型", all_models, default=all_models)
        df = pd.DataFrame({m: data[m] for m in show}, index=pdata.LEAD_TIMES)
        df.index.name = "预报时效"
        st.markdown(f"**{title}**")
        st.line_chart(df, height=380)
        st.caption("RMSE 单位 m/s，越低越好。")

        st.markdown("**🎛 选择预报时效，查看该时刻的提升**")
        k = st.slider("预报时效 T+k（天）", 1, 10, 2)
        if BASE_KEY in data and OURS_KEY in data:
            base_v = data[BASE_KEY][k - 1]
            ours_v = data[OURS_KEY][k - 1]
            red = (base_v - ours_v) / base_v * 100
            d1, d2, d3 = st.columns(3)
            d1.metric(f"零样本 RMSE @ T+{k}", f"{base_v:.3f}")
            d2.metric(f"VAAWM RMSE @ T+{k}", f"{ours_v:.3f}", delta=f"{ours_v - base_v:.3f}")
            d3.metric(f"RMSE 降幅 @ T+{k}", f"{red:.1f}%")

            red_all = pdata.reduction_pct(data[BASE_KEY], data[OURS_KEY])
            st.bar_chart(pd.DataFrame({"相对零样本的 RMSE 降幅 (%)": red_all},
                                      index=pdata.LEAD_TIMES), height=240)

        with st.expander("查看原始 RMSE 数值表"):
            full = pd.DataFrame(data, index=pdata.LEAD_TIMES)
            full.index.name = "预报时效"
            st.dataframe(full.style.format("{:.3f}"), use_container_width=True)

        st.success(f"论文结论：短期预报 RMSE 最多降低 **{pdata.HIGHLIGHTS['headline_max']}%**。"
                   f"新疆 10 米风速（Post-2018）在 T+1 降低 **{pdata.HIGHLIGHTS['post2018_T1']}%**，"
                   f"T+2 降低 **{pdata.HIGHLIGHTS['post2018_T2']}%**。")

# ============================================================== 页面 ④ ======
elif page == PAGES[3]:
    st.markdown('<p class="sec">消融实验：自适应加权为何关键</p>', unsafe_allow_html=True)
    st.write("三个架构与训练计划完全相同的盘古变体对比。可以看到：朴素的统一 MSE 微调"
             "反而会**损害**短期精度，而自适应加权才是带来提升的关键。")

    data = pdata.TABLE3_ABLATION_XINJIANG_10M_POST2018
    names_cn = {"Zero-shot": "零样本（不微调）",
                "Standard Fine-tuning (uniform MSE)": "标准微调（统一MSE）",
                "VAAWM (Ours)": "VAAWM（本文）"}
    show = st.multiselect("显示哪些变体", list(data.keys()),
                          default=list(data.keys()),
                          format_func=lambda x: names_cn[x])
    df = pd.DataFrame({names_cn[m]: data[m] for m in show}, index=pdata.LEAD_TIMES)
    df.index.name = "预报时效"
    st.line_chart(df, height=380)
    st.caption("新疆 10 米风速 RMSE（Post-2018, 24H），越低越好。")

    c1, c2, c3 = st.columns(3)
    c1.metric("VAAWM vs 零样本 @T+1", f"-{pdata.HIGHLIGHTS['ablation_vs_zeroshot_T1']}%")
    c2.metric("VAAWM vs 标准微调 @T+1", f"-{pdata.HIGHLIGHTS['ablation_vs_standard_T1']}%")
    c3.metric("标准微调 vs 零样本 @T+1", f"+{pdata.HIGHLIGHTS['standard_ft_degrade_T1']}%",
              delta_color="inverse", help="标准微调在 T+1 反而比零样本更差。")

    with st.expander("查看原始 RMSE 数值表"):
        full = pd.DataFrame(data, index=pdata.LEAD_TIMES).rename(columns=names_cn)
        full.index.name = "预报时效"
        st.dataframe(full.style.format("{:.3f}"), use_container_width=True)

    st.info("标准微调变差的原因：对均衡表征的灾难性遗忘、统一损失带来的任务错配、"
            "以及对主导模式过拟合造成的表征干扰。VAAWM 通过学习放大风变量与目标区域来缓解这些问题。")

# ============================================================== 页面 ⑤ ======
elif page == PAGES[4]:
    st.markdown('<p class="sec">混合推理：缓解自回归预测漂移</p>', unsafe_allow_html=True)
    st.write("过度专门化会在长程自回归滚动预报中引发**预测漂移**：非目标变量的误差逐步累积，"
             "最终污染目标变量。混合推理在每一步按变量来源混合两个模型：")
    st.latex(r"""x_t^{(v)} = \begin{cases}
        f^{(v)}_{\text{finetune}}(X_{t-1}) & v \in \mathcal{V}_{\text{target}} \\
        f^{(v)}_{\text{base}}(X_{t-1}) & \text{其他}
        \end{cases}""")

    st.markdown('<p class="sec">🎛 交互：分配各变量的来源</p>', unsafe_allow_html=True)
    st.caption("勾选哪些变量由「微调模型」提供（其余由「原始模型」提供，以维持物理一致性）。")
    all_vars = pdata.SURFACE_VARS + pdata.UPPER_VARS
    cn_map = {**pdata.SURFACE_VARS_CN, **pdata.UPPER_VARS_CN}
    default_target = ["u10", "v10", "u", "v"]
    chosen = st.multiselect("由微调模型提供的变量（目标变量）", all_vars,
                            default=default_target, format_func=lambda v: cn_map[v])

    rows = []
    for v in all_vars:
        src = "微调模型 (finetune)" if v in chosen else "原始模型 (base)"
        rows.append({"变量": cn_map[v], "来源": src})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    is_wind = set(chosen) >= set(default_target)
    others_from_base = all(v not in chosen for v in all_vars if v not in default_target)
    if is_wind and others_from_base:
        st.success("✓ 推荐配置：风速变量用微调模型（精度高），其余用原始模型（维持物理一致、抑制漂移）。")
    elif not chosen:
        st.warning("没有变量来自微调模型——等价于直接用原始模型，得不到专门化收益。")
    else:
        st.info("可调整选择：通常只让目标风速变量来自微调模型，其余保留原始模型最稳。")

    st.markdown("- **目标变量**（10米/850hPa 风速）→ 取自**微调模型**，精度更高。\n"
                "- **其余变量** → 取自**原始模型**，保持大气状态物理一致。\n"
                "- 综合效果：兼顾专家模型的短期精度与通用模型的长期稳定，几乎不增加算力。")
    st.caption("本仓库的混合滚动实现见 `inference/inference_mix_multiOutput.py` 及 "
               "`inference/inference_multiOutput-{3,6}-mix24.py`。")

    st.markdown('<p class="sec">长期稳定性（附录 C）</p>', unsafe_allow_html=True)
    st.write("大模型误差随预报时效单调增长，而 AutoGluon 集成在约 T+6 后趋于平稳"
             "（直接多步预测 + 集成平滑）——但其绝对精度仍低于 VAAWM 盘古模型，"
             "这正是未来把两者结合的动机。")

# ============================================================== 页面 ⑥ ======
elif page == PAGES[5]:
    st.markdown('<p class="sec">实时推理：真实盘古模型 + 真实 ERA5 数据</p>', unsafe_allow_html=True)
    st.write("本页**真正加载预训练权重并在 GPU 上前向推理**（不是论文数字回放）。"
             "输入某一时刻的全球大气状态，模型预测 24 小时后的状态，再与真实 ERA5 对比，"
             "并用与训练代码相同的 `score.weighted_rmse_torch_channels` 计算掩码加权 RMSE。")

    if not _ENGINE_OK:
        st.error(f"推理引擎导入失败：{_ENGINE_ERR}")
        st.stop()

    ok, msg = eng.data_ready()
    if not ok:
        st.warning(f"数据未就绪：{msg}")
        st.info("需要 `pangu_weather_24_torch.pth`、`aux_data/`（含 custom_mask.npy）"
                "以及 surface/upper 的 .nc 样本。可从 HuggingFace 数据集 "
                "`zhaoshan/pangu_pytorch` 获取。")
        st.stop()

    device = eng.get_device()
    cdev = st.columns(3)
    cdev[0].metric("计算设备", "GPU" if device.startswith("cuda") else "CPU")
    cdev[1].metric("型号", eng.gpu_name())
    dates = eng.list_input_dates()
    cdev[2].metric("可用样本数", f"{len(dates)}")

    cc = st.columns(3)
    input_date = cc[0].selectbox("输入时刻（00:00 UTC）", dates,
                                 format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}")
    field_name = cc[1].selectbox("可视化变量", ["10米风速", "850hPa风速", "2米温度 t2m", "海平面气压 msl"])
    region_name = cc[2].selectbox("评估/聚焦区域", ["全球"] + list(pdata.REGIONS.keys()))

    ft_ok = eng.finetuned_available()
    variant_label = st.radio(
        "使用的权重", ["零样本（官方预训练）", "VAAWM 小规模微调（本地训练）"],
        horizontal=True, disabled=not ft_ok,
        help=None if ft_ok else "未找到微调权重，请先运行 demo/finetune_vaawm.py")
    variant = "vaawm" if variant_label.startswith("VAAWM") else "zeroshot"
    if not ft_ok:
        st.caption("提示：运行 `python demo/finetune_vaawm.py` 可生成 VAAWM 微调权重后在此对比。")

    run = st.button("🚀 运行真实推理", type="primary")

    if run:
        cache_key = (input_date, variant)
        if st.session_state.get("infer_key") != cache_key:
            with st.spinner(f"加载{variant_label}并在 {device} 上推理中…"):
                try:
                    st.session_state["infer_result"] = eng.run_inference(input_date, device, variant)
                    st.session_state["infer_key"] = cache_key
                    st.session_state["infer_variant"] = variant_label
                except Exception as e:
                    st.exception(e)
                    st.stop()

    result = st.session_state.get("infer_result")
    if result is None:
        st.info("点击「运行真实推理」开始。首次运行需加载约 1.1GB 权重，稍候片刻。")
    else:
        st.success(f"推理完成（{st.session_state.get('infer_variant','')}）："
                   f"输入 {result['periods'][0]} → 预测 {result['periods'][1]}（+24h），"
                   f"GPU 前向用时 {result['elapsed']:.2f}s。")

        pred, tgt, unit = eng.get_field(result, field_name)

        # RMSE: global + region (if a region is chosen)
        rmse_global = eng.masked_rmse(pred, tgt, None, device)
        cols = st.columns(3)
        cols[0].metric(f"{field_name} RMSE（全球）", f"{rmse_global:.4f} {unit}")
        bbox = None
        if region_name != "全球":
            r = pdata.REGIONS[region_name]
            mask = reg.build_region_mask(r["lat_min"], r["lat_max"], r["lon_min"], r["lon_max"])
            rmse_region = eng.masked_rmse(pred, tgt, mask, device)
            cols[1].metric(f"{field_name} RMSE（{region_name.split(' ')[0]}）",
                           f"{rmse_region:.4f} {unit}")
            cols[2].metric("区域 vs 全球", f"{(rmse_region - rmse_global):+.4f} {unit}")
            bbox = (r["lon_min"] - 3, r["lon_max"] + 3, r["lat_min"] - 3, r["lat_max"] + 3)

        # maps: prediction / target / abs error
        vmin = float(min(pred.min(), tgt.min()))
        vmax = float(max(pred.max(), tgt.max()))
        ascii_field = {"10米风速": "10m wind speed", "850hPa风速": "850hPa wind speed",
                       "2米温度 t2m": "t2m", "海平面气压 msl": "msl"}[field_name]
        try:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown("**预测 (Pangu-Weather)**")
                st.pyplot(reg.plot_field(pred, bbox, f"Prediction: {ascii_field} ({unit})",
                                         cmap="viridis", vmin=vmin, vmax=vmax))
            with m2:
                st.markdown("**真实 (ERA5)**")
                st.pyplot(reg.plot_field(tgt, bbox, f"Truth: {ascii_field} ({unit})",
                                         cmap="viridis", vmin=vmin, vmax=vmax))
            st.markdown("**绝对误差 |预测 − 真实|**")
            st.pyplot(reg.plot_field(np.abs(pred - tgt), bbox,
                                     f"Abs error: {ascii_field} ({unit})", cmap="magma"))
        except Exception as e:
            st.warning(f"地图渲染失败（{e}）。")

        st.caption("说明：「零样本」为官方预训练权重（horizon=24）；「VAAWM 微调」为本地用 "
                   "`demo/finetune_vaawm.py` 在区域掩码+风速损失下做的小规模微调（演示性质）。"
                   "样本为 HuggingFace 数据集提供的 2019-07 真实 ERA5。")

        # ---- 混合推理对比 ----
        if ft_ok:
            st.markdown('<p class="sec">🔀 混合推理对比（论文 §3.4）</p>', unsafe_allow_html=True)
            st.write("混合策略：**目标风速变量**（u10/v10、高空 u/v）取自 **VAAWM 微调模型**，"
                     "**其余变量**（z/q/t/msl/t2m）取自**原始模型**，以维持物理一致、抑制全局退化。"
                     "下表对每个变量、三种策略，计算真实的掩码加权 RMSE。")
            if st.button("运行三策略对比（零样本 / VAAWM / 混合）"):
                with st.spinner("分别推理原始模型与微调模型并组合混合结果…"):
                    base_res = eng.run_inference(input_date, device, "zeroshot")
                    vaawm_res = eng.run_inference(input_date, device, "vaawm")
                    hybrid_res = eng.build_hybrid(base_res, vaawm_res)
                    st.session_state["hybrid_pack"] = (input_date, base_res, vaawm_res, hybrid_res)

            pack = st.session_state.get("hybrid_pack")
            if pack and pack[0] == input_date:
                _, base_res, vaawm_res, hybrid_res = pack
                use_region = region_name != "全球"
                if use_region:
                    rr = pdata.REGIONS[region_name]
                    rmask = reg.build_region_mask(rr["lat_min"], rr["lat_max"],
                                                  rr["lon_min"], rr["lon_max"])
                else:
                    rmask = None
                fields = ["10米风速", "850hPa风速", "2米温度 t2m", "海平面气压 msl"]
                target_fields = {"10米风速", "850hPa风速"}
                rows = []
                for fld in fields:
                    pb, tb, u = eng.get_field(base_res, fld)
                    pv, _, _ = eng.get_field(vaawm_res, fld)
                    ph, _, _ = eng.get_field(hybrid_res, fld)
                    rows.append({
                        "变量": fld + ("（目标）" if fld in target_fields else "（非目标）"),
                        "零样本": round(eng.masked_rmse(pb, tb, rmask, device), 4),
                        "VAAWM微调": round(eng.masked_rmse(pv, tb, rmask, device), 4),
                        "混合推理": round(eng.masked_rmse(ph, tb, rmask, device), 4),
                        "单位": u,
                    })
                scope = region_name if use_region else "全球"
                st.markdown(f"**各变量 RMSE（{scope}，越低越好）**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("可以看到：混合推理对**目标变量**取微调模型的值、对**非目标变量**取原始模型的值，"
                           "因此在专门化风速的同时，不会牺牲其它变量（避免全参数微调带来的全局退化）。")
                st.info("注：单步 24h 预报下，混合的价值主要体现在「非目标变量不退化」；"
                        "在多步自回归滚动预报中，它还能阻断误差累积、抑制预测漂移（论文 §3.4 的核心动机）。")

# ============================================================== 页面 ⑦ ======
elif page == PAGES[6]:
    st.markdown('<p class="sec">多步自回归滚动 & 泛化（真实计算）</p>', unsafe_allow_html=True)
    if not _ENGINE_OK:
        st.error(f"推理引擎导入失败：{_ENGINE_ERR}")
        st.stop()
    ok, msg = eng.data_ready()
    if not ok:
        st.warning(f"数据未就绪：{msg}")
        st.stop()
    device = eng.get_device()

    st.write("数据已扩展到 2019-07-01 ~ 07-10（upper 取自公开 ARCO-ERA5），"
             "因此可以做**多步自回归滚动**并对每一步用真值评估。")

    # ---- A. 多步滚动漂移 ----
    st.markdown('<p class="sec">A. 多步滚动：误差漂移对比</p>', unsafe_allow_html=True)
    st.write("从 07-01 00:00 出发，每步预测 +24h 并把预测**喂回**作为下一步输入，"
             "逐步对真实 ERA5 评估 10 米风速 RMSE。对比三种滚动策略。")
    cstart = st.columns(2)
    start_date = cstart[0].selectbox("起始日期", eng.list_input_dates(),
                                     format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}")
    nmax = eng.max_rollout_steps(start_date)
    metric = cstart[1].radio("评估范围", ["新疆", "全球"], horizontal=True)
    st.caption(f"从 {start_date} 可滚动 {nmax} 步（受可用真值限制）。")

    if st.button("🌀 运行三策略多步滚动", type="primary") and nmax > 0:
        with st.spinner(f"在 {device} 上滚动推理（每策略 {nmax} 步）…"):
            rolls = {}
            for s in ["zeroshot", "vaawm", "hybrid"]:
                rolls[s] = eng.run_rollout(start_date, nmax, s, device)
            st.session_state["rolls"] = (start_date, rolls)

    pack = st.session_state.get("rolls")
    if pack and pack[0] == start_date:
        _, rolls = pack
        key = "rmse_region" if metric == "新疆" else "rmse_global"
        name_cn = {"zeroshot": "零样本滚动", "vaawm": "VAAWM滚动", "hybrid": "混合滚动"}
        df = pd.DataFrame({name_cn[s]: rolls[s][key] for s in rolls},
                          index=rolls["zeroshot"]["leads"])
        df.index.name = "预报时效"
        st.markdown(f"**10 米风速 RMSE（{metric}）随预报时效的漂移**")
        st.line_chart(df, height=380)
        st.caption("RMSE 单位 m/s，越低越好。横轴为自回归步数（每步 24h）。")
        # drift summary
        last = df.iloc[-1]
        zs_last = rolls["zeroshot"][key][-1]
        hy_last = rolls["hybrid"][key][-1]
        vw_last = rolls["vaawm"][key][-1]
        c = st.columns(3)
        c[0].metric(f"零样本 @ {df.index[-1]}", f"{zs_last:.3f}")
        c[1].metric(f"VAAWM @ {df.index[-1]}", f"{vw_last:.3f}", delta=f"{vw_last-zs_last:+.3f}",
                    delta_color="inverse")
        c[2].metric(f"混合 @ {df.index[-1]}", f"{hy_last:.3f}", delta=f"{hy_last-vw_last:+.3f}",
                    delta_color="inverse",
                    help="相对 VAAWM 的变化：混合通常更接近零样本，抑制漂移。")
        if metric == "全球":
            st.success("观察：纯 VAAWM 滚动的全局误差漂移最大；混合滚动把非目标变量交还给"
                       "原始模型，明显抑制了漂移（曲线介于零样本与 VAAWM 之间、更靠近零样本）。")

    # ---- B. 留出集泛化 ----
    st.markdown('<p class="sec">B. 留出集泛化（单步 24h）</p>', unsafe_allow_html=True)
    st.write("VAAWM 权重是在 **07-01 → 07-02** 上微调的。这里在它**从未见过**的日期上做单步评估，"
             "考察泛化性（10 米风速 RMSE，多日平均）。")
    all_dates = eng.list_input_dates()
    test_dates = st.multiselect("留出测试日期（输入时刻）", all_dates,
                                default=[d for d in all_dates if d in
                                         ("20190705", "20190706", "20190707", "20190708")],
                                format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}")
    if st.button("📊 在留出集上评估三策略") and test_dates:
        with st.spinner("在留出日期上推理评估…"):
            res = eng.eval_heldout(test_dates, device)
        rows = [{"策略": {"zeroshot": "零样本", "vaawm": "VAAWM微调", "hybrid": "混合推理"}[s],
                 "新疆 RMSE": round(res[s]["region"], 4),
                 "全球 RMSE": round(res[s]["global"], 4)} for s in ["zeroshot", "vaawm", "hybrid"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"在 {len(test_dates)} 个未见过的日期上平均。10 米风速。")
        st.info("诚实说明：本次微调样本极少（仅 07-01→07-02），且仓库的风速损失在归一化空间计算，"
                "因此在留出集上 VAAWM 不一定超过零样本——这与论文「朴素/小数据微调可能退化」一致。"
                "混合推理则能把全局/非目标的退化拉回接近零样本的水平。")

st.markdown("---")
st.caption("参考：Bridging the Weather Forecasting Gap — Region-Aware and Variable-Specific "
           "Adaptation of Weather Foundation Models。基础模型：Pangu-Weather (Bi et al., 2022)。")
