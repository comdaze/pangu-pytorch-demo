import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import sys
import torch
import onnxruntime as ort
import xarray as xr
from datetime import datetime, timedelta
from PIL import Image
import io

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入项目中的模块
from era5_data.config import cfg
from era5_data import utils_data, utils, score
from models.pangu_sample import get_wind_speed
from utils import visualize_map, calculate_metrics, load_sample_data

# 设置页面配置
st.set_page_config(
    page_title="PanGu Weather Forecast Demo", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #2563EB;
    }
    .info-text {
        font-size: 1rem;
        color: #4B5563;
    }
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<p class="main-header">PanGu Weather Forecast Demo</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">基于PanGu-Weather模型的天气预报可视化演示</p>', unsafe_allow_html=True)

# 加载项目logo
try:
    logo_path = os.path.join(parent_dir, "fig", "VIS.png")
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.sidebar.image(logo, width=300, caption="PanGu Weather Model")
except Exception as e:
    st.sidebar.warning(f"无法加载项目图片: {e}")

# 侧边栏控制
with st.sidebar:
    st.markdown('<p class="sub-header">控制面板</p>', unsafe_allow_html=True)
    
    # 数据选择
    data_path = st.text_input("数据路径", value=cfg.PG_INPUT_PATH)
    
    # 模型选择
    st.markdown("### 模型选择")
    horizon_options = {"1小时预测": 1, "3小时预测": 3, "6小时预测": 6, "24小时预测": 24}
    selected_horizon = st.selectbox("预测时长", list(horizon_options.keys()))
    horizon = horizon_options[selected_horizon]
    
    # 日期选择
    st.markdown("### 时间设置")
    start_date = st.date_input("开始日期", datetime(2018, 1, 1))
    start_hour = st.selectbox("开始时间", ["00:00", "12:00"])
    
    # 变量选择
    st.markdown("### 数据可视化设置")
    data_type = st.radio("数据类型", ["高空数据", "地表数据"])
    
    if data_type == "高空数据":
        var_options = ["z (位势高度)", "q (比湿)", "t (温度)", "u (纬向风)", "v (经向风)", "wind_speed (风速)"]
        level = st.slider("气压层级 (hPa)", 50, 1000, 500, 50)
        level_idx = (1000 - level) // 50  # 转换为索引
    else:  # 地表数据
        var_options = ["t2m (2米温度)", "msl (海平面气压)", "u10 (10米纬向风)", "v10 (10米经向风)", "wind10 (10米风速)"]
        level_idx = None
    
    selected_var = st.selectbox("变量", var_options)
    
    # 可视化设置
    colormap = st.selectbox("颜色方案", 
                           ["viridis", "plasma", "inferno", "magma", "cividis", 
                            "jet", "rainbow", "cool", "hot", "coolwarm", "RdBu_r"])
    
    # 运行按钮
    run_button = st.button("运行预测", type="primary")

# 主界面
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="sub-header">预测结果</p>', unsafe_allow_html=True)
    result_placeholder = st.empty()

with col2:
    st.markdown('<p class="sub-header">实际数据</p>', unsafe_allow_html=True)
    target_placeholder = st.empty()

# 性能指标
metrics_expander = st.expander("性能指标", expanded=False)
with metrics_expander:
    metrics_placeholder = st.empty()

# 数据下载
download_expander = st.expander("结果导出", expanded=False)
with download_expander:
    download_placeholder = st.empty()

# 会话状态初始化
if 'output' not in st.session_state:
    st.session_state.output = None
if 'output_surface' not in st.session_state:
    st.session_state.output_surface = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'target_surface' not in st.session_state:
    st.session_state.target_surface = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# 运行预测
if run_button:
    with st.spinner("正在加载模型..."):
        try:
            # 在实际应用中，这里应该加载真实的模型
            # 简化示例，我们假设模型已经加载
            st.info("注意：这是一个演示版本，使用模拟数据。实际应用中应连接到真实模型。")
            progress_bar = st.progress(0)
            progress_bar.progress(25)
            
            # 在实际应用中，这里应该加载真实数据
            # 简化示例，我们使用随机数据
            input_data, input_surface, target, target_surface = load_sample_data()
            progress_bar.progress(50)
            
            # 模拟模型预测过程
            st.info("正在进行天气预报计算...")
            progress_bar.progress(75)
            
            # 在实际应用中，这里应该是真实的模型预测
            # 简化示例，我们使用随机数据作为输出
            output = target + np.random.normal(0, 0.1, target.shape).astype(np.float32)
            output_surface = target_surface + np.random.normal(0, 0.1, target_surface.shape).astype(np.float32)
            
            # 保存到会话状态
            st.session_state.output = output
            st.session_state.output_surface = output_surface
            st.session_state.target = target
            st.session_state.target_surface = target_surface
            
            # 计算性能指标
            st.session_state.metrics = calculate_metrics(
                torch.from_numpy(output), 
                torch.from_numpy(target),
                torch.from_numpy(output_surface),
                torch.from_numpy(target_surface)
            )
            
            progress_bar.progress(100)
            st.success("预测完成！")
            
        except Exception as e:
            st.error(f"处理失败: {e}")

# 如果有数据，显示结果
if st.session_state.output is not None and st.session_state.target is not None:
    # 解析选择的变量
    var_name = selected_var.split(" ")[0]
    
    # 准备可视化
    if data_type == "高空数据":
        # 高空数据可视化
        var_idx = {"z": 0, "q": 1, "t": 2, "u": 3, "v": 4, "wind_speed": 5}.get(var_name, 0)
        
        # 创建预测结果图
        fig_pred = visualize_map(
            st.session_state.output[var_idx, level_idx], 
            title=f"预测 - {selected_var} (Level: {level} hPa)",
            colormap=colormap
        )
        result_placeholder.pyplot(fig_pred)
        
        # 创建实际数据图
        fig_true = visualize_map(
            st.session_state.target[var_idx, level_idx], 
            title=f"实际 - {selected_var} (Level: {level} hPa)",
            colormap=colormap
        )
        target_placeholder.pyplot(fig_true)
        
    else:  # 地表数据
        # 地表数据可视化
        var_idx = {"t2m": 2, "msl": 3, "u10": 0, "v10": 1, "wind10": 4}.get(var_name, 0)
        
        if var_name == "wind10":
            # 计算风速
            u10_pred = st.session_state.output_surface[0]
            v10_pred = st.session_state.output_surface[1]
            wind10_pred = np.sqrt(u10_pred**2 + v10_pred**2)
            
            u10_true = st.session_state.target_surface[0]
            v10_true = st.session_state.target_surface[1]
            wind10_true = np.sqrt(u10_true**2 + v10_true**2)
            
            # 创建预测结果图
            fig_pred = visualize_map(
                wind10_pred, 
                title=f"预测 - {selected_var}",
                colormap=colormap
            )
            result_placeholder.pyplot(fig_pred)
            
            # 创建实际数据图
            fig_true = visualize_map(
                wind10_true, 
                title=f"实际 - {selected_var}",
                colormap=colormap
            )
            target_placeholder.pyplot(fig_true)
        else:
            # 创建预测结果图
            fig_pred = visualize_map(
                st.session_state.output_surface[var_idx], 
                title=f"预测 - {selected_var}",
                colormap=colormap
            )
            result_placeholder.pyplot(fig_pred)
            
            # 创建实际数据图
            fig_true = visualize_map(
                st.session_state.target_surface[var_idx], 
                title=f"实际 - {selected_var}",
                colormap=colormap
            )
            target_placeholder.pyplot(fig_true)
    
    # 显示性能指标
    with metrics_placeholder.container():
        if st.session_state.metrics:
            metrics = st.session_state.metrics
            
            # 创建两列布局
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                st.markdown("### 高空数据指标")
                st.write(f"Z (位势高度) RMSE: {metrics['rmse_z']:.4f}")
                st.write(f"Q (比湿) RMSE: {metrics['rmse_q']:.4f}")
                st.write(f"T (温度) RMSE: {metrics['rmse_t']:.4f}")
                st.write(f"U (纬向风) RMSE: {metrics['rmse_u']:.4f}")
                st.write(f"V (经向风) RMSE: {metrics['rmse_v']:.4f}")
                st.write(f"风速 RMSE: {metrics['rmse_wind']:.4f}")
            
            with m_col2:
                st.markdown("### 地表数据指标")
                st.write(f"T2M (2米温度) RMSE: {metrics['rmse_t2m']:.4f}")
                st.write(f"MSL (海平面气压) RMSE: {metrics['rmse_msl']:.4f}")
                st.write(f"U10 (10米纬向风) RMSE: {metrics['rmse_u10']:.4f}")
                st.write(f"V10 (10米经向风) RMSE: {metrics['rmse_v10']:.4f}")
                st.write(f"10米风速 RMSE: {metrics['rmse_wind10']:.4f}")
    
    # 提供下载选项
    with download_placeholder.container():
        # 创建两列布局
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            st.markdown("### 下载预测图像")
            
            # 将图像转换为字节流
            buf_pred = io.BytesIO()
            fig_pred.savefig(buf_pred, format='png', dpi=300, bbox_inches='tight')
            buf_pred.seek(0)
            
            # 提供下载按钮
            st.download_button(
                label="下载预测图像 (PNG)",
                data=buf_pred,
                file_name=f"pangu_prediction_{var_name}_{datetime.now().strftime('%Y%m%d%H%M')}.png",
                mime="image/png"
            )
        
        with d_col2:
            st.markdown("### 下载数据")
            
            # 创建CSV数据
            if data_type == "高空数据":
                csv_data = io.StringIO()
                np.savetxt(csv_data, st.session_state.output[var_idx, level_idx], delimiter=',')
                csv_data.seek(0)
                
                st.download_button(
                    label="下载预测数据 (CSV)",
                    data=csv_data,
                    file_name=f"pangu_data_{var_name}_{level}hPa_{datetime.now().strftime('%Y%m%d%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                csv_data = io.StringIO()
                if var_name == "wind10":
                    u10_pred = st.session_state.output_surface[0]
                    v10_pred = st.session_state.output_surface[1]
                    wind10_pred = np.sqrt(u10_pred**2 + v10_pred**2)
                    np.savetxt(csv_data, wind10_pred, delimiter=',')
                else:
                    np.savetxt(csv_data, st.session_state.output_surface[var_idx], delimiter=',')
                csv_data.seek(0)
                
                st.download_button(
                    label="下载预测数据 (CSV)",
                    data=csv_data,
                    file_name=f"pangu_data_{var_name}_{datetime.now().strftime('%Y%m%d%H%M')}.csv",
                    mime="text/csv"
                )
else:
    # 如果没有数据，显示提示信息
    with result_placeholder.container():
        st.info("请点击'运行预测'按钮开始预测")
    
    with target_placeholder.container():
        st.info("预测完成后将显示实际数据")

# 添加页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>PanGu Weather Forecast Demo | 基于 Streamlit 开发</p>
    <p>参考: Bi et al. (2023) - Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast</p>
</div>
""", unsafe_allow_html=True)