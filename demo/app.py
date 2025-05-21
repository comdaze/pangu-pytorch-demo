import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import sys
import torch
import onnxruntime as ort
import xarray as xr
from models.pangu_model import PanguModel
from datetime import datetime, timedelta
from PIL import Image
import io

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入项目中的模块
from era5_data.config import cfg

# --- BEGIN PATH OVERRIDES FOR TESTING ---
cfg.PG_INPUT_PATH = "/app/pangu_test_environment/era5_data_sample"
cfg.PG.BENCHMARK.PRETRAIN_1_torch = "/app/pangu_test_environment/pretrained_model/pangu_weather_1_torch.pth"
cfg.PG.BENCHMARK.PRETRAIN_3_torch = "/app/pangu_test_environment/pretrained_model/pangu_weather_3_torch.pth"
cfg.PG.BENCHMARK.PRETRAIN_6_torch = "/app/pangu_test_environment/pretrained_model/pangu_weather_6_torch.pth"
cfg.PG.BENCHMARK.PRETRAIN_24_torch = "/app/pangu_test_environment/pretrained_model/pangu_weather_24_torch.pth"
# --- END PATH OVERRIDES FOR TESTING ---

from era5_data import utils_data, utils, score
from models.pangu_sample import get_wind_speed
from utils import visualize_map, calculate_metrics # Removed load_sample_data
# Removed redundant import of utils_data
from torch.utils.data import DataLoader

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
if 'model' not in st.session_state:
    st.session_state.model = None
if 'aux_constants' not in st.session_state:
    st.session_state.aux_constants = None
if 'current_horizon' not in st.session_state:
    st.session_state.current_horizon = None

# 运行预测
if run_button:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.session_state.device = device # Store device in session state
    
    with st.spinner("正在加载模型..."):
        try:
            model_changed = st.session_state.model is None or st.session_state.current_horizon != horizon
            if model_changed:
                st.session_state.current_horizon = horizon
                # st.info("注意：这是一个演示版本，使用模拟数据。实际应用中应连接到真实模型。") # Removed placeholder
                model = PanguModel(device=device).to(device)
                
                horizon_to_checkpoint_attr = {
                    1: cfg.PG.BENCHMARK.PRETRAIN_1_torch,
                    3: cfg.PG.BENCHMARK.PRETRAIN_3_torch,
                    6: cfg.PG.BENCHMARK.PRETRAIN_6_torch,
                    24: cfg.PG.BENCHMARK.PRETRAIN_24_torch,
                }
                
                checkpoint_path = horizon_to_checkpoint_attr.get(horizon)
                if checkpoint_path is None:
                    st.error(f"未找到对应预测时长 {horizon} 小时的模型检查点路径。")
                    st.stop()
                
                if not os.path.exists(checkpoint_path):
                    st.error(f"模型检查点文件不存在: {checkpoint_path}")
                    st.stop()

                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
                model.eval()
                st.session_state.model = model
                st.info(f"模型加载成功 ({device})：{os.path.basename(checkpoint_path)}")

            if st.session_state.aux_constants is None:
                st.session_state.aux_constants = utils_data.loadAllConstants(device=device)
                st.info("辅助常量加载成功。")

            progress_bar = st.progress(0)
            progress_bar.text("模型加载完成. (25%)")
            progress_bar.progress(25)

            # Load real data
            st.info("正在加载真实数据...")
            hour, minute = map(int, start_hour.split(':'))
            start_datetime_obj = datetime(start_date.year, start_date.month, start_date.day, hour, minute)

            # Determine freq and calculate endDate for NetCDFDataset to make dataset[0] work
            # UI offers 00:00, 12:00. If data is only at these times, '12H' is best.
            # If data is at 00, 06, 12, 18Z, then '6H' is fine.
            # Using '12H' as it matches UI options more closely and common ERA5 data structure.
            dataset_freq_str = '12H' 
            try:
                dataset_freq_hours = int(dataset_freq_str[:-1]) # Extracts '12' from '12H'
            except ValueError:
                st.error(f"无法从频率字符串 '{dataset_freq_str}' 中解析小时数。请检查格式（例如 '6H', '12H'）。")
                st.stop()

            # Calculate how many steps the horizon is in terms of dataset frequency
            horizon_steps = horizon // dataset_freq_hours 
            # To ensure dataset.length is 1 (or more), we need:
            # num_keys = dataset.length (target 1) + horizon_steps + 1
            num_keys_required = 1 + horizon_steps + 1 
            
            # NetCDFDataset's pd.date_range uses these as strings.
            # The start date for pd.date_range must be the exact user selected datetime for keys[0] to match.
            dataset_param_start_date_str = start_datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate end datetime for pd.date_range to ensure enough keys are generated.
            end_datetime_obj_for_keys = start_datetime_obj + timedelta(hours=(num_keys_required - 1) * dataset_freq_hours)
            dataset_param_end_date_str = end_datetime_obj_for_keys.strftime('%Y-%m-%d %H:%M:%S')
            
            # Optional: Add some debug info to Streamlit interface
            # st.info(f"Debug Info: nc_path='{data_path}', startDate='{dataset_param_start_date_str}', endDate='{dataset_param_end_date_str}', freq='{dataset_freq_str}', horizon={horizon}")

            dataset = utils_data.NetCDFDataset(
                nc_path=data_path,
                startDate=dataset_param_start_date_str,
                endDate=dataset_param_end_date_str,
                freq=dataset_freq_str,
                horizon=horizon,
                training=False,
                validation=False
            )
            
            # st.info(f"Debug Info: len(dataset.keys)={len(dataset.keys if hasattr(dataset, 'keys') else 'N/A')}, dataset.length={dataset.length if hasattr(dataset, 'length') else 'N/A'}")

            if not hasattr(dataset, 'keys') or len(dataset.keys) == 0:
                st.error(f"数据加载失败: NetCDFDataset 未能生成任何时间密钥 (keys)。"
                         f"参数: startDate='{dataset_param_start_date_str}', endDate='{dataset_param_end_date_str}', freq='{dataset_freq_str}'. "
                         f"请检查数据路径 ('{data_path}') 是否正确，以及该路径下是否存在与指定时间范围和频率匹配的NetCDF文件。")
                st.stop()

            if dataset.length <= 0:
                st.error(f"数据加载失败: NetCDFDataset 的计算长度 (length) 为 {dataset.length}，必须大于0。"
                         f"这通常意味着时间密钥 (keys) 的数量 {len(dataset.keys)} 不足以满足预测时长 {horizon}h (需要 {horizon_steps} 步，共 {num_keys_required} 个时间点)。"
                         f"参数: startDate='{dataset_param_start_date_str}', endDate='{dataset_param_end_date_str}', freq='{dataset_freq_str}'.")
                st.stop()
            
            # Sanity check: the first key should be our start_datetime_obj
            if hasattr(dataset, 'keys') and len(dataset.keys) > 0 and start_datetime_obj != dataset.keys[0].to_pydatetime():
                 st.warning(f"调试警告: 请求的开始时间 {start_datetime_obj} 与 dataset.keys[0] ({dataset.keys[0].to_pydatetime()}) 不匹配。这可能导致加载了错误的数据时间点。")


            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) # num_workers=0 for easier debugging if issues arise
            
            input_data_tensor, input_surface_tensor, target_data_tensor, target_surface_tensor = None, None, None, None
            try:
                # This loop should ideally run only once for batch_size=1 and correctly calculated dataset.length = 1
                for i, batch_data_tuple in enumerate(dataloader): 
                    if i > 0: 
                        st.warning("DataLoader 生成了多个批次数据，但预期只有一个。将仅使用第一个批次。")
                        break
                    input_data_tensor, input_surface_tensor, target_data_tensor, target_surface_tensor, _ = batch_data_tuple
                
                if input_data_tensor is None: 
                    st.error("未能从 DataLoader 中成功提取数据。可能是 DataLoader 为空或在迭代过程中出现问题。")
                    st.stop()

            except Exception as e:
                st.error(f"从 DataLoader 加载数据时发生错误: {e}")
                st.error(f"请仔细检查 NetCDF 文件是否存在于路径 '{data_path}'，并且这些文件是否与所选的时间范围、频率 ({dataset_freq_str}) 及预测时长 ({horizon}h) 设置兼容。")
                if hasattr(dataset, 'keys'): st.error(f"Dataset keys (first 10): {dataset.keys[:10] if dataset.keys else 'N/A'}")
                if hasattr(dataset, 'length'): st.error(f"Dataset length: {dataset.length}")
                st.stop()

            st.session_state.input_data = input_data_tensor.to(st.session_state.device)
            st.session_state.input_surface_data = input_surface_tensor.to(st.session_state.device)
            st.session_state.target_data = target_data_tensor.to(st.session_state.device) 
            st.session_state.target_surface_data = target_surface_tensor.to(st.session_state.device)

            progress_bar.text("数据加载完成. (50%)")
            progress_bar.progress(50)
            
            st.info("正在进行天气预报计算...")
            
            # Retrieve model, data, and constants from session_state
            model_to_run = st.session_state.model
            input_data_for_pred = st.session_state.input_data
            input_surface_for_pred = st.session_state.input_surface_data
            aux_constants_for_pred = st.session_state.aux_constants

            # Ensure all are on the correct device (should be already, but good practice)
            current_device = st.session_state.device
            model_to_run.to(current_device)
            input_data_for_pred = input_data_for_pred.to(current_device)
            input_surface_for_pred = input_surface_for_pred.to(current_device)
            # Aux constants are loaded to device when st.session_state.aux_constants is created.

            with torch.no_grad():
                output_normalized, output_surface_normalized = model_to_run(
                    input_data_for_pred,
                    input_surface_for_pred,
                    aux_constants_for_pred['weather_statistics'],
                    aux_constants_for_pred['constant_maps'],
                    aux_constants_for_pred['const_h']
                )

            # Denormalize the outputs
            output_denormalized, output_surface_denormalized = utils_data.normBackData(
                output_normalized,
                output_surface_normalized,
                aux_constants_for_pred['weather_statistics_last']
            )
            
            # Store real predictions in session state
            st.session_state.output = output_denormalized
            st.session_state.output_surface = output_surface_denormalized
            
            progress_bar.text("天气预报计算完成. (75%)")
            progress_bar.progress(75)
            
            # 计算性能指标 (using real predictions now)
            st.session_state.metrics = calculate_metrics(
                st.session_state.output, 
                st.session_state.target_data, 
                st.session_state.output_surface,
                st.session_state.target_surface_data 
            )
            
            progress_bar.text("指标计算完成. (100%)")
            progress_bar.progress(100)
            st.success("预测完成！")
            
        except Exception as e:
            st.error(f"处理失败: {e}")

# 如果有数据，显示结果
if st.session_state.output is not None and st.session_state.target_data is not None: # Check target_data
    # 解析选择的变量
    var_name = selected_var.split(" ")[0]
    
    # 准备可视化
    # Ensure data is on CPU and converted to numpy for visualization via matplotlib
    # Detach tensors from graph before converting to numpy if they might have gradients
    output_to_viz = st.session_state.output.detach().cpu().numpy()
    target_to_viz = st.session_state.target_data.detach().cpu().numpy()
    output_surface_to_viz = st.session_state.output_surface.detach().cpu().numpy()
    target_surface_to_viz = st.session_state.target_surface_data.detach().cpu().numpy()

    # Tensors are expected to be [N, C, H, W] for surface, and [N, C, Pl, H, W] for upper, but
    # NetCDFDataset returns [C, Pl, H, W] for upper and [C, H, W] for surface,
    # and DataLoader adds batch dim N=1 at the beginning.
    # So, upper: [1, 5, 13, 721, 1440], surface: [1, 4, 721, 1440]

    if data_type == "高空数据":
        # 高空数据可视化
        # UI var_options: ["z", "q", "t", "u", "v", "wind_speed"]
        # NetCDFDataset nctonumpy order for upper: z, q, t, u, v (indices 0-4)
        ui_to_channel_idx_upper = {"z": 0, "q": 1, "t": 2, "u": 3, "v": 4} # "wind_speed" is derived

        if var_name == "wind_speed":
            u_data_pred = output_to_viz[0, ui_to_channel_idx_upper["u"], level_idx, :, :]
            v_data_pred = output_to_viz[0, ui_to_channel_idx_upper["v"], level_idx, :, :]
            data_to_plot_pred = np.sqrt(u_data_pred**2 + v_data_pred**2)
            
            u_data_true = target_to_viz[0, ui_to_channel_idx_upper["u"], level_idx, :, :]
            v_data_true = target_to_viz[0, ui_to_channel_idx_upper["v"], level_idx, :, :]
            data_to_plot_true = np.sqrt(u_data_true**2 + v_data_true**2)
        elif var_name in ui_to_channel_idx_upper:
            channel_idx = ui_to_channel_idx_upper[var_name]
            data_to_plot_pred = output_to_viz[0, channel_idx, level_idx, :, :]
            data_to_plot_true = target_to_viz[0, channel_idx, level_idx, :, :]
        else:
            st.error(f"未知的高空变量 '{var_name}'")
            data_to_plot_pred, data_to_plot_true = None, None

        if data_to_plot_pred is not None:
            fig_pred = visualize_map(data_to_plot_pred, title=f"预测 - {selected_var} (Level: {level} hPa)", colormap=colormap)
            result_placeholder.pyplot(fig_pred)
        if data_to_plot_true is not None:
            fig_true = visualize_map(data_to_plot_true, title=f"实际 - {selected_var} (Level: {level} hPa)", colormap=colormap)
            target_placeholder.pyplot(fig_true)
        
    else:  # 地表数据
        # UI var_options: ["t2m", "msl", "u10", "v10", "wind10"]
        # NetCDFDataset nctonumpy order for surface: msl, u10, v10, t2m (indices 0-3)
        ui_to_channel_idx_surface = {"msl": 0, "u10": 1, "v10": 2, "t2m": 3} # "wind10" is derived

        if var_name == "wind10":
            u10_pred = output_surface_to_viz[0, ui_to_channel_idx_surface["u10"], :, :] 
            v10_pred = output_surface_to_viz[0, ui_to_channel_idx_surface["v10"], :, :]
            data_to_plot_pred = np.sqrt(u10_pred**2 + v10_pred**2)
            
            u10_true = target_surface_to_viz[0, ui_to_channel_idx_surface["u10"], :, :]
            v10_true = target_surface_to_viz[0, ui_to_channel_idx_surface["v10"], :, :]
            data_to_plot_true = np.sqrt(u10_true**2 + v10_true**2)
        elif var_name in ui_to_channel_idx_surface:
            channel_idx = ui_to_channel_idx_surface[var_name]
            data_to_plot_pred = output_surface_to_viz[0, channel_idx, :, :]
            data_to_plot_true = target_surface_to_viz[0, channel_idx, :, :]
        else:
            st.error(f"未知的地表变量 '{var_name}'")
            data_to_plot_pred, data_to_plot_true = None, None

        if data_to_plot_pred is not None:
            fig_pred = visualize_map(data_to_plot_pred, title=f"预测 - {selected_var}", colormap=colormap)
            result_placeholder.pyplot(fig_pred)
        if data_to_plot_true is not None:
            fig_true = visualize_map(data_to_plot_true, title=f"实际 - {selected_var}", colormap=colormap)
            target_placeholder.pyplot(fig_true)
    
    # 显示性能指标
    with metrics_placeholder.container():
        if st.session_state.metrics:
            metrics = st.session_state.metrics
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
        d_col1, d_col2 = st.columns(2)
        if 'fig_pred' in locals() and fig_pred is not None: # Check if fig_pred was created
            with d_col1:
                st.markdown("### 下载预测图像")
                buf_pred = io.BytesIO()
                fig_pred.savefig(buf_pred, format='png', dpi=300, bbox_inches='tight')
                buf_pred.seek(0)
                st.download_button(
                    label="下载预测图像 (PNG)",
                    data=buf_pred,
                    file_name=f"pangu_prediction_{var_name}_{datetime.now().strftime('%Y%m%d%H%M')}.png",
                    mime="image/png"
                )
        
        # Prepare data for CSV download (using data_to_plot_pred if available)
        csv_download_data_np = None
        if data_type == "高空数据":
            if var_name == "wind_speed": # data_to_plot_pred is already 2D for wind_speed
                csv_download_data_np = data_to_plot_pred 
            elif var_name in ui_to_channel_idx_upper: # data_to_plot_pred is already 2D
                 csv_download_data_np = data_to_plot_pred
        else: # 地表数据
            if var_name == "wind10": # data_to_plot_pred is already 2D for wind10
                csv_download_data_np = data_to_plot_pred
            elif var_name in ui_to_channel_idx_surface: # data_to_plot_pred is already 2D
                 csv_download_data_np = data_to_plot_pred

        if csv_download_data_np is not None:
            with d_col2:
                st.markdown("### 下载数据")
                csv_data = io.StringIO()
                np.savetxt(csv_data, csv_download_data_np.flatten(), delimiter=',') # Flatten for simple CSV
                csv_data.seek(0)
                
                download_file_name = f"pangu_data_pred_{var_name}"
                if data_type == "高空数据":
                    download_file_name += f"_{level}hPa"
                download_file_name += f"_{datetime.now().strftime('%Y%m%d%H%M')}.csv"
                
                st.download_button(
                    label="下载预测数据 (CSV)",
                    data=csv_data,
                    file_name=download_file_name,
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