#!/bin/bash

# 确保脚本在任何错误时退出
set -e

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "===== PanGu Weather Forecast Demo ====="
echo "正在启动Streamlit应用..."

# 检查是否已安装所需依赖
if ! command -v streamlit &> /dev/null; then
    echo "未检测到Streamlit，正在安装依赖..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# 添加项目根目录到PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 启动Streamlit应用
cd "$SCRIPT_DIR"
streamlit run app.py