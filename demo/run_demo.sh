#!/bin/bash
# Launch the 风眼 demo (FastAPI service serving the assistant-ui frontend + API,
# behind HTTP Basic Auth). Builds the frontend if needed, then runs uvicorn.
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "===== 风眼 · 风电功率预报助手 ====="

# Build the frontend if the production bundle is missing
if [ ! -f "$SCRIPT_DIR/web/dist/index.html" ]; then
  echo "构建前端 (web/dist)..."
  (cd "$SCRIPT_DIR/web" && npm install && npm run build)
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"   # matplotlib/torch libstdc++
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # GPU inference
export FENGYAN_USER="${FENGYAN_USER:-admin}"
export FENGYAN_PASS="${FENGYAN_PASS:-pangu-wind-2026}"

cd "$SCRIPT_DIR"
echo "启动 FastAPI 服务于 http://0.0.0.0:8000 （登录: $FENGYAN_USER）"
exec uvicorn api:app --host 0.0.0.0 --port 8000
