# 风眼 · assistant-ui (Claude 风格) 前端

Claude 风格的聊天界面，调用后端 FastAPI（`demo/api.py`，包装 Bedrock Claude +
Pangu/CorrDiff 预报管线）。基于 [assistant-ui](https://www.assistant-ui.com/) 的
无样式 primitives 自定义 Claude 主题，流式渲染助手回复（含 base64 气象图）。

## 运行

后端（GPU 推理，matplotlib libstdc++ 修复；Bedrock 用默认凭证链）：

```bash
cd demo
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/conda/lib \
    uvicorn api:app --host 0.0.0.0 --port 8000
```

前端（Vite dev，开发态把 /api 代理到 8000）：

```bash
cd demo/web
npm install
npm run dev      # http://localhost:5173
```

## 结构

- `src/runtime.ts` — assistant-ui `useLocalRuntime` 适配器，流式读取 `/api/chat` 的纯文本（markdown）块。
- `src/App.tsx` — Thread / Composer / 示例 pill，使用 assistant-ui primitives + `MarkdownTextPrimitive`（remark-gfm 支持表格/图片）。
- `src/styles.css` — Claude 配色（暖米色背景、珊瑚色强调、Newsreader 衬线标题）。
- 底部输入框 + 示例 pill 用正常 flex 布局（无 fixed/transform 问题）。
