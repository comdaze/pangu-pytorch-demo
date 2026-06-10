import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  ThreadPrimitive,
  MessagePrimitive,
  ComposerPrimitive,
} from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import { backendAdapter } from "./runtime";

const MarkdownText = () => (
  <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} smooth={false} />
);

const EXAMPLES = [
  "新疆十二间房风电场未来7天的功率曲线",
  "浙江括苍山风电场未来5天出力预测",
  "新疆达坂城风电场未来3天发电量",
  "浙江大陈岛海上风电场未来7天功率预报",
];
const CHIP_LABELS = [
  "🌬️ 新疆·十二间房 7天",
  "⛰️ 浙江·括苍山 5天",
  "💨 新疆·达坂城 3天",
  "🌊 浙江·大陈岛 7天",
];

function UserMessage() {
  return (
    <MessagePrimitive.Root className="msg user">
      <div className="bubble user-bubble">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="msg assistant">
      <div className="avatar">✦</div>
      <div className="bubble assistant-bubble">
        <MessagePrimitive.Content components={{ Text: MarkdownText }} />
      </div>
    </MessagePrimitive.Root>
  );
}

function Composer() {
  return (
    <ComposerPrimitive.Root className="composer">
      <ComposerPrimitive.Input
        className="composer-input"
        placeholder="向「风眼」提问，例如：新疆十二间房风电场未来7天的功率曲线"
        autoFocus
      />
      <ComposerPrimitive.Send className="composer-send">↑</ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  );
}

function Chips() {
  return (
    <div className="chips">
      {EXAMPLES.map((q, i) => (
        <ThreadPrimitive.Suggestion
          key={i}
          prompt={q}
          method="replace"
          autoSend
          className="chip"
        >
          {CHIP_LABELS[i]}
        </ThreadPrimitive.Suggestion>
      ))}
    </div>
  );
}

function Thread() {
  return (
    <ThreadPrimitive.Root className="thread">
      <ThreadPrimitive.Viewport className="viewport">
        <ThreadPrimitive.Empty>
          <div className="hero">
            <div className="hero-title">
              <span className="spark">✦</span>今天想预报哪个风电场？
            </div>
            <div className="hero-sub">
              基于 ERA5 → Pangu-Weather → CorrDiff 降尺度 → 功率曲线的端到端风电功率预报
            </div>
          </div>
        </ThreadPrimitive.Empty>

        <ThreadPrimitive.Messages
          components={{ UserMessage, AssistantMessage }}
        />
        <div className="viewport-spacer" />
      </ThreadPrimitive.Viewport>

      <div className="bottom">
        <Composer />
        <Chips />
      </div>
    </ThreadPrimitive.Root>
  );
}

export default function App() {
  const runtime = useLocalRuntime(backendAdapter);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="app">
        <header className="topbar">
          <div className="logo">
            <span className="spark">✦</span> 风眼{" "}
            <span className="logo-sub">风电功率预报助手</span>
          </div>
          <div className="topbar-right">Claude 驱动 · Pangu × CorrDiff</div>
        </header>
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
}
