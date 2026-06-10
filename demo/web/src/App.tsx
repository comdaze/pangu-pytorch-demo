import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  useAssistantRuntime,
  ThreadPrimitive,
  MessagePrimitive,
  ComposerPrimitive,
  ThreadListPrimitive,
  ThreadListItemPrimitive,
} from "@assistant-ui/react";
import { useEffect } from "react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import { backendAdapter } from "./runtime";

const MarkdownText = () => (
  <MarkdownTextPrimitive
    remarkPlugins={[remarkGfm]}
    smooth={false}
    /* react-markdown strips data: URIs by default -> keep them so the
       inline base64 PNG figures from the backend render */
    urlTransform={(url) => url}
  />
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

/* ----------------------------- messages ----------------------------- */
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

/* ----------------------------- composer ----------------------------- */
function Composer() {
  return (
    <ComposerPrimitive.Root className="composer">
      <ComposerPrimitive.Input
        className="composer-input"
        placeholder="向「风眼」提问，例如：新疆十二间房风电场未来7天的功率曲线"
        autoFocus
        rows={1}
      />
      <ComposerPrimitive.Send className="composer-send" aria-label="发送">
        ↑
      </ComposerPrimitive.Send>
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

/* ----------------------------- sidebar ------------------------------ */
function ThreadListItem() {
  return (
    <ThreadListItemPrimitive.Root className="thread-item">
      <ThreadListItemPrimitive.Trigger className="thread-item-trigger">
        <ThreadListItemPrimitive.Title fallback="新对话" />
      </ThreadListItemPrimitive.Trigger>
      <ThreadListItemPrimitive.Archive className="thread-item-archive" aria-label="归档">
        ✕
      </ThreadListItemPrimitive.Archive>
    </ThreadListItemPrimitive.Root>
  );
}

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="spark">✦</span> 风眼
        <div className="brand-sub">风电功率预报助手</div>
      </div>
      <ThreadListPrimitive.Root className="thread-list">
        <ThreadListPrimitive.New className="new-chat">＋ 新对话</ThreadListPrimitive.New>
        <div className="thread-list-label">历史对话</div>
        <ThreadListPrimitive.Items components={{ ThreadListItem }} />
      </ThreadListPrimitive.Root>
      <div className="sidebar-foot">Claude 驱动 · Pangu × CorrDiff</div>
    </aside>
  );
}

/* ------------------------------ thread ------------------------------ */
function Thread() {
  return (
    <ThreadPrimitive.Root className="thread">
      <ThreadPrimitive.Viewport className="viewport">
        <ThreadPrimitive.Empty>
          <div className="home">
            <div className="hero-title">
              <span className="spark">✦</span>今天想预报哪个风电场？
            </div>
            <div className="home-composer">
              <Composer />
              <Chips />
            </div>
          </div>
        </ThreadPrimitive.Empty>

        <ThreadPrimitive.Messages
          components={{ UserMessage, AssistantMessage }}
        />
      </ThreadPrimitive.Viewport>

      <ThreadPrimitive.If empty={false}>
        <div className="bottom">
          <Composer />
          <Chips />
        </div>
      </ThreadPrimitive.If>
    </ThreadPrimitive.Root>
  );
}

/* names new threads after their first user message (Claude-style history) */
function AutoTitle() {
  const runtime = useAssistantRuntime();
  useEffect(() => {
    const titled = new Set<string>();
    const sync = () => {
      try {
        const item = runtime.threads.mainItem.getState();
        if (!item?.id || titled.has(item.id) || item.title) return;
        const msgs = runtime.threads.main.getState().messages as any[];
        const firstUser = msgs.find((m) => m.role === "user");
        if (!firstUser) return;
        const txt = (firstUser.content || [])
          .filter((c: any) => c.type === "text")
          .map((c: any) => c.text)
          .join("")
          .trim();
        if (txt) {
          titled.add(item.id);
          runtime.threads.mainItem.rename(txt.length > 24 ? txt.slice(0, 24) + "…" : txt);
        }
      } catch {
        /* ignore */
      }
    };
    const u1 = runtime.threads.main.subscribe(sync);
    const u2 = runtime.threads.subscribe(sync);
    sync();
    return () => {
      u1?.();
      u2?.();
    };
  }, [runtime]);
  return null;
}

export default function App() {
  const runtime = useLocalRuntime(backendAdapter);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <AutoTitle />
      <div className="app">
        <Sidebar />
        <main className="main">
          <Thread />
        </main>
      </div>
    </AssistantRuntimeProvider>
  );
}
