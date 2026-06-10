import type { ChatModelAdapter } from "@assistant-ui/react";

/**
 * Local-runtime adapter that streams plain-text (markdown) chunks from the
 * FastAPI backend (/api/chat) and feeds them to assistant-ui as a single
 * growing text part.
 */
export const backendAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const payload = messages.map((m) => ({
      role: m.role,
      content: m.content
        .map((c: any) => (c.type === "text" ? c.text : ""))
        .join(""),
    }));

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: payload }),
      signal: abortSignal,
    });

    if (!res.body) {
      yield { content: [{ type: "text", text: "（无响应）" }] };
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let text = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      yield { content: [{ type: "text", text }] };
    }
  },
};
