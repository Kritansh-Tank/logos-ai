"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Send,
  Menu,
  X,
  Trash2,
  MessageSquare,
  Image as ImageIcon,
  Cross,
  BookOpen,
  WifiOff,
} from "lucide-react";
import ChatBubble, { Message, TypingIndicator } from "./components/ChatBubble";
import DenominationSelector from "./components/DenominationSelector";
import ImageCard, { ImageMessage } from "./components/ImagePanel";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Denomination = "protestant" | "catholic" | "orthodox" | "nondenominational";
type Tab = "chat" | "image";

const EXAMPLE_PROMPTS = [
  "What does John 3:16 mean?",
  "Explain the Beatitudes from Matthew 5",
  "What does Psalm 23 say about the Lord as shepherd?",
  "Is Purgatory mentioned in the Bible?",
  "What does the Bible say about forgiveness?",
  "Who was Mary Magdalene?",
];

function uid() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [imageInput, setImageInput] = useState("");
  const [imageMessages, setImageMessages] = useState<ImageMessage[]>([]);
  const [sessionId] = useState(uid);
  const [denomination, setDenomination] = useState<Denomination>("nondenominational");
  const [isLoading, setIsLoading] = useState(false);
  const [isImageLoading, setIsImageLoading] = useState(false);
  const [tab, setTab] = useState<Tab>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Health check
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => setBackendOk(r.ok))
      .catch(() => setBackendOk(false));
  }, []);

  // Scroll to bottom on new messages or new images
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, imageMessages]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
  }, [input]);

  const sendMessage = useCallback(
    async (text?: string) => {
      const userText = (text ?? input).trim();
      if (!userText || isLoading) return;

      setInput("");
      setIsLoading(true);

      const userMsg: Message = { id: uid(), role: "user", content: userText };
      const assistantId = uid();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      try {
        const res = await fetch(`${API_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userText, denomination, session_id: sessionId }),
        });

        // Check if blocked (returns JSON instead of SSE)
        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
          const data = await res.json();
          if (data.blocked) {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, role: "blocked" as const, content: data.reason, isStreaming: false }
                  : m
              )
            );
            setIsLoading(false);
            return;
          }
        }

        // SSE stream
        const reader = res.body?.getReader();
        const decoder = new TextDecoder();
        if (!reader) throw new Error("No stream");

        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (!raw) continue;

            try {
              const event = JSON.parse(raw);
              if (event.type === "token") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId ? { ...m, content: m.content + event.content } : m
                  )
                );
              } else if (event.type === "done") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, isStreaming: false, passages: event.passages, validation: event.validation }
                      : m
                  )
                );
              } else if (event.type === "error") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, role: "error" as const, content: event.content, isStreaming: false }
                      : m
                  )
                );
              }
            } catch {
              /* malformed JSON — skip */
            }
          }
        }
      } catch {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, role: "error" as const, content: "Connection error. Is the backend running?", isStreaming: false }
              : m
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [input, isLoading, denomination, sessionId]
  );

  const generateImage = useCallback(async (text?: string) => {
    const promptText = (text ?? imageInput).trim();
    if (!promptText || isImageLoading) return;
    setImageInput("");
    setIsImageLoading(true);

    const msgId = uid();
    setImageMessages(prev => [...prev, { id: msgId, prompt: promptText, result: null, loading: true }]);

    try {
      const res = await fetch(`${API_URL}/image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: promptText }),
      });
      const data = await res.json();
      setImageMessages(prev => prev.map(m => m.id === msgId ? { ...m, result: data, loading: false } : m));
    } catch {
      setImageMessages(prev => prev.map(m =>
        m.id === msgId ? { ...m, result: { success: false, reason: "Network error. Please try again." }, loading: false } : m
      ));
    } finally {
      setIsImageLoading(false);
    }
  }, [imageInput, isImageLoading]);

  const clearChat = useCallback(async () => {
    setMessages([]);
    fetch(`${API_URL}/clear-session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => { });
  }, [sessionId]);

  const hasMessages = messages.length > 0;
  const hasImageMessages = imageMessages.length > 0;

  return (
    <div style={{ display: "flex", height: "100dvh", overflow: "hidden", background: "var(--bg-base)" }}>

      {/* ── Sidebar ────────────────────────────────────────────────── */}
      <aside
        style={{
          width: sidebarOpen ? 272 : 0,
          minWidth: sidebarOpen ? 272 : 0,
          overflow: "hidden",
          transition: "width 0.28s ease, min-width 0.28s ease",
          background: "var(--bg-surface)",
          borderRight: "1px solid var(--border-light)",
          display: "flex",
          flexDirection: "column",
          boxShadow: "2px 0 12px rgba(0,0,0,0.04)",
        }}
      >
        {sidebarOpen && (
          <div style={{ display: "flex", flexDirection: "column", height: "100%", padding: "20px 18px", overflow: "hidden" }}>

            {/* Logo */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div
                    style={{
                      width: 34,
                      height: 34,
                      borderRadius: 10,
                      background: "linear-gradient(135deg, #d97706, #92400e)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      boxShadow: "0 2px 8px rgba(217,119,6,0.3)",
                    }}
                  >
                    <Cross size={14} color="#fff" strokeWidth={2.5} />
                  </div>
                  <div>
                    <h1 className="font-cinzel" style={{ fontSize: "1rem", color: "var(--amber-800)", lineHeight: 1.2 }}>
                      Logos AI
                    </h1>
                    <p style={{ fontSize: "0.6rem", color: "var(--text-faint)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                      Scripture Assistant
                    </p>
                  </div>
                </div>
                <button className="btn-icon" onClick={() => setSidebarOpen(false)}>
                  <X size={14} />
                </button>
              </div>
            </div>

            {/* Tab switcher */}
            <div className="sidebar-section">
              <div
                style={{
                  display: "flex",
                  background: "var(--bg-elevated)",
                  borderRadius: "var(--radius-sm)",
                  padding: 3,
                  gap: 2,
                }}
              >
                {([
                  { key: "chat" as Tab, label: "Chat", Icon: MessageSquare },
                  { key: "image" as Tab, label: "Image", Icon: ImageIcon },
                ]).map(({ key, label, Icon }) => (
                  <button
                    key={key}
                    onClick={() => setTab(key)}
                    style={{
                      flex: 1,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 6,
                      padding: "7px 10px",
                      border: "none",
                      borderRadius: 6,
                      fontSize: "0.78rem",
                      fontWeight: 600,
                      fontFamily: "Inter, sans-serif",
                      cursor: "pointer",
                      transition: "var(--transition)",
                      background: tab === key ? "var(--bg-surface)" : "transparent",
                      color: tab === key ? "var(--amber-700)" : "var(--text-muted)",
                      boxShadow: tab === key ? "var(--shadow-sm)" : "none",
                    }}
                  >
                    <Icon size={13} />
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Denomination (chat tab only) */}
            {tab === "chat" && (
              <div className="sidebar-section">
                <DenominationSelector value={denomination} onChange={setDenomination} />
              </div>
            )}

            {/* Spacer */}
            <div style={{ flex: 1 }} />

            {/* Example prompts */}
            {!hasMessages && tab === "chat" && (
              <div className="sidebar-section" style={{ borderBottom: "none" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
                  <BookOpen size={12} color="var(--text-faint)" />
                  <p className="sidebar-label" style={{ marginBottom: 0 }}>Try asking</p>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {EXAMPLE_PROMPTS.map((p, i) => (
                    <button
                      key={i}
                      className="prompt-chip"
                      onClick={() => sendMessage(p)}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Clear conversation */}
            {hasMessages && tab === "chat" && (
              <div style={{ paddingTop: 12 }}>
                <button className="btn-ghost" style={{ width: "100%", justifyContent: "center" }} onClick={clearChat}>
                  <Trash2 size={13} />
                  New Conversation
                </button>
              </div>
            )}
          </div>
        )}
      </aside>

      {/* ── Main ───────────────────────────────────────────────────── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>

        {/* Header */}
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "12px 20px",
            background: "transparent",
            flexShrink: 0,
            gap: 12,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            {!sidebarOpen && (
              <button className="btn-icon" onClick={() => setSidebarOpen(true)} title="Open sidebar">
                <Menu size={15} />
              </button>
            )}
            {!sidebarOpen && (
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div
                  style={{
                    width: 26,
                    height: 26,
                    borderRadius: 8,
                    background: "linear-gradient(135deg, #d97706, #92400e)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Cross size={11} color="#fff" strokeWidth={2.5} />
                </div>
                <h1 className="font-cinzel" style={{ fontSize: "0.95rem", color: "var(--amber-800)" }}>
                  Logos AI
                </h1>
              </div>
            )}
          </div>

          {/* Status badge */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "4px 10px",
              background: "var(--bg-elevated)",
              border: "1px solid var(--border-light)",
              borderRadius: 999,
              fontSize: "0.7rem",
              color: "var(--text-muted)",
            }}
          >
            {backendOk === false ? (
              <WifiOff size={11} color="#ef4444" />
            ) : (
              <div className="status-dot" />
            )}
            <span>
              {backendOk === false
                ? "Backend offline"
                : "31,102 KJV verses · Grounded"}
            </span>
          </div>
        </header>

        <div className="messages-area">

          {/* ── Chat tab content ── */}
          {tab === "chat" && (
            <>
              {/* Welcome screen */}
              {!hasMessages && (
                <div
                  className="fade-in"
                  style={{
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    textAlign: "center",
                    padding: "60px 24px",
                    gap: 20,
                  }}
                >
                  <div
                    style={{
                      width: 68,
                      height: 68,
                      borderRadius: 20,
                      background: "linear-gradient(135deg, #d97706, #92400e)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      boxShadow: "0 8px 32px rgba(217,119,6,0.25)",
                    }}
                  >
                    <Cross size={28} color="#fff" strokeWidth={2} />
                  </div>

                  <div>
                    <h2 className="font-cinzel" style={{ fontSize: "1.8rem", color: "var(--amber-800)", marginBottom: 10 }}>
                      Logos AI
                    </h2>
                  </div>

                  <p style={{ color: "var(--text-muted)", fontSize: "0.88rem", maxWidth: 380, lineHeight: 1.7 }}>
                    Ask anything about scripture, theology, or Christian life.<br />
                    Every answer is grounded in verified KJV Bible verses.
                  </p>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(2, 1fr)",
                      gap: 8,
                      maxWidth: 500,
                      width: "100%",
                      marginTop: 8,
                    }}
                  >
                    {EXAMPLE_PROMPTS.slice(0, 4).map((p, i) => (
                      <button key={i} className="prompt-chip" onClick={() => sendMessage(p)}>
                        <BookOpen size={12} color="var(--amber-600)" style={{ flexShrink: 0 }} />
                        {p}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg) => (
                <ChatBubble key={msg.id} message={msg} />
              ))}

              {isLoading && !messages.find((m) => m.isStreaming && m.content.length > 0) && (
                <TypingIndicator />
              )}
            </>
          )}

          {/* ── Image tab content ── */}
          {tab === "image" && (
            <>
              {/* Image welcome */}
              {!hasImageMessages && (
                <div
                  className="fade-in"
                  style={{
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    textAlign: "center",
                    padding: "60px 24px",
                    gap: 20,
                  }}
                >
                  <div
                    style={{
                      width: 68,
                      height: 68,
                      borderRadius: 20,
                      background: "linear-gradient(135deg, #d97706, #92400e)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      boxShadow: "0 8px 32px rgba(217,119,6,0.25)",
                    }}
                  >
                    <Cross size={28} color="#fff" strokeWidth={2} />
                  </div>

                  <div>
                    <h2 className="font-cinzel" style={{ fontSize: "1.8rem", color: "var(--amber-800)", marginBottom: 10 }}>
                      Logos AI
                    </h2>
                  </div>

                  <p style={{ color: "var(--text-muted)", fontSize: "0.88rem", maxWidth: 500, lineHeight: 1.7 }}>
                    Describe a biblical scene or moment and generate Christian artwork.<br />
                    All prompts are moderated for safety.
                  </p>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(2, 1fr)",
                      gap: 8,
                      maxWidth: 500,
                      width: "100%",
                    }}
                  >
                    {[
                      "Jesus walking on water at sunrise",
                      "The Good Shepherd with his flock",
                      "Noah's Ark during the great flood",
                      "The Last Supper, Renaissance style",
                    ].map((p, i) => (
                      <button key={i} className="prompt-chip" onClick={() => generateImage(p)}>
                        <ImageIcon size={12} color="var(--amber-600)" style={{ flexShrink: 0 }} />
                        {p}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {imageMessages.map(msg => (
                <ImageCard key={msg.id} msg={msg} />
              ))}
            </>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Chat input bar */}
        {tab === "chat" && (
          <div
            style={{
              padding: "14px 20px",
              background: "transparent",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                display: "flex",
                gap: 10,
                alignItems: "flex-end",
                maxWidth: 860,
                margin: "0 auto",
              }}
            >
              <textarea
                ref={textareaRef}
                className="input-field"
                style={{ flex: 1, minHeight: 44, maxHeight: 160 }}
                placeholder="Ask about scripture, theology, or Christian life..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                disabled={isLoading}
                rows={1}
              />
              <button
                className="btn-send"
                onClick={() => sendMessage()}
                disabled={isLoading || !input.trim()}
                title="Send (Enter)"
              >
                {isLoading ? (
                  <div className="typing-dots" style={{ transform: "scale(0.85)" }}>
                    <span /><span /><span />
                  </div>
                ) : (
                  <Send size={16} />
                )}
              </button>
            </div>
            <p
              style={{
                textAlign: "center",
                fontSize: "0.67rem",
                color: "var(--text-faint)",
                marginTop: 8,
              }}
            >
              Shift+Enter for new line
            </p>
          </div>
        )}

        {/* Image input bar */}
        {tab === "image" && (
          <div
            style={{
              padding: "14px 20px",
              background: "transparent",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                display: "flex",
                gap: 10,
                alignItems: "flex-end",
                maxWidth: 860,
                margin: "0 auto",
              }}
            >
              <textarea
                className="input-field"
                style={{ flex: 1, minHeight: 44, maxHeight: 120 }}
                placeholder='Describe a biblical scene... e.g. "Jesus walking on water at sunrise"'
                value={imageInput}
                onChange={(e) => setImageInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    generateImage();
                  }
                }}
                disabled={isImageLoading}
                rows={1}
              />
              <button
                className="btn-send"
                onClick={() => generateImage()}
                disabled={isImageLoading || !imageInput.trim()}
                title="Generate (Enter)"
                style={{ width: "auto", padding: "0 16px", gap: 6 }}
              >
                {isImageLoading ? (
                  <div className="typing-dots" style={{ transform: "scale(0.85)" }}>
                    <span /><span /><span />
                  </div>
                ) : (
                  <ImageIcon size={16} />
                )}
              </button>
            </div>
            <p
              style={{
                textAlign: "center",
                fontSize: "0.67rem",
                color: "var(--text-faint)",
                marginTop: 8,
              }}
            >
              Shift+Enter for new line
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
