"use client";

import { useState } from "react";
import {
  ShieldAlert,
  AlertTriangle,
  BookOpen,
  ChevronDown,
  ChevronUp,
  Cross,
} from "lucide-react";

interface Passage {
  reference: string;
  text: string;
  book: string;
  score: number;
}

interface Validation {
  is_clean: boolean;
  warning: string | null;
  flagged: string[];
  verified: string[];
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "blocked" | "error";
  content: string;
  passages?: Passage[];
  validation?: Validation;
  isStreaming?: boolean;
}

interface Props {
  message: Message;
}

export default function ChatBubble({ message }: Props) {
  const [showCitations, setShowCitations] = useState(false);

  if (message.role === "user") {
    return (
      <div className="fade-in" style={{ display: "flex", justifyContent: "flex-end" }}>
        <div className="bubble-user">{message.content}</div>
      </div>
    );
  }

  if (message.role === "blocked") {
    return (
      <div className="fade-in" style={{ display: "flex", justifyContent: "flex-start" }}>
        <div className="bubble-blocked">
          <ShieldAlert size={15} style={{ marginTop: 2, flexShrink: 0 }} />
          <span>{message.content}</span>
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="fade-in" style={{ display: "flex", justifyContent: "flex-start" }}>
        <div className="bubble-error">
          <AlertTriangle size={15} style={{ marginTop: 2, flexShrink: 0 }} />
          <span>{message.content}</span>
        </div>
      </div>
    );
  }

  // Don't render until the first token arrives — TypingIndicator handles the pre-token state
  if (message.role === "assistant" && message.isStreaming && message.content.length === 0) {
    return null;
  }

  // Assistant
  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
      {/* Avatar row */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <div
          style={{
            width: 26,
            height: 26,
            borderRadius: "50%",
            background: "linear-gradient(135deg, #d97706, #92400e)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <Cross size={11} color="#fff" strokeWidth={2.5} />
        </div>
        <span
          className="font-cinzel"
          style={{ fontSize: "0.65rem", color: "var(--text-faint)", letterSpacing: "0.1em" }}
        >
          LOGOS AI
        </span>
      </div>

      <div className="bubble-assistant">
        {/* Response text */}
        <div
          style={{ whiteSpace: "pre-wrap" }}
          dangerouslySetInnerHTML={{
            __html: message.content
              .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
              .replace(/\*(.*?)\*/g, "<em>$1</em>"),
          }}
        />
        {message.isStreaming && <span className="cursor-blink" />}

        {/* Hallucination warning */}
        {message.validation && !message.validation.is_clean && message.validation.warning && (
          <div className="warning-banner">
            <AlertTriangle size={13} style={{ marginTop: 1, flexShrink: 0 }} />
            <span>{message.validation.warning}</span>
          </div>
        )}

        {/* Citations toggle */}
        {message.passages && message.passages.length > 0 && !message.isStreaming && (
          <div style={{ marginTop: 12 }}>
            <button
              className="btn-ghost"
              style={{ fontSize: "0.76rem", padding: "5px 10px" }}
              onClick={() => setShowCitations((v) => !v)}
            >
              <BookOpen size={12} />
              {showCitations ? "Hide" : "Show"} {message.passages.length} retrieved passages
              {showCitations ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>

            {showCitations && (
              <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
                {message.passages.map((p, i) => (
                  <div key={i} className="citation-card">
                    <div className="citation-ref">{p.reference}</div>
                    <div>&ldquo;{p.text}&rdquo;</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* Typing indicator */
export function TypingIndicator() {
  return (
    <div className="fade-in" style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div
        style={{
          width: 26,
          height: 26,
          borderRadius: "50%",
          background: "linear-gradient(135deg, #d97706, #92400e)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        <Cross size={11} color="#fff" strokeWidth={2.5} />
      </div>
      <div className="bubble-assistant" style={{ padding: "12px 18px" }}>
        <div className="typing-dots">
          <span /><span /><span />
        </div>
      </div>
    </div>
  );
}
