"use client";

import { useState } from "react";
import { ShieldAlert, ImageOff, Cross } from "lucide-react";

export interface ImageMessage {
  id: string;
  prompt: string;
  result: {
    success: boolean;
    url?: string;
    blocked?: boolean;
    reason?: string;
    enhanced_prompt?: string;
  } | null;
  loading: boolean;
}

/* Shared avatar row — identical to ChatBubble */
function LogosAvatar() {
  return (
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
  );
}

export default function ImageCard({ msg }: { msg: ImageMessage }) {
  const { prompt, result, loading } = msg;
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgError, setImgError] = useState(false);

  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: 10 }}>

      {/* User prompt bubble — right-aligned, same as chat */}
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <div className="bubble-user">{prompt}</div>
      </div>

      {/* Generating state — matches TypingIndicator layout */}
      {loading && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
          <LogosAvatar />
          <div className="bubble-assistant" style={{ padding: "12px 18px" }}>
            <div className="typing-dots">
              <span /><span /><span />
            </div>
          </div>
        </div>
      )}

      {/* Result */}
      {!loading && result && (
        result.blocked || !result.success ? (
          /* Blocked — matches chat bubble-blocked */
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
            <LogosAvatar />
            <div className="bubble-blocked">
              <ShieldAlert size={14} style={{ marginTop: 1, flexShrink: 0 }} />
              <span>{result.reason || "This prompt was blocked for safety."}</span>
            </div>
          </div>
        ) : result.url ? (
          /* Success — avatar + bubble-assistant box containing image + caption */
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
            <LogosAvatar />
            <div className="bubble-assistant" style={{ padding: 0, overflow: "hidden", maxWidth: 420 }}>
              {/* Skeleton while image loads */}
              {!imgLoaded && !imgError && (
                <div
                  className="skeleton"
                  style={{ height: 220, display: "flex", alignItems: "center", justifyContent: "center" }}
                >
                  <ImageOff size={24} color="var(--text-faint)" />
                </div>
              )}

              {/* Error fallback */}
              {imgError && (
                <div
                  style={{
                    height: 120,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 8,
                    background: "var(--bg-elevated)",
                    color: "var(--text-faint)",
                    fontSize: "0.8rem",
                    padding: "0 16px",
                    textAlign: "center",
                  }}
                >
                  <ImageOff size={20} />
                  Image failed to load — pollinations.ai may be slow
                </div>
              )}

              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={result.url}
                alt={`Generated: ${prompt}`}
                onLoad={() => setImgLoaded(true)}
                onError={() => setImgError(true)}
                style={{ display: imgLoaded ? "block" : "none", width: "100%", height: "auto" }}
              />

              {/* Caption inside the bubble box */}
              <div style={{ padding: "10px 14px", borderTop: imgLoaded ? "1px solid var(--border-light)" : "none" }}>
                {result.enhanced_prompt && (
                  <p style={{ fontSize: "0.72rem", color: "var(--text-faint)", fontStyle: "italic", marginBottom: 4, lineHeight: 1.5 }}>
                    {result.enhanced_prompt}
                  </p>
                )}
                <p style={{ fontSize: "0.67rem", color: "var(--text-faint)" }}>
                  AI-generated biblical artwork · pollinations.ai
                </p>
              </div>
            </div>
          </div>
        ) : null
      )}
    </div>
  );
}
