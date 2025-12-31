"use client";

import { useMemo } from "react";
import { useChat } from "ai/react";

import styles from "./page.module.css";

export default function HomePage() {
  const { messages, input, handleInputChange, handleSubmit, isLoading, error } = useChat(
    {
      api: "/api/chat",
      streamProtocol: "text",
    }
  );

  const renderedMessages = useMemo(
    () =>
      messages.map((message) => (
        <div key={message.id} className={styles.message} data-role={message.role}>
          <div className={styles.avatar}>{message.role === "user" ? "你" : "智乃"}</div>
          <div className={styles.bubble}>{message.content}</div>
        </div>
      )),
    [messages]
  );

  const fallbackMessage = error
    ? "抱歉，目前無法取得回應，請確認後端伺服器（FastAPI/Ollama）是否已啟動。"
    : null;

  return (
    <main className={styles.container}>
      <section className={styles.chatWindow}>
        <header className={styles.header}>
          <h1>智乃（Chino）</h1>
          <p>透過 Vercel AI 建立的網頁聊天視窗</p>
        </header>

        <div className={styles.messages}>
          {renderedMessages}
          {fallbackMessage ? (
            <div className={styles.message} data-role="assistant">
              <div className={styles.avatar}>智乃</div>
              <div className={styles.bubble}>{fallbackMessage}</div>
            </div>
          ) : null}
        </div>

        <form className={styles.form} onSubmit={handleSubmit}>
          <input
            className={styles.input}
            value={input}
            onChange={handleInputChange}
            placeholder="輸入訊息..."
            disabled={isLoading}
            aria-label="輸入要傳送給智乃的訊息"
          />
          <button className={styles.button} type="submit" disabled={isLoading || input.trim().length === 0}>
            {isLoading ? "傳送中..." : "發送"}
          </button>
        </form>

        {error ? <p className={styles.error}>無法取得回應，請稍後再試。</p> : null}
      </section>
    </main>
  );
}