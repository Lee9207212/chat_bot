import { NextRequest } from "next/server";
import { StreamingTextResponse } from "ai";

const BACKEND_URL =
  process.env.CHAT_BACKEND_URL ?? "http://localhost:8000/api/chat";
const encoder = new TextEncoder();

type ChatRole = "user" | "assistant" | "system";

// 若你用的是 Vercel AI SDK，content 可能是 string 或 blocks 陣列
type ChatMessage = {
  role: ChatRole;
  content:
    | string
    | Array<
        | { type: "text"; text: string }
        | { type: string; [k: string]: unknown }
      >;
};

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as { messages?: unknown };

    const messages: ChatMessage[] = Array.isArray(body?.messages)
      ? (body!.messages as ChatMessage[])
      : [];

    // 找最後一則使用者訊息
    const lastUserMessage = [...messages]
      .reverse()
      .find((m) => m && m.role === "user");

    // 將 content 正規化為純文字
    const userContent =
      typeof lastUserMessage?.content === "string"
        ? lastUserMessage.content
        : Array.isArray(lastUserMessage?.content)
        ? lastUserMessage!.content
            .map((c) => (typeof c === "object" && "text" in c ? (c as any).text ?? "" : ""))
            .join("\n")
            .trim()
        : "";

    if (!userContent.trim()) {
      return new Response("Missing user message", { status: 400 });
    }

    try {
      const backendResponse = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        body: JSON.stringify({ message: userContent }),
      });

      if (!backendResponse.ok) {
        // 回傳後端的錯誤文字，有助除錯
        const errText = await backendResponse.text().catch(() => "");
        console.error(
          "[chat route] backend error:",
          backendResponse.status,
          errText
        );
        return buildFallbackResponse(
          `後端回應異常（HTTP ${backendResponse.status}）。`,
          errText
        );
      }

      const data = (await backendResponse.json()) as {
        reply?: string;
        emotion?: string;
      };

      const replyText = data?.reply ?? "";
      const emotion = data?.emotion ?? "未知";
      const combined = `${replyText}\n\n（偵測情緒：${emotion}）`;

      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(combined));
          controller.close();
        },
      });

      return new StreamingTextResponse(stream);
    } catch (err) {
      console.error("[chat route] fetch backend failed:", err);
      return buildFallbackResponse(
        "無法連線後端，請確認 FastAPI 伺服器是否已啟動並可在本機 8000 埠存取。",
        err instanceof Error ? err.message : String(err)
      );
    }
  } catch (err) {
    console.error("Route POST error:", err);
    return new Response("Bad request", { status: 400 });
  }
}

function buildFallbackResponse(reason: string, detail?: string) {
  const text =
    `目前無法取得後端回應。\n${reason}` +
    (detail ? `\n\n除錯資訊：${detail}` : "");

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(text));
      controller.close();
    },
  });

  // 返回 200，讓前端仍可顯示 fallback 訊息，不會直接呈現錯誤提示
  return new StreamingTextResponse(stream);
}