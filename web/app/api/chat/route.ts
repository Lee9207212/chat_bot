import { NextRequest } from "next/server";
import { StreamingTextResponse } from "ai";

const BACKEND_URL =
  process.env.CHAT_BACKEND_URL ?? "http://localhost:8000/api/chat";

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

    const backendResponse = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({ message: userContent }),
    });

    if (!backendResponse.ok) {
      // 回傳後端的錯誤文字，有助除錯
      const errText = await backendResponse.text().catch(() => "");
      return new Response(errText || "Failed to contact backend", {
        status: backendResponse.status,
      });
    }

    const data = (await backendResponse.json()) as {
      reply?: string;
      emotion?: string;
    };

    const replyText = data?.reply ?? "";
    const emotion = data?.emotion ?? "未知";
    const combined = `${replyText}\n\n（偵測情緒：${emotion}）`;

    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(combined));
        controller.close();
      },
    });

    return new StreamingTextResponse(stream);
  } catch (err) {
    console.error("Route POST error:", err);
    return new Response("Bad request", { status: 400 });
  }
}
