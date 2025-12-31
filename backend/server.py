"""FastAPI server exposing chat endpoints for the Chino assistant."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat.ollama_client import get_reply


class ChatRequest(BaseModel):
    """Schema describing a chat request payload."""

    message: str


class ChatResponse(BaseModel):
    """Schema describing the chat response returned to the client."""

    reply: str


app = FastAPI(title="Chino Chat API", version="1.0.0")

# Allow the Next.js dev server (default on port 3000) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Return the model reply."""

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    reply = get_reply(message)
    return ChatResponse(reply=reply)
