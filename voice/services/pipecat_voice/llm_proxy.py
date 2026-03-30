"""
llm_proxy.py — LLM 代理層

封裝 openai_agent 的 /v1/chat/completions 呼叫。
主要功能：
1. 自動加入 X-Voice-Mode: true header，啟用工具進度語音提示
2. 提供 OpenAI-compatible 介面給 Pipecat OpenAILLMService

注意：如果你只是要用 OpenAILLMService 直連 openai_agent，
不需要此模組——直接設定 base_url 即可。
此模組用於需要自訂 header / 中介邏輯的場景。
"""

import logging
from typing import Optional

import aiohttp

from config import config

logger = logging.getLogger("llm-proxy")


class LLMProxy:
    """
    LLM 代理：透過 HTTP 呼叫 openai_agent /v1/chat/completions

    使用方式：
        proxy = LLMProxy()
        async for chunk in proxy.chat_completion_stream(messages):
            print(chunk)  # SSE delta text
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice_mode: bool = True,
    ):
        self._base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self._api_key = api_key or config.LLM_API_KEY
        self._model = model or config.LLM_MODEL
        self._voice_mode = voice_mode
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._voice_mode:
            headers["X-Voice-Mode"] = "true"
        return headers

    async def chat_completion_stream(self, messages: list[dict]):
        """
        呼叫 openai_agent /v1/chat/completions（streaming mode）

        Yields SSE lines (raw text, not parsed)
        """
        session = await self._get_session()
        url = f"{self._base_url}/chat/completions"

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }

        headers = self._build_headers()
        logger.debug(f"LLM request → {url} (voice_mode={self._voice_mode})")

        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"LLM API error {resp.status}: {error_text}")
                raise RuntimeError(f"LLM API error: {resp.status}")

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if decoded:
                    yield decoded

    async def chat_completion(self, messages: list[dict]) -> dict:
        """
        呼叫 openai_agent /v1/chat/completions（非 streaming mode）

        Returns complete response dict
        """
        session = await self._get_session()
        url = f"{self._base_url}/chat/completions"

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        headers = self._build_headers()

        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"LLM API error: {resp.status} - {error_text}")
            return await resp.json()
