import os
from typing import Final

import requests

# 使用 Ollama 來判斷情緒，預設走本機端點，必要時可透過環境變數覆蓋
EMOTION_MODEL: Final[str] = os.environ.get("OLLAMA_EMOTION_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL: Final[str] = os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
OLLAMA_GENERATE_URL: Final[str] = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

def infer_emotion_llm(user_input: str) -> str:
    """
    使用 LLM 推論情緒，只回傳單一情緒類別。

    改用直接呼叫 Ollama，避免 LiteLLM 連線錯誤影響前端體驗。
    """
    prompt = f"""
請判斷下列句子的情緒，只回傳單一情緒類別（不要多餘描述）：

可用情緒類別：
開心 😄、悲傷 😢、生氣 😠、驚訝 😲、害羞 🙈、無聊 😐、緊張 😰、中立 😶

句子：「{user_input}」
回答：
""".strip()
    payload = {"model": EMOTION_MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip()
        
        return result if result else "中立 😶"
    except Exception:
        return "中立 😶"
