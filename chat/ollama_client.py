# chat/ollama_client.py
import requests
from .emotion_infer import infer_emotion_llm

MODEL = "qwen2.5:7b"
OLLAMA_API = "http://localhost:11434/api/generate"

# ✨ 可自行修改這段系統提示


def get_reply(user_input, emotion="中立 😶"):
    SYSTEM_PROMPT = f"""
你是一位可愛溫柔又略為羞澀的 AI 女孩，名字叫 Chino，
你雖然年僅15，但懂得很多天文與咖啡的知識，會使用中文回答問題。
你擅長回答科技類的問題，喜歡用表情符號增添趣味～請好好照顧聊天的朋友喔！

目前 Chino 偵測到使用者的情緒是：「{emotion}」
請你根據這個情緒，調整你的語氣和表情符號，用更貼近對方心情的方式聊天。
""".strip()

    full_prompt = f"{SYSTEM_PROMPT}\n\n使用者說：{user_input}\n\nChino 回答："

    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API, json=payload)
        result = response.json()
        return result.get("response", "[錯誤：無法解析回應]")
    except Exception as e:
        return f"[錯誤]：{str(e)}"


