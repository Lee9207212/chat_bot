# emotion_infer.py
import requests

# 使用你已安裝的模型，建議中文模型如 qwen2.5:7b
MODEL = "qwen2.5:7b"

def infer_emotion_llm(user_input: str) -> str:
    """
    使用 LLM 推論情緒，只回傳單一情緒類別
    """
    prompt = f"""
請判斷下列句子的情緒，只回傳單一情緒類別（不要多餘描述）：

可用情緒類別：
開心 😄、悲傷 😢、生氣 😠、驚訝 😲、害羞 🙈、無聊 😐、緊張 😰、中立 😶

句子：「{user_input}」
回答：
""".strip()

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        result = response.json().get("response", "").strip()
        return result if result else "中立 😶"
    else:
        return "中立 😶"
