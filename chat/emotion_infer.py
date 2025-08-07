from litellm import completion

# 使用 LiteLLM 的模型格式，建議中文模型如 qwen2.5:7b
MODEL = "ollama/qwen2.5:7b"
BASE_URL = "http://localhost:11434"

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

    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            base_url=BASE_URL,
        )
        result = response["choices"][0]["message"]["content"].strip()
        return result if result else "中立 😶"
    except Exception:
        return "中立 😶"
