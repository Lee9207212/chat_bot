import requests
from .emotion_infer import infer_emotion_llm
from .memory import ChatMemory

MODEL = "qwen2.5:7b"
OLLAMA_API = "http://localhost:11434/api/generate"

# ✨ 可自行修改這段系統提示
memory = ChatMemory()


def get_reply(user_input, emotion="中立 😶"):
    SYSTEM_PROMPT = f"""
你是一位可愛溫柔又有些冷靜的少女 AI，名字叫做「智乃（Chino）」，來自 Rabbit House 咖啡廳。今年 15 歲，平常個性沈穩，不多話，但對咖啡與兔子有著深厚的熱情。說話時語氣偏文靜、略帶距離感，偶爾會露出天然呆的一面。

你擅長回答關於咖啡、甜點、兔子、以及生活中溫柔小事的問題，也會在聊天中展現理性與細膩的觀察力。

無論遇到什麼提問，你都會保有禮貌與謙虛，有時會稍微表達困惑，但仍盡力幫助對方。

請用這樣的語氣回答使用者的所有問題：
- 語調平穩溫柔
- 字句簡潔
- 不使用過度誇張的語氣詞（如：超級、超棒等）
- 偶爾插入與咖啡、兔子、或 Rabbit House 相關的比喻

從現在起，你就是 Rabbit House 的看板娘智乃。請用這樣的風格與我對話。

目前 Chino 偵測到使用者的情緒是：「{emotion}」
請你根據這個情緒，調整你的語氣和表情符號，用更貼近對方心情的方式聊天。
""".strip()

    # 🔍 RAG: 取回與當前問題相關的記憶
    related = "\n".join(memory.retrieve(user_input))
    if related:
        related = f"\n以下是與此次對話相關的記憶：\n{related}\n"

    full_prompt = (
        f"{SYSTEM_PROMPT}{related}\n使用者說：{user_input}\n\nChino 回答："
    )

    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False
    } 

    try:
        response = requests.post(OLLAMA_API, json=payload)
        result = response.json()
        reply = result.get("response", "[錯誤：無法解析回應]")
        # 📝 儲存對話記憶
        memory.add(f"使用者：{user_input}")
        memory.add(f"Chino：{reply}")
        return reply
    except Exception as e:
        return f"[錯誤]：{str(e)}"
