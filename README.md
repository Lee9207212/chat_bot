# Chino Chat Bot

這個專案提供一個可愛的智乃（Chino）聊天機器人，後端使用 Python 與 FastAPI，前端則使用 Next.js 與 [Vercel AI](https://github.com/vercel/ai) 建立的對話視窗。

## 開發環境準備

1. 建立並啟動 Python 虛擬環境，安裝需求套件：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. 確保本地已啟動 Ollama/LiteLLM 服務與模型（預設連線到 `http://localhost:11434` 的 `qwen2.5:7b`）。

3. 啟動 FastAPI 後端：
   ```bash
   python main.py
   ```
   伺服器預設會在 `http://localhost:8000` 提供 `/api/chat` 端點。

4. 準備前端：
   ```bash
   cd web
   npm install
   npm run dev
   ```
   前端開發伺服器預設為 `http://localhost:3000`，可直接在瀏覽器打開聊天視窗。

## 專案結構

- `backend/server.py`：FastAPI 伺服器與 `/api/chat` 端點。
- `chat/`：聊天邏輯、情緒判斷與記憶模組。
- `web/`：Next.js 前端，使用 `ai/react` 的 `useChat` 建立對話介面。

## 測試對話

啟動上述服務後即可在瀏覽器中輸入訊息，前端會呼叫 Python 後端，再由後端與 Ollama 模型互動並回傳回覆與情緒標籤。