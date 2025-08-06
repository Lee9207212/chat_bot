# gui/interface.py
import os
import tkinter as tk
from tkinter import scrolledtext
from chat.ollama_client import get_reply
from chat.emotion_infer import infer_emotion_llm  # ← 加這行

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - pillow is an optional dependency
    Image = ImageTk = None

class ChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chino - AI聊天機器人")

        # 嘗試載入背景圖片（background.png 或 background.jpg）
        self._load_background()

        # 情緒狀態標籤（新增）
        self.emotion_label = tk.Label(root, text="目前偵測情緒：😶 中立", font=("Arial", 12), fg="blue")
        self.emotion_label.pack(pady=(10, 0))

        # 對話框
        self.chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=25)
        self.chat_box.pack(padx=10, pady=10)
        self.chat_box.config(state=tk.DISABLED)

        # 輸入區
        self.entry = tk.Entry(root, width=60)
        self.entry.pack(padx=10, pady=5, side=tk.LEFT)
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(root, text="發送", command=self.send_message)
        self.send_button.pack(padx=10, pady=5, side=tk.RIGHT)

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.entry.delete(0, tk.END)

        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, f"你：{user_input}\n")
        self.chat_box.see(tk.END)

        # 🌟 推論情緒
        emotion = infer_emotion_llm(user_input)
        self.emotion_label.config(text=f"目前偵測情緒：{emotion}")

        # 🌟 傳送訊息並顯示回應
        reply = get_reply(user_input, emotion=emotion)
        self.chat_box.insert(tk.END, f"Chino：{reply.strip()}\n\n")
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def _load_background(self):
        """Load background image if available."""
        if Image is None:
            return
        for ext in ("png", "jpg", "jpeg"):
            path = os.path.join(os.getcwd(), f"background.{ext}")
            if os.path.exists(path):
                image = Image.open(path)
                self._bg_image = ImageTk.PhotoImage(image)
                label = tk.Label(self.root, image=self._bg_image)
                label.place(relwidth=1, relheight=1)
                break
