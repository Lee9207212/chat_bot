# main.py
import tkinter as tk
from GUI.interface import ChatGUI

def main():
    root = tk.Tk()
    app = ChatGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
