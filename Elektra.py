import tkinter as tk

def on_click():
    label.config(text="Hello, Packaged Executable!")

root = tk.Tk()
root.title("My GUI App")
root.geometry("300x150")

label = tk.Label(root, text="Click the button")
label.pack(pady=10)

button = tk.Button(root, text="Click Me", command=on_click)
button.pack()

root.mainloop()
