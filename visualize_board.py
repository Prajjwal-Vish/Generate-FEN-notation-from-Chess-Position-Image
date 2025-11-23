import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from chessboard_snipper import process_image

def select_and_process_image():
    filepath = filedialog.askopenfilename(
        title="Select Chessboard Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not filepath:
        return

    model_inputs, board_image, bbox = process_image(filepath)

    if board_image is None:
        tk.messagebox.showerror("Error", "Failed to load or process image.")
        return

    # Display the output image
    img_rgb = cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    panel.config(image=img_tk)
    panel.image = img_tk

root = tk.Tk()
root.title("Chessboard Visualizer")

panel = tk.Label(root)
panel.pack()

btn = tk.Button(root, text="Select Chessboard Image", command=select_and_process_image)
btn.pack()

root.mainloop()
