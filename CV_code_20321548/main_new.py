# Zepei Luo
# 20321548
# scyzl9@nottingham.edu.cn

# Necessary libraries
# pip install opencv-python

from utils import check_opencv_version
from gen_panorama import panorama_gen

import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

def update_progress(progress):
    progress_var.set(progress * 100)
    progress_bar.update()

def select_files():
    file_types = [('Video files', '*.mp4')]
    file_paths = filedialog.askopenfilenames(title='Select Video Path', filetypes=file_types)
    if file_paths:
        entry.delete(0, tk.END)
        entry.insert(0, '; '.join(file_paths))
    return file_paths

def stitch_videos():
    file_paths = entry.get()
    if not file_paths:
        messagebox.showinfo("Prompt", "Please select the video file first!")
        return
    try:
        #print(file_paths)
        result_image_path = panorama_gen(video_path = file_paths, progress_callback=update_progress)
        display_image(result_image_path)
        messagebox.showinfo("Finished", "Panorama generation complete!")
    except Exception as e:
        messagebox.showerror("ERROR", str(e))

def display_image(image_path):
    # Load and display the image
    os.startfile(image_path)
        
        
# Create main window
root = tk.Tk()
root.title("Panorama generation tool")

# Set file selection input box
entry = tk.Entry(root, width=80)
entry.pack(pady=20)

# Set the select file button
button_select = tk.Button(root, text="Select video file", command=select_files)
button_select.pack(pady=10)

# Set the Start generating button
button_stitch = tk.Button(root, text="Begin generation", command=stitch_videos)
button_stitch.pack(pady=10)

# Setup progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate", variable=progress_var)
progress_bar.pack(pady=10)

# Start event loop
root.mainloop()
