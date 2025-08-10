import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import subprocess
import os

class MusicGeneratorGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI Music Generator")
        master.geometry("400x300")

        # Sequence Length
        self.length_label = tk.Label(master, text="Length of sequence:")
        self.length_label.pack()
        self.length_entry = tk.Entry(master)
        self.length_entry.insert(0, "100")
        self.length_entry.pack()

        # BPM
        self.bpm_label = tk.Label(master, text="Beats Per Minute (BPM):")
        self.bpm_label.pack()
        self.bpm_entry = tk.Entry(master)
        self.bpm_entry.insert(0, "120")
        self.bpm_entry.pack()

        # Output Folder
        self.output_folder = tk.StringVar()
        self.output_button = tk.Button(master, text="Choose Output Folder", command=self.choose_folder)
        self.output_button.pack()
        self.output_label = tk.Label(master, textvariable=self.output_folder)
        self.output_label.pack()

        # Generate Button
        self.generate_button = tk.Button(master, text="Generate Music", command=self.generate_music)
        self.generate_button.pack(pady=20)

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)

    def generate_music(self):
        length = self.length_entry.get()
        bpm = self.bpm_entry.get()
        output_dir = self.output_folder.get()

        if not output_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        out_wav = os.path.join(output_dir, "output.wav")
        out_midi = os.path.join(output_dir, "output.mid")

        command = [
            "python", "generate.py",
            "--mode", "model",
            "--model", "model/music_model.h5",
            "--length", length,
            "--bpm", bpm,
            "--out_wav", out_wav,
            "--out_midi", out_midi
        ]

        try:
            subprocess.run(command, check=True)
            messagebox.showinfo("Success", "Music generated and saved!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Generation failed:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGeneratorGUI(root)
    root.mainloop()
