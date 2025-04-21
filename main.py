import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ml import random_forest_inference

matplotlib.use("TkAgg")

class ElektraApp:
  def __init__(self, master):
    self.master = master
    self.master.title("Elektra - Harmful Brain Activity Classifier")
    self.master.geometry("1600x800")  

    self.data_dir = "data"
    os.makedirs(self.data_dir, exist_ok=True)

    self.selected_model = tk.StringVar(value="Random Forest")
    self.eeg_file_path = None

    self.create_widgets()
    self.create_plots()
  
  def create_widgets(self):
    # Frame for top controls
    top_frame = tk.Frame(self.master)
    top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    # Title label
    title_label = tk.Label(
      top_frame,
      text="Elektra: Harmful Brain Activity Classifier",
      font=("Arial", 24, "bold")
    )
    title_label.pack(side=tk.LEFT, padx=10)

    # Drop-down for model selection
    model_label = tk.Label(top_frame, text="Select Model:", font=("Arial", 24))
    model_label.pack(side=tk.LEFT, padx=10)

    model_options = ["Random Forest", "Transformer", "CNN"]
    model_selector = ttk.Combobox(
      top_frame,
      textvariable=self.selected_model,
      values=model_options,
      font=("Arial", 24),
      state="readonly"
    )
    model_selector.pack(side=tk.LEFT, padx=5)

    # Buttons frame
    buttons_frame = tk.Frame(self.master)
    buttons_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    # Upload button
    upload_button = tk.Button(
      buttons_frame,
      text="Upload",
      font=("Arial", 24),
      command=self.upload_file
    )
    upload_button.pack(side=tk.LEFT, padx=5)

    # Run button
    run_button = tk.Button(
      buttons_frame,
      text="Run",
      font=("Arial", 24),
      command=self.run_inference
    )
    run_button.pack(side=tk.LEFT, padx=5)

    # Frame for results (accuracy, confidence, classification result, etc.)
    results_frame = tk.Frame(self.master)
    results_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    # Labels to show inference results
    self.inference_result_label = tk.Label(
      results_frame,
      text="Inference Result: N/A",
      font=("Arial", 24)
    )
    self.inference_result_label.pack(side=tk.TOP, anchor="w", pady=2)

    self.model_accuracy_label = tk.Label(
      results_frame,
      text="Model Accuracy: N/A",
      font=("Arial", 24)
    )
    self.model_accuracy_label.pack(side=tk.TOP, anchor="w", pady=2)

    self.model_confidence_label = tk.Label(
      results_frame,
      text="Confidence: N/A",
      font=("Arial", 24)
    )
    self.model_confidence_label.pack(side=tk.TOP, anchor="w", pady=2)

  def create_plots(self):
    # Frame for the plots
    plots_frame = tk.Frame(self.master)
    plots_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    # Raw EEG Plot
    self.fig_eeg, self.ax_eeg = plt.subplots(figsize=(5, 3))
    self.ax_eeg.set_title("Raw EEG Signal")
    self.ax_eeg.set_xlabel("Time")
    self.ax_eeg.set_ylabel("Amplitude")

    self.canvas_eeg = FigureCanvasTkAgg(self.fig_eeg, master=plots_frame)
    self.canvas_eeg.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Spectrogram Plot
    self.fig_spectrogram, self.ax_spectrogram = plt.subplots(figsize=(5, 3))
    self.ax_spectrogram.set_title("Spectrogram")
    self.ax_spectrogram.set_xlabel("Time")
    self.ax_spectrogram.set_ylabel("Frequency")

    self.canvas_spectrogram = FigureCanvasTkAgg(self.fig_spectrogram, master=plots_frame)
    self.canvas_spectrogram.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

  def upload_file(self):
    file_path = filedialog.askopenfilename(
      title="Select EEG File",
      filetypes=(("EEG Files", "*.csv *.txt *.edf *.parquet"), ("All Files", "*.*"))
    )
    if file_path:
      # Copy or just store path
      self.eeg_file_path = file_path
      messagebox.showinfo("File Uploaded", f"EEG file selected:\n{file_path}")
      self.plot_eeg_data(file_path)

  def plot_eeg_data(self, file_path):
    # Clear existing plots
    self.ax_eeg.clear()
    self.ax_spectrogram.clear()

    df = pd.read_parquet(file_path)
    time_axis = df.index
    # Plot raw EEG
    for column in df.columns:
      self.ax_eeg.plot(time_axis, df[column], label=column)
    self.ax_eeg.set_title("Raw EEG Signal")
    self.ax_eeg.set_xlabel("Time (s)")
    self.ax_eeg.set_ylabel("Amplitude")
    self.ax_eeg.legend(loc="best")

    # Plot spectrogram
    self.ax_spectrogram.set_title("Spectrogram")
    self.ax_spectrogram.set_xlabel("Time")
    self.ax_spectrogram.set_ylabel("Frequency")

    # Refresh canvases
    self.canvas_eeg.draw()
    self.canvas_spectrogram.draw()

  def run_inference(self):
    if not self.eeg_file_path:
        messagebox.showwarning("No File", "Please upload an EEG file first.")
        return

    model_type = self.selected_model.get()

    try:
        if model_type == "Random Forest":
            result, _, confidence = random_forest_inference(self.eeg_file_path)
            accuracy = 0.57  # ⬅️ Fixed accuracy for RF model
        else:
            result, accuracy, confidence = "Coming Soon", 0.0, 0.0

        self.inference_result_label.config(text=f"Inference Result: {result}")
        self.model_accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}")
        self.model_confidence_label.config(text=f"Confidence: {confidence:.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to run inference:\n{e}")

def main():
    root = tk.Tk()
    app = ElektraApp(root)

    def on_close():
        try:
            root.destroy()
        except:
            pass
        sys.exit()  # Ensures the Python process exits completely

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()

