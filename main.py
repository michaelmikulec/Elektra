import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from predictor import random_forest_inference
from tf import infer_transformer
from cnn import infer_cnn
from preprocessor import df_to_spectrograms

matplotlib.use("TkAgg")
fontFamily="Arial"
fontSize=20
font=(fontFamily, fontSize)
boldFont = (fontFamily, fontSize, "bold")

class ElektraApp:
  def __init__(self, master):
    self.master = master
    self.master.title("Elektra - Harmful Brain Activity Classifier")
    self.master.geometry("1600x800")  
    self.dataDir = "data"
    os.makedirs(self.dataDir, exist_ok=True)
    self.selectedModel = tk.StringVar(value="Random Forest")
    self.eegFilePath   = None
    self.create_widgets()
    self.create_plots()

  def create_widgets(self):
    topFrame      = tk.Frame(self.master)
    titleLabel    = tk.Label( topFrame, text="Elektra: Harmful Brain Activity Classifier", font=boldFont)
    modelLabel    = tk.Label(topFrame, text="Select Model:", font=font)
    modelOptions  = ["Random Forest", "Transformer", "CNN"]
    modelSelector = ttk.Combobox(topFrame, textvariable=self.selectedModel, values=modelOptions, font=font, state="readonly")
    buttonFrame   = tk.Frame(self.master)
    uploadButton  = tk.Button( buttonFrame, text="Upload", font=font, command=self.upload_file)
    runButton     = tk.Button( buttonFrame, text="Run", font=font, command=self.run_inference)
    resultsFrame  = tk.Frame(self.master)

    topFrame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    titleLabel.pack(side=tk.LEFT, padx=10)
    modelLabel.pack(side=tk.LEFT, padx=10)
    modelSelector.pack(side=tk.LEFT, padx=5)
    buttonFrame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    uploadButton.pack(side=tk.LEFT, padx=5)
    runButton.pack(side=tk.LEFT, padx=5)
    resultsFrame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    self.inferenceResultLabel = tk.Label(resultsFrame, text="Inference Result: N/A", font=font)
    self.modelAccuracyLabel   = tk.Label(resultsFrame, text="Model Accuracy: N/A", font=font)
    self.modelConfidenceLabel = tk.Label(resultsFrame, text="Confidence: N/A", font=font)

    self.inferenceResultLabel.pack(side=tk.TOP, anchor="w", pady=2)
    self.modelAccuracyLabel.pack(side=tk.TOP, anchor="w", pady=2)
    self.modelConfidenceLabel.pack(side=tk.TOP, anchor="w", pady=2)

  def upload_file(self):
    fPath = filedialog.askopenfilename(title="Select EEG File", filetypes=(("EEG File", "*.parquet"), ("All Files", "*.*")))
    if fPath:
      self.eegFilePath = fPath
      messagebox.showinfo("File Uploaded", f"EEG file selected:\n{fPath}")
      self.plot_eeg_data(fPath)

  def run_inference(self):
    if not self.eegFilePath:
      messagebox.showwarning("No File", "Please upload an EEG file first.")
      return
    model_type = self.selectedModel.get()
    class_names = ['seizure','lpd','gpd','lrda','grda','other']
    try:
      if model_type == "Random Forest":
        result, _, confidence = random_forest_inference(self.eegFilePath)
        accuracy = 0.57
      elif model_type == "Transformer":
        tconfig = {
          "maxSeqLen":2000,
          "numChannels":19,
          "dimModel":256,
          "dimFeedForward":512,
          "dropout":0.1,
          "numHeads":8,
          "numLayers":6,
          "numClasses":6
        }
        result, accuracy, confidence = infer_transformer(
          self.eegFilePath,
          "models/t5.pth",
          class_names,
          **tconfig
        )
      elif model_type == "CNN":
        result, accuracy, confidence = infer_cnn(
          self.eegFilePath,
          "models/cnn.pth",
          class_names,
          in_channels=19,
          num_classes=6
        )
      else:
        result, accuracy, confidence = "Coming Soon", 0.0, 0.0
      self.inferenceResultLabel.config(text=f"Inference Result: {result}")
      self.modelAccuracyLabel.config(text=f"Model Accuracy: {accuracy:.2f}")
      self.modelConfidenceLabel.config(text=f"Confidence: {confidence:.2f}")
    except Exception as e:
      messagebox.showerror("Error", f"Failed to run inference:\n{e}")

  def create_plots(self):
    plots_frame = tk.Frame(self.master)
    plots_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
    self.fig_eeg, self.ax_eeg = plt.subplots(figsize=(5, 3))
    self.ax_eeg.set_title("Raw EEG Signal")
    self.ax_eeg.set_xlabel("Time")
    self.ax_eeg.set_ylabel("Amplitude")
    self.canvas_eeg = FigureCanvasTkAgg(self.fig_eeg, master=plots_frame)
    self.canvas_eeg.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    self.fig_spectrogram, self.ax_spectrogram = plt.subplots(figsize=(5, 3))
    self.ax_spectrogram.set_title("Spectrogram")
    self.ax_spectrogram.set_xlabel("Time")
    self.ax_spectrogram.set_ylabel("Frequency")
    self.canvas_spectrogram = FigureCanvasTkAgg(self.fig_spectrogram, master=plots_frame)
    self.canvas_spectrogram.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

  def plot_eeg_data(self, file_path):
    self.ax_eeg.clear()
    df          = pd.read_parquet(file_path)
    fs          = 200
    t           = df.index / fs
    dataMin     = df.values.min()
    dataMax     = df.values.max()
    dataRange   = dataMax - dataMin
    offset      = dataRange * 1.1
    yticks      = []
    ytickLabels = []
    for i, col in enumerate(df.columns):
      self.ax_eeg.plot(t, df[col] + i * offset, label=col)
      yticks.append(i * offset)
      ytickLabels.append(col)
    self.ax_eeg.set_yticks(yticks)
    self.ax_eeg.set_yticklabels(ytickLabels)
    self.ax_eeg.set_title("Raw EEG Signal")
    self.ax_eeg.set_xlabel("Time (s)")
    self.ax_eeg.set_ylabel("Channel + offset")
    self.canvas_eeg.draw()

    spec, channelNames, frequencyBins, timeBins = df_to_spectrograms(df, fs=fs, win_len=128, hop_len=64)
    fig = self.fig_spectrogram
    fig.clear()
    fig.set_constrained_layout(True)
    rows, cols = 5, 4
    axes = fig.subplots(rows, cols, sharex=True, sharey=True)
    flat = axes.ravel()
    for i, ax in enumerate(flat):
      if i < spec.shape[0]:
        pcm = ax.pcolormesh(timeBins, frequencyBins, spec[i], shading="gouraud", cmap="viridis")
        ax.set_title(channelNames[i], fontsize=fontSize)
        ax.label_outer()
      else:
        ax.axis("off")
    fig.colorbar(pcm, ax=axes.flatten().tolist(), orientation="vertical", label="Power (dB)")
    self.canvas_spectrogram.draw()


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

