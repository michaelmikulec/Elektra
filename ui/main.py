import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# For plotting inside Tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# from ml import random_forest_inference
# from dl import transformer_inference, cnn_inference
# from utils import load_eeg_data, compute_spectrogram

class ElektraApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Elektra - Harmful Brain Activity Classifier")
        self.master.geometry("1000x600")  # Adjust size as needed

        # Path where user input data is stored
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

        # Variables
        self.selected_model = tk.StringVar(value="Random Forest")
        self.eeg_file_path = None

        # Set up UI
        self.create_widgets()
        self.create_plots()

    def create_widgets(self):
        """
        Create and place all the UI components:
        - Title label
        - Buttons (Upload, Run)
        - Model selector (Combo box)
        - Display areas (labels or placeholders for results)
        """
        # Frame for top controls
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Title label
        title_label = tk.Label(
            top_frame,
            text="Elektra: Harmful Brain Activity Classifier",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=10)

        # Drop-down for model selection
        model_label = tk.Label(top_frame, text="Select Model:", font=("Arial", 10))
        model_label.pack(side=tk.LEFT, padx=10)

        model_options = ["Random Forest", "Transformer", "CNN"]
        model_selector = ttk.Combobox(
            top_frame,
            textvariable=self.selected_model,
            values=model_options,
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
            command=self.upload_file
        )
        upload_button.pack(side=tk.LEFT, padx=5)

        # Run button
        run_button = tk.Button(
            buttons_frame,
            text="Run",
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
            font=("Arial", 12)
        )
        self.inference_result_label.pack(side=tk.TOP, anchor="w", pady=2)

        self.model_accuracy_label = tk.Label(
            results_frame,
            text="Model Accuracy: N/A",
            font=("Arial", 12)
        )
        self.model_accuracy_label.pack(side=tk.TOP, anchor="w", pady=2)

        self.model_confidence_label = tk.Label(
            results_frame,
            text="Confidence: N/A",
            font=("Arial", 12)
        )
        self.model_confidence_label.pack(side=tk.TOP, anchor="w", pady=2)

    def create_plots(self):
        """
        Create Matplotlib figures/canvases for:
        - Raw EEG Plot
        - Spectrogram
        """
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
        """
        Let user pick an EEG file (e.g. CSV, TXT, EDF, etc.).
        Copy it into the data directory (if needed), and store the path.
        """
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
        """
        Load the raw EEG data and plot it on self.ax_eeg.
        Also compute and plot the spectrogram on self.ax_spectrogram.
        """
        # Clear existing plots
        self.ax_eeg.clear()
        self.ax_spectrogram.clear()

        # -------------
        # Load EEG data
        # Replace with actual logic, e.g.:
        # data = load_eeg_data(file_path)
        # This is just a placeholder
        import numpy as np
        t = np.linspace(0, 2, 400)  # 2 seconds at 200 Hz
        data = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave as placeholder

        # Plot raw EEG
        self.ax_eeg.plot(t, data)
        self.ax_eeg.set_title("Raw EEG Signal")
        self.ax_eeg.set_xlabel("Time (s)")
        self.ax_eeg.set_ylabel("Amplitude")

        # Plot spectrogram
        self.ax_spectrogram.specgram(data, Fs=200)
        self.ax_spectrogram.set_title("Spectrogram")
        self.ax_spectrogram.set_xlabel("Time")
        self.ax_spectrogram.set_ylabel("Frequency")

        # Refresh canvases
        self.canvas_eeg.draw()
        self.canvas_spectrogram.draw()

    def run_inference(self):
        """
        Run inference based on the selected model.
        Update labels with the result, accuracy, and confidence.
        """
        if not self.eeg_file_path:
            messagebox.showwarning("No File", "Please upload an EEG file first.")
            return

        model_type = self.selected_model.get()
        # Placeholder for actual inference logic
        # Replace with e.g.:
        # if model_type == "Random Forest":
        #     result, accuracy, confidence = random_forest_inference(self.eeg_file_path)
        # elif model_type == "Transformer":
        #     result, accuracy, confidence = transformer_inference(self.eeg_file_path)
        # elif model_type == "CNN":
        #     result, accuracy, confidence = cnn_inference(self.eeg_file_path)

        # Mock results
        result = "Potentially Harmful" if model_type == "Random Forest" else "Normal"
        accuracy = 0.95
        confidence = 0.88

        # Update labels
        self.inference_result_label.config(text=f"Inference Result: {result}")
        self.model_accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}")
        self.model_confidence_label.config(text=f"Confidence: {confidence:.2f}")

def main():
    root = tk.Tk()
    app = ElektraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
