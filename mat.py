# deep_signal_analyzer.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.stats import skew, kurtosis

class ProfessionalMATAnalyzer:
    def __init__(self, master):
        self.master = master
        master.title("Professional MAT Signal Analyzer")
        master.geometry("1200x700")
        
        # --- Frames ---
        self.left_frame = tk.Frame(master, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Load Button ---
        self.load_btn = tk.Button(self.left_frame, text="Load .mat File", command=self.load_mat)
        self.load_btn.pack(pady=5, fill=tk.X)

        # --- Treeview for Keys ---
        self.tree = ttk.Treeview(self.left_frame, columns=("shape","dtype"), show='headings')
        self.tree.heading("shape", text="Shape")
        self.tree.heading("dtype", text="Dtype")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_key_select)

        # --- Signal Selection ---
        self.signal_label = tk.Label(self.left_frame, text="Select Signal (index):")
        self.signal_label.pack(pady=(10,0))
        self.signal_slider = tk.Scale(self.left_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_signal_select)
        self.signal_slider.pack(fill=tk.X)

        # --- Deep Analysis Button ---
        self.analyze_btn = tk.Button(self.left_frame, text="Deep Analysis", command=self.deep_analysis)
        self.analyze_btn.pack(pady=5, fill=tk.X)

        # --- Analysis Area ---
        self.fig = Figure(figsize=(7,5))
        self.ax_signal = self.fig.add_subplot(211)
        self.ax_fft = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Feature Display ---
        self.feature_text = tk.Text(self.left_frame, height=12, wrap=tk.WORD)
        self.feature_text.pack(fill=tk.BOTH, expand=False, pady=(10,0))

        # --- Internal Variables ---
        self.mat_data = None
        self.keys = []
        self.current_key = None
        self.signals = None
        self.current_signal_idx = 0

        # --- Log ---
        self.log_text = tk.Text(self.left_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=False, pady=(10,0))

    # --- Logging Helper ---
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        print(msg)

    # --- Load MAT ---
    def load_mat(self):
        path = filedialog.askopenfilename(title="Select .mat file", filetypes=[("MAT files","*.mat"),("All files","*.*")])
        if not path: return
        try:
            self.mat_data = loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load .mat: {e}")
            return

        self.tree.delete(*self.tree.get_children())
        self.keys = [k for k in self.mat_data.keys() if not k.startswith("__")]
        for k in self.keys:
            try:
                arr = np.array(self.mat_data[k])
                shape = arr.shape
                dtype = arr.dtype
            except Exception:
                shape = "N/A"
                dtype = type(self.mat_data[k]).__name__
            self.tree.insert("", "end", iid=k, values=(shape,dtype))
        self.log(f"Loaded .mat with keys: {', '.join(self.keys)}")

    # --- Key Selected ---
    def on_key_select(self, event):
        if not self.mat_data: return
        selected = self.tree.selection()
        if not selected: return
        key = selected[0]
        self.current_key = key
        arr = np.array(self.mat_data[key])
        if arr.ndim == 1: arr = arr.reshape(1,-1)
        self.signals = arr
        self.signal_slider.config(from_=0, to=arr.shape[0]-1)
        self.current_signal_idx = 0
        self.update_display()

    # --- Signal Slider ---
    def on_signal_select(self, val):
        idx = int(val)
        self.current_signal_idx = idx
        self.update_display()

    # --- Update Display ---
    def update_display(self):
        if self.signals is None: return
        sig = self.signals[self.current_signal_idx,:]
        self.ax_signal.clear()
        self.ax_signal.plot(sig, color='blue')
        self.ax_signal.set_title(f"{self.current_key} - Signal {self.current_signal_idx}")
        self.ax_signal.set_xlabel("Sample")
        self.ax_signal.set_ylabel("Amplitude")
        self.ax_signal.grid(True)

        # FFT
        self.ax_fft.clear()
        N = len(sig)
        fft_vals = np.fft.fft(sig)
        freqs = np.fft.fftfreq(N, d=1)
        self.ax_fft.plot(freqs[:N//2], np.abs(fft_vals)[:N//2], color='red')
        self.ax_fft.set_title("FFT Magnitude")
        self.ax_fft.set_xlabel("Normalized Frequency")
        self.ax_fft.set_ylabel("Magnitude")
        self.ax_fft.grid(True)

        self.canvas.draw()
        self.update_features(sig)

    # --- Basic Features ---
    def update_features(self, sig):
        features = {
            'Mean': np.mean(sig),
            'Std': np.std(sig),
            'Min': np.min(sig),
            'Max': np.max(sig),
            'RMS': np.sqrt(np.mean(sig**2)),
            'Energy': np.sum(sig**2),
            'Skewness': skew(sig),
            'Kurtosis': kurtosis(sig)
        }
        self.feature_text.delete("1.0", tk.END)
        self.feature_text.insert(tk.END, "--- Basic Features ---\n")
        for k,v in features.items():
            self.feature_text.insert(tk.END, f"{k}: {v:.4f}\n")

    # --- Deep Analysis ---
    def deep_analysis(self):
        if self.signals is None: 
            return
        sig = self.signals[self.current_signal_idx,:]

        # --- Prepare new window ---
        detail_win = tk.Toplevel(self.master)
        detail_win.title(f"Deep Analysis Report - Signal {self.current_signal_idx}")
        detail_win.geometry("1200x800")

        # --- Create figure ---
        fig, axs = plt.subplots(2,2, figsize=(12,8))
        fig.suptitle(f"Deep Analysis - Signal {self.current_signal_idx}", fontsize=14)

        # 1. Time domain plot
        axs[0,0].plot(sig, color="blue")
        axs[0,0].set_title("Time Domain")
        axs[0,0].set_xlabel("Sample")
        axs[0,0].set_ylabel("Amplitude")
        axs[0,0].grid(True)

        # 2. FFT
        N = len(sig)
        fft_vals = np.fft.fft(sig)
        freqs = np.fft.fftfreq(N, d=1)
        axs[0,1].plot(freqs[:N//2], np.abs(fft_vals)[:N//2], color="red")
        axs[0,1].set_title("FFT Spectrum")
        axs[0,1].set_xlabel("Frequency (normalized)")
        axs[0,1].set_ylabel("Magnitude")
        axs[0,1].grid(True)

        # 3. Histogram
        axs[1,0].hist(sig, bins=50, color="green", alpha=0.7)
        axs[1,0].set_title("Value Distribution (Histogram)")
        axs[1,0].set_xlabel("Value")
        axs[1,0].set_ylabel("Count")
        axs[1,0].grid(True)

        # 4. Spectrogram
        axs[1,1].specgram(sig, NFFT=256, Fs=1, noverlap=128, cmap="viridis")
        axs[1,1].set_title("Spectrogram")
        axs[1,1].set_xlabel("Time")
        axs[1,1].set_ylabel("Frequency")

        fig.tight_layout(rect=[0,0,1,0.96])

        # Embed figure in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=detail_win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        # --- Save Report Button ---
        def save_report():
            file = filedialog.asksaveasfilename(defaultextension=".png", 
                                                filetypes=[("PNG Image","*.png"),("PDF File","*.pdf")])
            if file:
                fig.savefig(file, dpi=300)
                messagebox.showinfo("Saved", f"Report saved to:\n{file}")

        save_btn = tk.Button(detail_win, text="Save Report", command=save_report)
        save_btn.pack(pady=5)


# --- Run App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalMATAnalyzer(root)
    root.mainloop()
