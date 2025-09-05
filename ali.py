"""
signal_gui_improved_fixed.py

Final patch: strict column-limited feature-summary + robust feature-selection behavior
- Ensures the feature-summary shows only the columns for the current page (never all columns at once)
- Keeps track of `current_selected_features` (what the user selected) separately from computed DataFrame
- If selected features already exist in computed feature_matrix, do NOT recompute (avoid heavy work)
- Adds `Refresh Display` button (Refresh both summary and plotting) to guarantee UI shows the current state
- Ensures update_feature_summary_page is safe and always replaces the widget content (no leftover text)
- Selection routines run in background threads and interact with UI on the main thread only
"""

# -------------------------
# Part 1: imports, constants, utility
# -------------------------
import os
import glob
import random
import math
import threading
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

DEFAULT_TARGET_LEN = 1024
FS = 1024

FEATURE_NAMES = [
    "Mean",
    "Variance",
    "Skewness",
    "Kurtosis",
    "FFT_Power",
    "BandPower_5_20",
    "PeakFreq",
]

def resample_to_length(x, length):
    x = np.asarray(x).flatten()
    if x.size == length:
        return x
    return signal.resample(x, length)

def ensure_length(sig, length):
    sig = np.asarray(sig).flatten()
    if sig.size < length:
        return np.pad(sig, (0, length - sig.size))
    elif sig.size > length:
        return sig[:length]
    return sig


# -------------------------
# Part 2: feature computation helper
# -------------------------
def compute_feature_vector(x):
    x = np.asarray(x).flatten().astype(float)
    out = {}
    try:
        out['Mean'] = float(np.mean(x))
    except Exception:
        out['Mean'] = np.nan
    try:
        out['Variance'] = float(np.var(x))
    except Exception:
        out['Variance'] = np.nan
    try:
        out['Skewness'] = float(stats.skew(x))
    except Exception:
        out['Skewness'] = np.nan
    try:
        out['Kurtosis'] = float(stats.kurtosis(x))
    except Exception:
        out['Kurtosis'] = np.nan
    try:
        fft_vals = rfft(x)
        ps = np.abs(fft_vals) ** 2
        out['FFT_Power'] = float(ps.sum())
        if len(ps) > 2:
            freqs = rfftfreq(len(x), 1/FS)
            mask = (freqs >= 5) & (freqs <= 20)
            out['BandPower_5_20'] = float(ps[mask].sum()) if mask.any() else float(ps.sum())
            out['PeakFreq'] = float(freqs[np.argmax(ps)])
        else:
            out['BandPower_5_20'] = float(ps.sum())
            out['PeakFreq'] = 0.0
    except Exception:
        out['FFT_Power'] = np.nan
        out['BandPower_5_20'] = np.nan
        out['PeakFreq'] = np.nan
    return out


# -------------------------
# Part 3: GUI class (creation + layout)
# -------------------------
class SignalGUI:
    def __init__(self, root):
        self.root = root
        root.title("Signal Project Manager - Final (strict summary)")
        self.root.geometry("1250x780")

        # Data containers
        self.X_signals = []
        self.y_labels = []
        self.class_names = []
        self.feature_matrix = None
        self.scaler = None
        self.model = None

        # track user-selected features (checkbox state) separately from computed df
        self.current_selected_features = []

        # UI state
        self.boxplot_page = 0
        self.summary_page = 0

        # Controls
        self.preview_rows_var = tk.IntVar(value=20)
        self.random_preview_var = tk.IntVar(value=1)
        self.boxplot_page_size_var = tk.IntVar(value=2)
        self.summary_page_size_var = tk.IntVar(value=4)

        # Buttons to disable during background work
        self.busy_widgets = []

        # Build UI
        self.create_widgets()

        # trace changes to page-size and auto-refresh summary when changed
        try:
            self.summary_page_size_var.trace_add('write', lambda *a: self._on_summary_size_change())
        except Exception:
            try:
                self.summary_page_size_var.trace('w', lambda *a: self._on_summary_size_change())
            except Exception:
                pass

        self.log("App started — strict summary patch")

    def create_widgets(self):
        top = ttk.Frame(self.root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)

        ttk.Label(top, text="Target length").pack(side=tk.LEFT)
        self.target_len_var = tk.IntVar(value=DEFAULT_TARGET_LEN)
        ttk.Entry(top, textvariable=self.target_len_var, width=8).pack(side=tk.LEFT, padx=4)
        self.resample_var = tk.IntVar(value=1)
        ttk.Checkbutton(top, text="Resample if needed", variable=self.resample_var).pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="Chunk overlap (%)").pack(side=tk.LEFT, padx=(12,0))
        self.overlap_var = tk.DoubleVar(value=0.0)
        ttk.Entry(top, textvariable=self.overlap_var, width=6).pack(side=tk.LEFT)

        btn_load_mat = ttk.Button(top, text="Load .mat (X1,X2)", command=self.load_mat)
        btn_load_mat.pack(side=tk.LEFT, padx=6)
        self.busy_widgets.append(btn_load_mat)

        btn_load_folder = ttk.Button(top, text="Load folder (CSV per signal)", command=self.load_folder)
        btn_load_folder.pack(side=tk.LEFT, padx=6)
        self.busy_widgets.append(btn_load_folder)

        btn_gen = ttk.Button(top, text="Generate synthetic", command=self.generate_synthetic)
        btn_gen.pack(side=tk.LEFT, padx=6)
        self.busy_widgets.append(btn_gen)

        btn_export = ttk.Button(top, text="Export features CSV", command=self.export_features_csv)
        btn_export.pack(side=tk.LEFT, padx=6)
        self.busy_widgets.append(btn_export)
        
        btn_clear_plot = ttk.Button(top, text="Clear Plot", command=self.clear_plot)
        btn_clear_plot.pack(side=tk.LEFT, padx=6)
        self.busy_widgets.append(btn_clear_plot)

        # ستون‌ها رو تعریف می‌کنیم → سمت راست همیشه بزرگ‌تره
        self.root.columnconfigure(0, weight=1)   # چپ
        self.root.columnconfigure(1, weight=2)   # راست
        self.root.rowconfigure(1, weight=2)

        # نوت‌بوک سمت چپ
        self.nb = ttk.Notebook(self.root)
        self.nb.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        self.tab_data = ttk.Frame(self.nb)
        self.nb.add(self.tab_data, text="Data & Preview")
        self.create_tab_data()

        self.tab_features = ttk.Frame(self.nb)
        self.nb.add(self.tab_features, text="Features & Selection")
        self.create_tab_features()

        self.tab_train = ttk.Frame(self.nb)
        self.nb.add(self.tab_train, text="Train & Evaluate")
        self.create_tab_train()

        self.tab_test = ttk.Frame(self.nb)
        self.nb.add(self.tab_test, text="Test / Classify")
        self.create_tab_test()

        # سمت راست
        right = ttk.Frame(self.root)
        right.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)

        self.fig = Figure(figsize=(6,5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Log").pack(anchor=tk.W)
        self.log_widget = ScrolledText(right, height=10)
        self.log_widget.pack(fill=tk.BOTH, expand=True)


    # -------------------------
    # Part 4: data loading & preview
    
    def clear_plot(self):
        """
        Clear only the plotting area (all boxplots / matplotlib figure) WITHOUT touching
        text widgets (feature_summary, logs, info_var, etc).
        Thread-safe: can be called from background threads.
        """
        def _do_clear():
            try:
                # reset only plotting paging (keep summary pages intact)
                try:
                    self.boxplot_page = 0
                except Exception:
                    pass

                # Fully clear the matplotlib Figure (remove all subplots/artists)
                try:
                    if hasattr(self, 'fig') and self.fig is not None:
                        self.fig.clf()
                        # create a stable blank axis so later plot code expecting self.ax won't crash
                        try:
                            self.ax = self.fig.add_subplot(111)
                            # don't show ticks for the blank canvas
                            self.ax.set_xticks([])
                            self.ax.set_yticks([])
                        except Exception:
                            self.ax = None
                    # redraw canvas without touching any text widgets
                    if hasattr(self, 'canvas') and self.canvas is not None:
                        try:
                            self.canvas.draw_idle()
                        except Exception:
                            try:
                                self.canvas.draw()
                            except Exception:
                                pass
                except Exception as e:
                    # log but do NOT clear any text widgets
                    try:
                        self.log(f"clear_plot: figure clear error: {e}")
                    except Exception:
                        pass

                # Important: DO NOT touch feature_summary, info_var, log_widget, etc.
                # Only log the action (log_widget remains)
                try:
                    self.log("Plot area cleared (only figure/canvas; text widgets left intact)")
                except Exception:
                    pass

            except Exception as e:
                try:
                    self.log(f"clear_plot unexpected error: {e}")
                except Exception:
                    pass

        # thread-safety: schedule on main thread if called from background thread
        if threading.current_thread() is threading.main_thread():
            _do_clear()
        else:
            self.root.after(1, _do_clear)

    # -------------------------
    def create_tab_data(self):
        frm = self.tab_data
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        ttk.Label(left, text="Loaded classes").pack()
        self.class_listbox = tk.Listbox(left, height=8)
        self.class_listbox.pack(fill=tk.Y)

        ttk.Button(left, text="Show random per class", command=self.show_random_per_class).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Show stacked per class", command=lambda: self.show_n_per_class(3)).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Show waveform of index", command=self.show_index_waveform).pack(fill=tk.X, pady=2)

        idxfrm = ttk.Frame(left)
        idxfrm.pack(fill=tk.X, pady=4)
        ttk.Label(idxfrm, text="Index").pack(side=tk.LEFT)
        self.index_var = tk.IntVar(value=0)
        ttk.Entry(idxfrm, textvariable=self.index_var, width=6).pack(side=tk.LEFT)

        ttk.Button(left, text="Clear data", command=self.clear_all).pack(fill=tk.X, pady=6)

        right = ttk.Frame(frm)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Preview (first 400 samples)").pack()
        self.info_var = tk.StringVar(value="No data loaded")
        ttk.Label(right, textvariable=self.info_var).pack(anchor=tk.W)

    def update_class_listbox(self):
        self.class_listbox.delete(0, tk.END)
        for i, name in enumerate(self.class_names):
            count = self.y_labels.count(i)
            self.class_listbox.insert(tk.END, f"{i}: {name} ({count} samples)")
        self.info_var.set(f"Total signals: {len(self.X_signals)} | Classes: {len(self.class_names)}")

    def generate_synthetic(self):
        target_len = int(self.target_len_var.get())
        n_per_class = 100
        freqs = [3, 7]
        self.X_signals = []
        self.y_labels = []
        self.class_names = ['Synth0','Synth1']
        for c in [0,1]:
            for i in range(n_per_class):
                t = np.arange(target_len)/FS
                sig = np.sin(2*math.pi*(freqs[c] + np.random.randn()*0.1)*t) + 0.6*np.random.randn(target_len)
                self.X_signals.append(sig)
                self.y_labels.append(c)
        self.log(f"Generated synthetic {len(self.X_signals)} signals, len={target_len}")
        self.update_class_listbox()
        self.show_random_per_class()

    def show_random_per_class(self):
        if not self.X_signals:
            self.log("No data to preview")
            return
        self.ax.clear()
        plotted = 0
        for c in sorted(set(self.y_labels)):
            idxs = [i for i,y in enumerate(self.y_labels) if y==c]
            if not idxs:
                continue
            idx = random.choice(idxs)
            self.ax.plot(self.X_signals[idx][:min(400,int(self.target_len_var.get()))], label=f"{self.class_names[c]} idx{idx}")
            plotted += 1
        self.ax.legend()
        self.ax.set_title("Random sample per class (first 400 samples)")
        self.canvas.draw()
        self.log(f"Previewed {plotted} random samples (one per class)")

    def show_n_per_class(self, n=3):
        """
        Plot up to n random samples per class stacked, using bright color families per class.
        Within each class shades are very similar (small variation).
        """
        if not self.X_signals:
            self.log("No data to preview")
            return

        self.ax.clear()
        offset = 0

        from matplotlib import cm

        # choose a colormap family per class index (bright families)
        colormap_families = [cm.get_cmap('Reds'), cm.get_cmap('Blues'),
                            cm.get_cmap('Greens'), cm.get_cmap('Purples'), cm.get_cmap('Oranges')]

        for c in sorted(set(self.y_labels)):
            idxs = [i for i, y in enumerate(self.y_labels) if y == c]
            if len(idxs) >= n:
                sel = random.sample(idxs, n)
            else:
                sel = idxs

            num_sel = max(1, len(sel))
            cmap = colormap_families[c % len(colormap_families)]

            # define narrow, bright range for colormap sampling
            LO = 0.70   # lower bound of colormap (bright)
            HI = 0.88   # upper bound (still bright)
            span = HI - LO

            for i, idx in enumerate(sel):
                sig = self.X_signals[idx]

                # small variation around equally-spaced positions in [LO, HI]
                # using center-offset small steps so differences inside class are tiny
                if num_sel == 1:
                    pos = (LO + HI) / 2.0
                else:
                    # distribute within narrow range but keep spread very small
                    pos = LO + ( (i + 0.5) / num_sel ) * span

                # fetch RGBA from cmap, then slightly adjust brightness to make lines pop
                try:
                    rgba = cmap(pos)
                    # brightness factor near 0.9-1.0 (very small variation)
                    center_brightness = 0.94
                    # offset per sample: small step (e.g. +/- 0.02)
                    step = 0.02
                    brightness = center_brightness + ( (i - (num_sel-1)/2.0) * (step / max(1, num_sel-1)) )
                    # clamp
                    brightness = max(0.7, min(1.0, brightness))
                    # apply brightness by scaling RGB channels (keep alpha from cmap)
                    color = (min(1, rgba[0]*brightness), min(1, rgba[1]*brightness), min(1, rgba[2]*brightness), rgba[3])
                except Exception:
                    color = None

                label = (self.class_names[c] if i == 0 else None)
                try:
                    self.ax.plot(
                        sig[:min(400, int(self.target_len_var.get()))] + i * 3 + offset,
                        alpha=0.95,             # more opaque
                        label=label,
                        linewidth=1.6,         # slightly thicker for visibility
                        color=color
                    )
                except Exception:
                    # fallback without explicit color
                    self.ax.plot(
                        sig[:min(400, int(self.target_len_var.get()))] + i * 3 + offset,
                        alpha=0.95,
                        label=label,
                        linewidth=1.6
                    )

            offset += 12

        # legend de-dup
        try:
            handles, labels = self.ax.get_legend_handles_labels()
            if handles:
                seen = set()
                unique = []
                for h, lab in zip(handles, labels):
                    if lab and lab not in seen:
                        seen.add(lab)
                        unique.append((h, lab))
                if unique:
                    hs, ls = zip(*unique)
                    self.ax.legend(hs, ls)
                else:
                    self.ax.legend()
            else:
                self.ax.legend()
        except Exception:
            try:
                self.ax.legend()
            except Exception:
                pass

        try:
            self.canvas.draw()
        except Exception:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

        self.log(f"Stacked view for up to {n} random samples per class (bright, low intra-class variation)")



    def show_index_waveform(self):
        idx = int(self.index_var.get())
        if idx < 0 or idx >= len(self.X_signals):
            self.log("Index out of range")
            return
        self.ax.clear()
        self.ax.plot(self.X_signals[idx][:min(800,int(self.target_len_var.get()))])
        self.ax.set_title(f"Signal index {idx} (label={self.y_labels[idx]})")
        self.canvas.draw()
        self.log(f"Shown waveform of index {idx}")

    # -------------------------
    # Data loading: load .mat and folder
    # -------------------------
    def load_mat(self):
        path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat"), ("All files","*.*")])
        if not path:
            return
        try:
            data = loadmat(path)
        except Exception as e:
            self.log(f"Error loading .mat: {e}")
            return
        vars_list = [k for k in data.keys() if not k.startswith('__')]
        # try to find X1 and X2 case-insensitive
        keys = {k.lower(): k for k in vars_list}
        if 'x1' not in keys or 'x2' not in keys:
            self.log(f".mat must contain X1 and X2 variables. Found: {vars_list}")
            return
        X1 = np.asarray(data[keys['x1']])
        X2 = np.asarray(data[keys['x2']])
        self.process_and_store_mat_arrays(X1, X2)

    def process_and_store_mat_arrays(self, X1, X2):
        """
        Convert loaded MATLAB arrays X1 and X2 into self.X_signals and self.y_labels.
        Handles 1D/2D shapes, resampling, chunking with optional overlap, padding/truncation.
        """
        target_len = int(self.target_len_var.get())
        do_resample = bool(self.resample_var.get())
        overlap_pct = float(self.overlap_var.get())/100.0

        def to_signals(arr):
            arr = np.asarray(arr)
            # flatten 2D with single row/col
            if arr.ndim == 2 and 1 in arr.shape:
                arr = arr.flatten()
            sigs = []
            if arr.ndim == 2:
                # if one dimension equals target_len treat rows/cols as signals
                if arr.shape[1] == target_len:
                    for r in range(arr.shape[0]):
                        sigs.append(arr[r,:].astype(float))
                elif arr.shape[0] == target_len:
                    for c in range(arr.shape[1]):
                        sigs.append(arr[:,c].astype(float))
                else:
                    # treat each row as a signal and resample/ensure length
                    for r in range(arr.shape[0]):
                        s = arr[r,:].astype(float)
                        if s.size != target_len:
                            s = resample_to_length(s, target_len) if do_resample else ensure_length(s, target_len)
                        sigs.append(s)
            elif arr.ndim == 1:
                n = arr.size
                if n == target_len:
                    sigs.append(arr.astype(float))
                elif n > target_len:
                    # chunk with overlap or split
                    if overlap_pct <= 0:
                        if n % target_len == 0:
                            chunks = n // target_len
                            for i in range(chunks):
                                s = arr[i*target_len:(i+1)*target_len]
                                sigs.append(s.astype(float))
                        else:
                            chunks = n // target_len
                            for i in range(chunks):
                                s = arr[i*target_len:(i+1)*target_len]
                                sigs.append(s.astype(float))
                            last = arr[chunks*target_len:]
                            last = ensure_length(last, target_len)
                            sigs.append(last.astype(float))
                    else:
                        step = int(max(1, target_len * (1 - overlap_pct)))
                        for start in range(0, n - target_len + 1, step):
                            s = arr[start:start+target_len]
                            sigs.append(s.astype(float))
                        # if tail remains, add last chunk
                        if (n - target_len) % step != 0:
                            s = arr[-target_len:]
                            sigs.append(s.astype(float))
                else:
                    # n < target_len
                    if do_resample:
                        sigs.append(resample_to_length(arr, target_len).astype(float))
                    else:
                        sigs.append(ensure_length(arr, target_len).astype(float))
            else:
                # fallback flatten then try again
                flat = arr.flatten()
                sigs.extend(to_signals(flat))
            return sigs

        s1 = to_signals(X1)
        s2 = to_signals(X2)
        # overwrite current dataset
        self.X_signals = []
        self.y_labels = []
        for s in s1:
            self.X_signals.append(s)
            self.y_labels.append(0)
        for s in s2:
            self.X_signals.append(s)
            self.y_labels.append(1)
        self.class_names = ['X1','X2']
        self.log(f"Loaded .mat -> X1: {len(s1)} signals, X2: {len(s2)} signals, target_len={target_len}, resample={do_resample}, overlap%={overlap_pct*100}")
        self.update_class_listbox()
        # show a quick preview
        try:
            self.show_random_per_class()
        except Exception:
            pass
    # -------------------------
    # Part 5: features, selection, plotting, training, testing
    # -------------------------
    def create_tab_features(self):
        frm = self.tab_features
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        ttk.Label(left, text="Select features to compute").pack()
        self.feature_checks = {}
        for name in FEATURE_NAMES:
            var = tk.IntVar(value=1)
            chk = ttk.Checkbutton(left, text=name, variable=var)
            chk.pack(anchor=tk.W)
            self.feature_checks[name] = var

        self.btn_compute = ttk.Button(left, text="Compute features", command=self.compute_features)
        self.btn_compute.pack(fill=tk.X, pady=4)
        self.busy_widgets.append(self.btn_compute)

        ttk.Button(left, text="Show feature table (preview)", command=self.show_feature_table).pack(fill=tk.X, pady=2)
        ttk.Label(left, text="Preview rows:").pack(anchor=tk.W)
        ttk.Spinbox(left, from_=5, to=500, textvariable=self.preview_rows_var, width=6).pack(anchor=tk.W)
        ttk.Checkbutton(left, text="Random sample preview", variable=self.random_preview_var).pack(anchor=tk.W)

        ttk.Label(left, text="Boxplots per page:").pack(anchor=tk.W, pady=(8,0))
        ttk.Spinbox(left, from_=1, to=12, textvariable=self.boxplot_page_size_var, width=6).pack(anchor=tk.W)
        btnfrm = ttk.Frame(left)
        btnfrm.pack(fill=tk.X, pady=4)
        self.btn_boxplots = ttk.Button(btnfrm, text="Boxplots (page)", command=self.plot_boxplots_page)
        self.btn_boxplots.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.busy_widgets.append(self.btn_boxplots)
        ttk.Button(btnfrm, text="Prev page", command=lambda: self.change_boxplot_page(-1)).pack(side=tk.LEFT)
        ttk.Button(btnfrm, text="Next page", command=lambda: self.change_boxplot_page(1)).pack(side=tk.LEFT)

        ttk.Button(left, text="PCA 2D scatter", command=self.plot_pca_scatter).pack(fill=tk.X, pady=2)
        self.btn_kbest = ttk.Button(left, text="SelectKBest (k=3)", command=lambda: self.select_kbest(3))
        self.btn_kbest.pack(fill=tk.X, pady=2)
        self.busy_widgets.append(self.btn_kbest)
        self.btn_incr = ttk.Button(left, text="Incremental-add eval", command=self.incremental_add_eval)
        self.btn_incr.pack(fill=tk.X, pady=2)
        self.busy_widgets.append(self.btn_incr)

        ttk.Separator(left, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Label(left, text="Feature summary (paged, strict columns)").pack()
        sumfrm = ttk.Frame(left)
        sumfrm.pack(fill=tk.X)
        ttk.Label(sumfrm, text="Cols/page:").pack(side=tk.LEFT)
        ttk.Spinbox(sumfrm, from_=1, to=6, textvariable=self.summary_page_size_var, width=4).pack(side=tk.LEFT)
        ttk.Button(sumfrm, text="Prev", command=lambda: self.change_summary_page(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(sumfrm, text="Next", command=lambda: self.change_summary_page(1)).pack(side=tk.LEFT)
        ttk.Button(sumfrm, text="Refresh Display", command=self.refresh_display).pack(side=tk.LEFT, padx=4)
        ttk.Button(sumfrm, text="Clear Summary", command=self.clear_summary).pack(side=tk.LEFT, padx=4)

        right = ttk.Frame(frm)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Feature summary").pack(anchor=tk.W)
        self.feature_summary = ScrolledText(right, height=20)
        self.feature_summary.pack(fill=tk.BOTH, expand=True)

    def compute_features(self):
        if not self.X_signals:
            self.log("No data to compute features")
            return
        chosen = [name for name,var in self.feature_checks.items() if var.get()==1]
        if not chosen:
            self.log("No features selected")
            return
        # store selected features explicitly
        self.current_selected_features = chosen.copy()
        # immediate computing message
        try:
            self.feature_summary.delete(1.0, tk.END)
            self.feature_summary.insert(tk.END, "Computing features...\n")
            self.feature_summary.see(tk.END)
        except Exception:
            pass
        thread = threading.Thread(target=self._compute_features_thread, args=(chosen,), daemon=True)
        thread.start()

    def _compute_features_thread(self, chosen):
        try:
            self.set_busy(True)
            rows = []
            total = len(self.X_signals)
            for i,x in enumerate(self.X_signals):
                vec = compute_feature_vector(x)
                row = [vec.get(name, np.nan) for name in chosen]
                rows.append(row)
                if (i+1) % 50 == 0:
                    self.log(f"Computed features for {i+1}/{total} signals...")
            df = pd.DataFrame(rows, columns=chosen)
            df['label'] = self.y_labels
            def _finish():
                # assign feature matrix and ensure selected list matches computed columns
                self.feature_matrix = df
                self.current_selected_features = [c for c in df.columns if c!='label']
                self.summary_page = 0
                self.update_feature_summary_page()
                self.log(f"Computed feature matrix: {df.shape}")
                self.set_busy(False)
            self.root.after(1, _finish)
        except Exception as e:
            self.log(f"Feature computation failed: {e}")
            self.set_busy(False)

    def _on_summary_size_change(self):
        # whenever page size changes, reset to first page and refresh summary
        try:
            self.summary_page = 0
            self.update_feature_summary_page()
            self.log(f"Summary cols/page changed -> {self.summary_page_size_var.get()} (reset to page 1)")
        except Exception:
            pass

    def update_feature_summary_page(self):
        # safe to call from any thread; ensure runs on main thread
        if threading.current_thread() is not threading.main_thread():
            self.root.after(1, self.update_feature_summary_page)
            return
        # decide which feature list to page: prefer computed DF columns, else current_selected_features
        if self.feature_matrix is not None:
            available_cols = [c for c in self.feature_matrix.columns if c!='label']
        else:
            available_cols = self.current_selected_features.copy()
        per_page = max(1, int(self.summary_page_size_var.get()))
        total_pages = max(1, math.ceil(len(available_cols)/per_page))
        # clamp page index
        if self.summary_page < 0:
            self.summary_page = 0
        if self.summary_page >= total_pages:
            self.summary_page = total_pages-1
        start = self.summary_page * per_page
        sel = available_cols[start:start+per_page]
        # prepare text
        if not sel:
            txt = "No features selected (or none computed). Use the checkboxes and Compute features."
        else:
            if self.feature_matrix is None:
                # no DF yet — show selected feature names only
                txt = f"Page {self.summary_page+1}/{total_pages} - selected features (not computed yet):\n\n"
                txt += '\n'.join([f"{i+1}. {name}" for i,name in enumerate(sel)])
            else:
                try:
                    part = self.feature_matrix[sel].describe().round(6).to_string()
                    head = self.feature_matrix[['label'] + sel].head(10).to_string(index=False)
                    txt = f"Page {self.summary_page+1}/{total_pages} - showing {len(sel)} feature(s)\n\n" + part + "\n\nHead (first up to 10 rows):\n" + head
                except Exception as e:
                    txt = f"Unable to prepare summary: {e}"
        # update widget (replace fully)
        try:
            self.feature_summary.delete(1.0, tk.END)
            self.feature_summary.insert(tk.END, txt)
            self.feature_summary.see(1.0)
        except Exception:
            pass

    def change_summary_page(self, delta):
        self.summary_page += delta
        self.update_feature_summary_page()

    def refresh_display(self):
        # refresh both summary and current plotted boxplots (keeps state consistent)
        try:
            self.update_feature_summary_page()
            # also refresh the figure to show current page of boxplots if features exist
            if self.feature_matrix is not None:
                self.plot_boxplots_page()
            else:
                # clear figure if nothing computed
                self.ax.clear()
                self.canvas.draw()
            self.log('Display refreshed')
        except Exception as e:
            self.log(f'Refresh failed: {e}')

    def clear_summary(self):
        self.summary_page = 0
        try:
            self.feature_summary.delete(1.0, tk.END)
            self.feature_summary.insert(tk.END, "Summary cleared")
            self.feature_summary.see(1.0)
        except Exception:
            pass
        self.log('Feature summary cleared')


    def show_feature_table(self):
        if self.feature_matrix is None:
            self.log("No feature matrix. Compute features first")
            return
        preview_n = int(self.preview_rows_var.get())
        rnd = bool(self.random_preview_var.get())
        df = self.feature_matrix
        if rnd and preview_n < len(df):
            dfp = df.sample(n=preview_n, random_state=42)
        else:
            dfp = df.head(preview_n)
        w = tk.Toplevel(self.root)
        w.title(f"Feature table preview ({len(dfp)} rows)")
        cols = list(dfp.columns)
        tv = ttk.Treeview(w, columns=cols, show='headings')
        vsb = ttk.Scrollbar(w, orient="vertical", command=tv.yview)
        hsb = ttk.Scrollbar(w, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=120, anchor=tk.CENTER)
        for i, row in dfp.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    vals.append("")
                else:
                    try:
                        vals.append(f"{float(v):.6g}")
                    except Exception:
                        vals.append(str(v))
            tv.insert('', tk.END, values=vals)
        tv.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        w.grid_rowconfigure(0, weight=1)
        w.grid_columnconfigure(0, weight=1)

    def change_boxplot_page(self, delta):
        self.boxplot_page += delta
        if self.boxplot_page < 0:
            self.boxplot_page = 0
        self.plot_boxplots_page()

    def plot_boxplots_page(self):
        if self.feature_matrix is None:
            self.log("No features: compute first")
            return
        df = self.feature_matrix
        chosen = [c for c in df.columns if c != 'label']
        per_page = int(self.boxplot_page_size_var.get())
        total_pages = max(1, math.ceil(len(chosen)/per_page))
        if self.boxplot_page >= total_pages:
            self.boxplot_page = total_pages-1
        start = self.boxplot_page * per_page
        sel_feats = chosen[start:start+per_page]
        if not sel_feats:
            self.log("No features on this page")
            return
        cols = min(3, len(sel_feats))
        rows = math.ceil(len(sel_feats)/cols)
        self.fig.clf()
        for i,name in enumerate(sel_feats):
            ax = self.fig.add_subplot(rows, cols, i+1)
            groups = [df[df['label']==lab][name].dropna().values for lab in sorted(df['label'].unique())]
            try:
                ax.boxplot(groups, labels=[self.class_names[l] if l < len(self.class_names) else f"C{l}" for l in sorted(df['label'].unique())])
            except Exception:
                ax.text(0.5,0.5,'no data',ha='center')
            ax.set_title(name)
            if 'Power' in name or 'FFT' in name:
                try:
                    ax.set_yscale('log')
                except Exception:
                    pass
        self.fig.tight_layout()
        self.canvas.draw()
        self.log(f"Displayed boxplots page {self.boxplot_page+1}/{total_pages}")

    def plot_pca_scatter(self):
        if self.feature_matrix is None:
            self.log("No features: compute first")
            return
        df = self.feature_matrix.dropna()
        X = df.drop(columns=['label']).values
        if X.shape[1] < 2:
            self.log("Need at least 2 features for PCA")
            return
        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        labels = df['label'].values
        for lab in sorted(np.unique(labels)):
            sel = labels==lab
            ax.scatter(Z[sel,0], Z[sel,1], label=self.class_names[lab] if lab < len(self.class_names) else f"C{lab}")
        ax.legend()
        ax.set_title('PCA 2D scatter')
        self.canvas.draw()
        self.log('PCA 2D plotted')

    def select_kbest(self, k=3):
        if self.feature_matrix is None:
            self.log('Compute features first')
            return
        thread = threading.Thread(target=self._select_kbest_thread, args=(k,), daemon=True)
        thread.start()

    def _select_kbest_thread(self, k):
        try:
            self.set_busy(True)
            X = self.feature_matrix.drop(columns=['label']).values
            y = self.feature_matrix['label'].values
            k = min(k, X.shape[1])
            sel = SelectKBest(score_func=mutual_info_classif, k=k)
            sel.fit(X, y)
            feat_cols = [c for c in self.feature_matrix.columns if c!='label']
            chosen = [feat_cols[i] for i in range(len(feat_cols)) if sel.get_support()[i]]
            # finish on main thread
            def _finish():
                try:
                    self.log(f'SelectKBest selected: {chosen}')
                    # update checkboxes
                    for name, var in self.feature_checks.items():
                        var.set(1 if name in chosen else 0)
                    # update current_selected_features
                    self.current_selected_features = chosen.copy()
                    # if feature_matrix already contains chosen cols -> don't recompute, just refresh displays
                    missing = [c for c in chosen if c not in self.feature_matrix.columns]
                    if missing:
                        self.log(f'Missing columns {missing} — computing features for selected set.')
                        # compute_features will set busy True/False appropriately
                        self.compute_features()
                    else:
                        # update summary and boxplots to reflect new selection
                        self.summary_page = 0
                        self.update_feature_summary_page()
                        try:
                            self.plot_boxplots_page()
                        except Exception:
                            # if plotting fails, ignore but log
                            self.log("Plotting boxplots after SelectKBest failed.")
                        self.set_busy(False)
                except Exception as e:
                    self.log(f'SelectKBest finish error: {e}')
                    self.set_busy(False)
            self.root.after(1, _finish)
        except Exception as e:
            self.log(f'SelectKBest failed: {e}')
            self.set_busy(False)

    def incremental_add_eval(self):
        if self.feature_matrix is None:
            self.log('Compute features first')
            return
        thread = threading.Thread(target=self._incremental_add_thread, daemon=True)
        thread.start()

    def _incremental_add_thread(self):
        try:
            self.set_busy(True)
            order = [name for name,var in self.feature_checks.items() if var.get()==1]
            if not order:
                self.log('No features selected for incremental test')
                self.set_busy(False)
                return
            results = []
            for k in range(1, len(order)+1):
                use = order[:k]
                X = self.feature_matrix[use].values
                y = self.feature_matrix['label'].values
                clf = SVC(kernel='rbf')
                try:
                    scores = cross_val_score(clf, X, y, cv=4)
                    acc = float(scores.mean())
                except Exception:
                    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.3, stratify=y)
                    clf.fit(Xtr,ytr)
                    acc = float(clf.score(Xte,yte))
                results.append((k, use.copy(), acc))
            def _finish():
                self.fig.clf()
                ax = self.fig.add_subplot(111)
                ax.plot([r[0] for r in results], [r[2] for r in results], marker='o')
                ax.set_xlabel('# features')
                ax.set_ylabel('Accuracy')
                ax.set_title('Incremental add evaluation')
                self.canvas.draw()
                txt = '\n'.join([f"{r[0]}: {r[1]} -> {r[2]:.3f}" for r in results])
                self.log('Incremental results:\n' + txt)
                self.set_busy(False)
            self.root.after(1, _finish)
        except Exception as e:
            self.log(f'Incremental eval failed: {e}')
            self.set_busy(False)

    def train_model(self):
        if self.feature_matrix is None:
            self.log('Compute features first')
            return
        thread = threading.Thread(target=self._train_model_thread, daemon=True)
        thread.start()

    def _train_model_thread(self):
        try:
            self.set_busy(True)
            test_size = float(self.test_size_var.get())
            X = self.feature_matrix.drop(columns=['label']).values
            y = self.feature_matrix['label'].values
            Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=test_size, stratify=y, random_state=42)
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xte_s = scaler.transform(Xte)
            clf = SVC(kernel='rbf', probability=True)
            clf.fit(Xtr_s, ytr)
            train_acc = float(clf.score(Xtr_s, ytr))
            test_acc = float(clf.score(Xte_s, yte))
            def _finish():
                self.scaler = scaler
                self.model = clf
                self.train_result_var.set(f"Train: {train_acc:.3f} | Test: {test_acc:.3f}")
                self.log(f"Trained SVM | train={train_acc:.3f} test={test_acc:.3f}")
                self.set_busy(False)
            self.root.after(1, _finish)
        except Exception as e:
            self.log(f"Training failed: {e}")
            self.set_busy(False)

    def cross_val(self):
        if self.feature_matrix is None:
            self.log('Compute features first')
            return
        thread = threading.Thread(target=self._cross_val_thread, daemon=True)
        thread.start()

    def _cross_val_thread(self):
        try:
            self.set_busy(True)
            X = self.feature_matrix.drop(columns=['label']).values
            y = self.feature_matrix['label'].values
            clf = SVC(kernel='rbf')
            scores = cross_val_score(clf, X, y, cv=5)
            self.log(f'Cross-val scores (5-fold): {scores} mean={scores.mean():.3f}')
            self.set_busy(False)
        except Exception as e:
            self.log(f'Cross-val failed: {e}')
            self.set_busy(False)

    def show_confusion(self):
        if self.model is None or self.scaler is None:
            self.log('Train a model first')
            return
        X = self.feature_matrix.drop(columns=['label']).values
        y = self.feature_matrix['label'].values
        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs)
        cm = confusion_matrix(y, preds)
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        disp = ConfusionMatrixDisplay(cm, display_labels=[self.class_names[i] if i<len(self.class_names) else f"C{i}" for i in range(cm.shape[0])])
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        self.canvas.draw()
        self.log('Confusion matrix displayed')

    def save_model(self):
        if self.model is None or self.scaler is None or self.feature_matrix is None:
            self.log('No trained model to save')
            return
        path = filedialog.asksaveasfilename(defaultextension='.joblib', filetypes=[('Joblib','*.joblib')])
        if not path:
            return
        joblib.dump({'model':self.model, 'scaler':self.scaler, 'feature_cols':list(self.feature_matrix.columns.drop('label'))}, path)
        self.log(f'Model saved to {path}')

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[('Joblib','*.joblib'),('All','*.*')])
        if not path:
            return
        try:
            d = joblib.load(path)
            self.model = d.get('model')
            self.scaler = d.get('scaler')
            featcols = d.get('feature_cols')
            self.log(f'Loaded model from {path} with features {featcols}')
        except Exception as e:
            self.log(f'Load model failed: {e}')

    # -------------------------
    # Part 6: testing, misc, export, busy helpers, main
    # -------------------------
    def create_tab_train(self):
        frm = self.tab_train
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        ttk.Label(left, text="Train settings").pack()
        ttk.Label(left, text="Test size (0-1)").pack()
        self.test_size_var = tk.DoubleVar(value=0.3)
        ttk.Entry(left, textvariable=self.test_size_var, width=6).pack()

        self.btn_train = ttk.Button(left, text="Train SVM", command=self.train_model)
        self.btn_train.pack(fill=tk.X, pady=6)
        self.busy_widgets.append(self.btn_train)
        self.btn_cv = ttk.Button(left, text="Cross-val (5-fold)", command=self.cross_val)
        self.btn_cv.pack(fill=tk.X, pady=2)
        self.busy_widgets.append(self.btn_cv)
        ttk.Button(left, text="Show confusion matrix", command=self.show_confusion).pack(fill=tk.X, pady=2)
        btn_save = ttk.Button(left, text="Save model", command=self.save_model)
        btn_save.pack(fill=tk.X, pady=2)
        self.busy_widgets.append(btn_save)
        btn_load = ttk.Button(left, text="Load model", command=self.load_model)
        btn_load.pack(fill=tk.X, pady=2)
        self.busy_widgets.append(btn_load)

        right = ttk.Frame(frm)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.train_result_var = tk.StringVar(value="No training yet")
        ttk.Label(right, textvariable=self.train_result_var).pack(anchor=tk.W)

    def create_tab_test(self):
        frm = self.tab_test
        ttk.Button(frm, text="Generate test signal and classify", command=self.generate_and_classify).pack(pady=6)
        ttk.Label(frm, text="Model prediction / probabilities shown in Log").pack()

    def generate_and_classify(self):
        target_len = int(self.target_len_var.get())
        t = np.arange(target_len)/FS

        freq = np.random.uniform(2, 10)
        amp = np.random.uniform(0.8, 1.2)
        noise_level = np.random.uniform(0.3, 0.6)

        sig = amp * np.sin(2 * math.pi * freq * t) + noise_level * np.random.randn(target_len)


        self.ax.clear()
        self.ax.plot(sig[:min(800, target_len)])
        self.ax.set_title(f'Test signal freq={freq:.2f}, amp={amp:.2f}')
        self.canvas.draw()


        thread = threading.Thread(target=lambda: self.classify_signal(sig), daemon=True)
        thread.start()



    def classify_signal(self, sig):
        try:
            if self.model is None or self.scaler is None or self.feature_matrix is None:
                self.log('Need a trained model and feature matrix')
                return
            feat_cols = [c for c in self.feature_matrix.columns if c!='label']
            vec = compute_feature_vector(sig)
            vals = [vec.get(name, np.nan) for name in feat_cols]
            X = np.array(vals).reshape(1,-1)
            Xs = self.scaler.transform(X)
            pred = self.model.predict(Xs)[0]
            probs = self.model.predict_proba(Xs)[0] if hasattr(self.model, 'predict_proba') else None
            self.log(f'Classified signal -> Pred: {self.class_names[pred] if pred < len(self.class_names) else pred}')
            if probs is not None:
                prob_txt = ', '.join([f"{(self.class_names[i] if i < len(self.class_names) else i)}:{p:.3f}" for i,p in enumerate(probs)])
                self.log('Probs: ' + prob_txt)
        except Exception as e:
            self.log(f'Classification failed: {e}')

    def export_features_csv(self):
        if self.feature_matrix is None:
            self.log('No features to export')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        self.feature_matrix.to_csv(path, index=False)
        self.log(f'Feature matrix exported to {path}')

    def clear_all(self):
        self.X_signals = []
        self.y_labels = []
        self.class_names = []
        self.feature_matrix = None
        self.model = None
        self.scaler = None
        self.current_selected_features = []
        self.update_class_listbox()
        self.ax.clear(); self.canvas.draw()
        try:
            self.feature_summary.delete(1.0, tk.END)
        except Exception:
            pass
        self.log('Cleared all data and models')

    # Busy helpers
    def set_busy(self, busy=True):
        def _set():
            for w in self.busy_widgets:
                try:
                    if busy:
                        w.state(['disabled'])
                    else:
                        w.state(['!disabled'])
                except Exception:
                    try:
                        w.config(state=('disabled' if busy else 'normal'))
                    except Exception:
                        pass
        if threading.current_thread() is threading.main_thread():
            _set()
        else:
            self.root.after(1, _set)

    def load_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        subs = [d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)]
        if not subs:
            self.log("No subfolders found in folder")
            return
        target_len = int(self.target_len_var.get())
        self.X_signals = []
        self.y_labels = []
        self.class_names = []
        for idx, sub in enumerate(sorted(subs)):
            self.class_names.append(os.path.basename(sub))
            files = glob.glob(os.path.join(sub, "*.csv"))
            for f in files:
                try:
                    arr = np.loadtxt(f, delimiter=',')
                    arr = ensure_length(arr, target_len)
                    self.X_signals.append(arr.astype(float))
                    self.y_labels.append(idx)
                except Exception as e:
                    self.log(f"skip {f}: {e}")
        self.log(f"Loaded folder -> {len(self.X_signals)} signals from {len(self.class_names)} classes")
        self.update_class_listbox()
        try:
            self.show_random_per_class()
        except Exception:
            pass
  
    # Logging
    def log(self, txt):
        def _append():
            try:
                self.log_widget.insert(tk.END, f"{txt}\n")
                self.log_widget.see(tk.END)
            except Exception:
                pass
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.root.after(1, _append)


if __name__ == '__main__':
    root = tk.Tk()
    app = SignalGUI(root)
    root.mainloop()
