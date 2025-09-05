#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Analyzer & Classifier (monolithic, full)

Features:
 - Input: Synthetic / Folder(CSV) / .mat auto-detect (pads/truncates to 1024)
 - Preview: Canvas embedded (matplotlib) with Next/Prev buttons, one plot at a time
 - Feature extraction: time + freq features (presets: basic/advanced/custom)
 - Boxplot viewer with navigation (one feature per view) and PCA scatter
 - Train: optional SelectKBest (ANOVA), StandardScaler, SVM (RBF)
 - Incremental feature eval: accuracy vs number of features (ANOVA order)
 - Test: generate 3 random signals, classify using trained pipeline
 - Export/import model (.pkl) with metadata, export features (.csv)
 - Logging pane, save/clear log, progress bar, UI busy locking
 - All-in-one single file. Readable, commented.
"""

import os
import re
import threading
import datetime
import pickle
import traceback
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('TkAgg')  # ensure TkAgg backend for embedding
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

# ----------------------------
# Constants & small utilities
# ----------------------------
DEFAULT_SIG_LEN = 1024
DEFAULT_FS = 1.0

def ensure_2d_signals(arr, target_len=DEFAULT_SIG_LEN):
    """
    Normalize various array shapes to 2D array of signals (n_signals, target_len).
    Pads/truncates rows as needed.
    """
    a = np.asarray(arr)
    if a.size == 0:
        return np.zeros((0, target_len), dtype=float)
    if a.ndim == 1:
        if a.size % target_len == 0:
            return a.reshape((-1, target_len)).astype(float)
        else:
            if a.size >= target_len:
                return a[:target_len].reshape(1, -1).astype(float)
            else:
                return np.pad(a, (0, target_len - a.size), 'constant').reshape(1, -1).astype(float)
    elif a.ndim == 2:
        rows = []
        for r in a:
            r = np.asarray(r).flatten()
            if r.size > target_len:
                rows.append(r[:target_len])
            elif r.size < target_len:
                rows.append(np.pad(r, (0, target_len - r.size), 'constant'))
            else:
                rows.append(r)
        return np.vstack(rows).astype(float)
    else:
        # flatten higher dims
        return ensure_2d_signals(a.ravel(), target_len=target_len)

def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default

# ----------------------------
# Main Application class
# ----------------------------
class SignalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Analyzer & Classifier")
        self.root.geometry("1260x820")

        # Data & state
        self.signal_length = DEFAULT_SIG_LEN
        self.fs = DEFAULT_FS

        self.data = None            # numpy array shape (n_samples, signal_length)
        self.labels = None          # numpy array shape (n_samples,)
        self.features_full = None   # (n_samples, P)
        self.features = None        # (n_samples, p) active after selector
        self.all_feature_names = [] # list of P names
        self.feature_names = []     # active p names

        # model elements
        self.model = None
        self.scaler = None
        self.selector = None

        # test signals
        self.test_signals = []

        # UI & threading
        self._busy_lock = threading.Lock()
        self._is_busy = False
        self.ui_buttons = []

        # last confusion
        self._last_cm = None

        # build UI
        self._build_ui()
        self._log("Application started")

    # ----------------------------
    # Build entire UI
    # ----------------------------
    def _build_ui(self):
        # Top controls (fs and band)
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', padx=8, pady=(6,2))

        ttk.Label(top_frame, text="Sampling rate (fs Hz):").pack(side='left')
        self.fs_var = tk.DoubleVar(value=self.fs)
        ent_fs = ttk.Entry(top_frame, textvariable=self.fs_var, width=8)
        ent_fs.pack(side='left', padx=(4,12))
        self.ui_buttons.append(ent_fs)

        ttk.Label(top_frame, text="Band-low:").pack(side='left')
        self.band_low = tk.DoubleVar(value=0.0)
        ttk.Entry(top_frame, textvariable=self.band_low, width=8).pack(side='left', padx=4)
        ttk.Label(top_frame, text="Band-high:").pack(side='left', padx=(8,0))
        self.band_high = tk.DoubleVar(value=10.0)
        ttk.Entry(top_frame, textvariable=self.band_high, width=8).pack(side='left', padx=4)

        # Notebook tabs
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True, padx=8, pady=6)

        # Tabs
        self.tab_input = ttk.Frame(self.nb)
        self.tab_preview = ttk.Frame(self.nb)
        self.tab_features = ttk.Frame(self.nb)
        self.tab_train = ttk.Frame(self.nb)
        self.tab_test = ttk.Frame(self.nb)
        self.tab_help = ttk.Frame(self.nb)

        self.nb.add(self.tab_input, text="1. Input Data")
        self.nb.add(self.tab_preview, text="2. Preview")
        self.nb.add(self.tab_features, text="3. Features")
        self.nb.add(self.tab_train, text="4. Train & Eval")
        self.nb.add(self.tab_test, text="5. Test")
        self.nb.add(self.tab_help, text="6. Help & Log")

        # Build content for tabs
        self._build_input_tab(self.tab_input)
        self._build_preview_tab(self.tab_preview)
        self._build_features_tab(self.tab_features)
        self._build_train_tab(self.tab_train)
        self._build_test_tab(self.tab_test)
        self._build_help_tab(self.tab_help)

        # Progress & status
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill='x', padx=8, pady=(0,8))
        self.progress = ttk.Progressbar(bottom_frame, mode='indeterminate')
        self.progress.pack(fill='x', side='left', expand=True, padx=(0,8))
        self.status_label = ttk.Label(bottom_frame, text="Ready", foreground='green')
        self.status_label.pack(side='right')

    # ----------------------------
    # Input Tab
    # ----------------------------
    def _build_input_tab(self, frame):
        desc = ("Input sources:\n"
                " - Synthetic generation (configurable classes/samples/target focus)\n"
                " - Folder: each subfolder is a class; CSV files are signals\n"
                " - .mat files: auto-detect or use keys (X, y)\n"
                f"Signals are forced to length {self.signal_length} (pad/truncate).")
        ttk.Label(frame, text=desc, wraplength=1000, justify='left').pack(anchor='w', padx=8, pady=6)

        # mode selection
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(anchor='w', padx=8, pady=4)
        self.input_mode = tk.StringVar(value='synthetic')
        for txt, val in [("Synthetic Data","synthetic"), ("Folder (CSV)","folder"), (".mat file","mat")]:
            rb = ttk.Radiobutton(mode_frame, text=txt, variable=self.input_mode, value=val)
            rb.pack(side='left', padx=10)
            self.ui_buttons.append(rb)

        # Synthetic options
        synth_frame = ttk.LabelFrame(frame, text="Synthetic options")
        synth_frame.pack(fill='x', padx=8, pady=6)
        ttk.Label(synth_frame, text="Number of classes:").grid(row=0, column=0, padx=6, pady=4, sticky='w')
        self.num_classes = tk.IntVar(value=2)
        ttk.Spinbox(synth_frame, from_=2, to=10, textvariable=self.num_classes, width=6).grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(synth_frame, text="Total samples:").grid(row=0, column=2, padx=6, pady=4, sticky='w')
        self.num_samples = tk.IntVar(value=200)
        ttk.Spinbox(synth_frame, from_=2, to=5000, increment=10, textvariable=self.num_samples, width=8).grid(row=0, column=3, padx=6, pady=4)

        ttk.Label(synth_frame, text="Target class (1..N):").grid(row=1, column=0, padx=6, pady=4, sticky='w')
        self.target_class = tk.IntVar(value=1)
        ttk.Spinbox(synth_frame, from_=1, to=10, textvariable=self.target_class, width=6).grid(row=1, column=1, padx=6, pady=4)

        ttk.Label(synth_frame, text="Focus strength (0..1):").grid(row=1, column=2, padx=6, pady=4, sticky='w')
        self.focus_multiplier = tk.DoubleVar(value=0.5)
        ttk.Scale(synth_frame, from_=0.0, to=1.0, orient='horizontal', variable=self.focus_multiplier, length=200).grid(row=1, column=3, padx=6, pady=4)

        # .mat keys
        mat_frame = ttk.LabelFrame(frame, text=".mat loader keys (optional)")
        mat_frame.pack(fill='x', padx=8, pady=6)
        ttk.Label(mat_frame, text="Signals key (X):").grid(row=0, column=0, padx=6, pady=4)
        self.mat_x_key = tk.StringVar(value='X')
        ttk.Entry(mat_frame, textvariable=self.mat_x_key, width=12).grid(row=0, column=1, padx=6, pady=4)
        ttk.Label(mat_frame, text="Labels key (y):").grid(row=0, column=2, padx=6, pady=4)
        self.mat_y_key = tk.StringVar(value='y')
        ttk.Entry(mat_frame, textvariable=self.mat_y_key, width=12).grid(row=0, column=3, padx=6, pady=4)

        # action buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', padx=8, pady=8)
        btn_load = ttk.Button(btn_frame, text="Run / Load", command=self._handle_input)
        btn_load.pack(side='left', padx=6)
        self.ui_buttons.append(btn_load)

        btn_clear = ttk.Button(btn_frame, text="Clear Data", command=self._clear_data)
        btn_clear.pack(side='left', padx=6)
        self.ui_buttons.append(btn_clear)

        btn_export = ttk.Button(btn_frame, text="Export Features CSV", command=self._export_features)
        btn_export.pack(side='left', padx=6)
        self.ui_buttons.append(btn_export)

        # status label
        self.input_status = ttk.Label(frame, text="", foreground='green')
        self.input_status.pack(anchor='w', padx=8, pady=6)

    # ----------------------------
    # Preview Tab (embedded matplotlib canvas)
    # ----------------------------
    def _build_preview_tab(self, frame):
        top = ttk.Frame(frame)
        top.pack(fill='x', padx=8, pady=4)
        ttk.Button(top, text="Show One Random Sample per Class (new figure)", command=self._show_preview).pack(side='left', padx=6)
        ttk.Button(top, text="Show 3 Samples per Class (new figures)", command=self._show_three_per_class).pack(side='left', padx=6)

        # Canvas area: single figure with navigation (prev/next)
        canvas_frame = ttk.LabelFrame(frame, text="Signal Browser (one plot at a time)")
        canvas_frame.pack(fill='both', expand=True, padx=8, pady=8)

        # matplotlib Figure
        self.preview_fig, self.preview_ax = plt.subplots(figsize=(9,4))
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=canvas_frame)
        self.preview_canvas.draw()
        self.preview_canvas.get_tk_widget().pack(fill='both', expand=True)

        nav_frame = ttk.Frame(frame)
        nav_frame.pack(pady=4)
        self.preview_index = 0
        self.preview_list = []  # list of (title, signal)
        ttk.Button(nav_frame, text="Prev", command=self._preview_prev).pack(side='left', padx=6)
        ttk.Button(nav_frame, text="Next", command=self._preview_next).pack(side='left', padx=6)
        ttk.Button(nav_frame, text="Refresh list", command=self._preview_refresh_list).pack(side='left', padx=6)
        ttk.Button(nav_frame, text="Clear preview list", command=self._clear_preview_list).pack(side='left', padx=6)

        self.preview_info = ttk.Label(frame, text="No preview loaded")
        self.preview_info.pack(anchor='w', padx=8, pady=4)

    def _preview_refresh_list(self):
        """Fill preview_list with one random sample per class (if data available)."""
        if self.data is None or self.labels is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        self.preview_list = []
        classes = np.unique(self.labels)
        for cls in classes:
            idxs = np.where(self.labels == cls)[0]
            if len(idxs) == 0:
                continue
            i = np.random.choice(idxs)
            self.preview_list.append((f"Class {int(cls)} Sample {i}", self.data[i]))
        self.preview_index = 0
        self._render_preview_item()
        self._log("Preview list refreshed")

    def _clear_preview_list(self):
        self.preview_list = []
        self.preview_index = 0
        self.preview_ax.clear()
        self.preview_canvas.draw()
        self.preview_info.config(text="Preview list cleared")
        self._log("Cleared preview list")

    def _preview_prev(self):
        if not self.preview_list:
            return
        self.preview_index = (self.preview_index - 1) % len(self.preview_list)
        self._render_preview_item()

    def _preview_next(self):
        if not self.preview_list:
            return
        self.preview_index = (self.preview_index + 1) % len(self.preview_list)
        self._render_preview_item()

    def _render_preview_item(self):
        if not self.preview_list:
            self.preview_ax.clear()
            self.preview_ax.text(0.5,0.5,"No preview items", ha='center', va='center')
            self.preview_canvas.draw()
            self.preview_info.config(text="No preview loaded")
            return
        title, sig = self.preview_list[self.preview_index]
        self.preview_ax.clear()
        self.preview_ax.plot(sig)
        self.preview_ax.set_title(title)
        self.preview_ax.set_xlabel("Index")
        self.preview_ax.set_ylabel("Amplitude")
        self.preview_canvas.draw()
        self.preview_info.config(text=f"{self.preview_index+1}/{len(self.preview_list)}: {title}")

    # simple show in separate matplotlib figures (keeps compatibility)
    def _show_preview(self):
        if self.data is None or self.labels is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        classes = np.unique(self.labels)
        plt.figure(figsize=(10,4))
        for cls in classes:
            idxs = np.where(self.labels == cls)[0]
            if len(idxs) == 0:
                continue
            plt.plot(self.data[np.random.choice(idxs)], alpha=0.9, label=f"Class {cls}")
        plt.legend()
        plt.title("One random sample per class")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
        self._log("Displayed preview (one per class)")

    def _show_three_per_class(self):
        if self.data is None or self.labels is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        classes = np.unique(self.labels)
        for cls in classes:
            idxs = np.where(self.labels == cls)[0]
            if len(idxs) == 0:
                continue
            sel = np.random.choice(idxs, size=min(3, len(idxs)), replace=False)
            plt.figure(figsize=(9,3))
            for i in sel:
                plt.plot(self.data[i], alpha=0.7)
            plt.title(f"Class {cls} - {len(sel)} samples")
            plt.tight_layout()
            plt.show()
        self._log("Displayed 3 samples per class")

    # ----------------------------
    # Features Tab (with boxplot browser and PCA)
    # ----------------------------
    def _build_features_tab(self, frame):
        ttk.Label(frame, text="Feature Extraction & Visualization", font=('Arial', 11)).pack(anchor='w', padx=8, pady=6)

        # feature presets and checkboxes
        preset_frame = ttk.Frame(frame)
        preset_frame.pack(anchor='w', padx=8)
        self.preset_var = tk.StringVar(value='basic')
        ttk.Radiobutton(preset_frame, text='Basic', variable=self.preset_var, value='basic', command=self._apply_preset).pack(side='left', padx=6)
        ttk.Radiobutton(preset_frame, text='Advanced', variable=self.preset_var, value='advanced', command=self._apply_preset).pack(side='left', padx=6)
        ttk.Radiobutton(preset_frame, text='Custom', variable=self.preset_var, value='custom', command=self._apply_preset).pack(side='left', padx=6)

        # time and freq features
        td_frame = ttk.LabelFrame(frame, text="Time-domain features")
        td_frame.pack(fill='x', padx=8, pady=6)
        self.feature_vars = {}
        td_list = ['mean','variance','skewness','kurtosis','rms','ptp']
        for feat in td_list:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(td_frame, text=feat, variable=var)
            cb.pack(side='left', padx=6, pady=4)
            self.feature_vars[feat] = var
            self.ui_buttons.append(cb)

        fd_frame = ttk.LabelFrame(frame, text="Frequency-domain features")
        fd_frame.pack(fill='x', padx=8, pady=6)
        fd_list = ['spec_centroid','dom_freq','spec_entropy','fft_power','band_power']
        for feat in fd_list:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(fd_frame, text=feat, variable=var)
            cb.pack(side='left', padx=6, pady=4)
            self.feature_vars[feat] = var
            self.ui_buttons.append(cb)

        # info + extract
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill='x', padx=8, pady=6)
        ttk.Button(info_frame, text="Feature Info", command=self._show_feature_info).pack(side='left', padx=6)
        self.extract_button = ttk.Button(info_frame, text="Extract & Show Boxplots + PCA", command=self._extract_and_plot_threaded)
        self.extract_button.pack(side='left', padx=6)
        self.ui_buttons.append(self.extract_button)

        # boxplot canvas area (single feature view)
        box_frame = ttk.LabelFrame(frame, text="Boxplot Browser (single feature at a time)")
        box_frame.pack(fill='both', expand=True, padx=8, pady=8)
        self.box_fig, self.box_ax = plt.subplots(figsize=(8,4))
        self.box_canvas = FigureCanvasTkAgg(self.box_fig, master=box_frame)
        self.box_canvas.draw()
        self.box_canvas.get_tk_widget().pack(fill='both', expand=True)

        box_nav = ttk.Frame(frame)
        box_nav.pack(pady=4)
        self.box_index = 0
        self.box_list = []  # feature names
        ttk.Button(box_nav, text="Prev feature", command=self._box_prev).pack(side='left', padx=6)
        ttk.Button(box_nav, text="Next feature", command=self._box_next).pack(side='left', padx=6)
        ttk.Button(box_nav, text="Refresh features", command=self._box_refresh_list).pack(side='left', padx=6)

        self.features_status = ttk.Label(frame, text="", foreground='green')
        self.features_status.pack(anchor='w', padx=8, pady=6)

    def _apply_preset(self):
        p = self.preset_var.get()
        basic = {'mean','variance','fft_power','dom_freq'}
        if p == 'basic':
            for k,var in self.feature_vars.items():
                var.set(k in basic)
            self._log("Preset applied: basic")
        elif p == 'advanced':
            for k,var in self.feature_vars.items():
                var.set(True)
            self._log("Preset applied: advanced")
        else:
            # custom random selection
            keys = list(self.feature_vars.keys())
            n = np.random.randint(1, len(keys)+1)
            chosen = list(np.random.choice(keys, size=n, replace=False))
            for k,var in self.feature_vars.items():
                var.set(k in chosen)
            self._log(f"Preset applied: custom random {chosen}")

    # feature compute helpers
    def _time_features(self, sig):
        sig = np.asarray(sig).astype(float)
        try:
            return {
                'mean': float(np.mean(sig)),
                'variance': float(np.var(sig)),
                'skewness': float(skew(sig)),
                'kurtosis': float(kurtosis(sig)),
                'rms': float(np.sqrt(np.mean(sig**2))),
                'ptp': float(np.ptp(sig))
            }
        except Exception:
            return {'mean':0.0,'variance':0.0,'skewness':0.0,'kurtosis':0.0,'rms':0.0,'ptp':0.0}

    def _freq_features(self, sig, fs):
        n = len(sig)
        if n == 0:
            return {'spec_centroid':0.0,'dom_freq':0.0,'spec_entropy':0.0,'fft_power':0.0,'band_power':0.0}
        fftv = np.abs(rfft(sig))
        P = fftv**2
        Psum = P.sum() if P.sum() > 0 else 1.0
        freqs = rfftfreq(n, d=1.0/fs)
        centroid = float((freqs * P).sum() / Psum) if freqs.size > 0 else 0.0
        dom_idx = int(np.argmax(P)) if P.size > 0 else 0
        dom_freq = float(freqs[dom_idx]) if freqs.size > dom_idx else 0.0
        spec = P / Psum
        entropy = float(-np.sum(spec * np.log2(spec + 1e-12)))
        fft_power = float(P.sum())
        low = float(self.band_low.get()) if hasattr(self, 'band_low') else 0.0
        high = float(self.band_high.get()) if hasattr(self, 'band_high') else freqs.max() if freqs.size>0 else 0.0
        if freqs.size>0 and (high > low):
            mask = (freqs >= low) & (freqs <= high)
            band_power = float(P[mask].sum()) if mask.sum()>0 else float(P.sum())
        else:
            band_power = float(P.sum())
        return {'spec_centroid':centroid,'dom_freq':dom_freq,'spec_entropy':entropy,'fft_power':fft_power,'band_power':band_power}

    # ----------------------------
    # Feature extraction threaded
    # ----------------------------
    def _extract_and_plot_threaded(self):
        if self._check_busy(): return
        if self.data is None or self.labels is None:
            messagebox.showerror("Error", "No data/labels loaded.")
            return
        t = threading.Thread(target=self._extract_and_plot_worker, daemon=True)
        t.start()

    def _extract_and_plot_worker(self):
        try:
            self._set_ui_busy(True)
            self._log("Feature extraction started (background)")
            # update fs
            try:
                self.fs = float(self.fs_var.get())
                if self.fs <= 0:
                    self.fs = DEFAULT_FS
            except Exception:
                self.fs = DEFAULT_FS

            sel = [k for k,v in self.feature_vars.items() if v.get()]
            if not sel:
                self.root.after(0, lambda: messagebox.showerror("Error", "Select at least one feature."))
                self._set_ui_busy(False)
                return

            sample_sig = self.data[0]
            time_keys = list(self._time_features(sample_sig).keys())
            freq_keys = list(self._freq_features(sample_sig, self.fs).keys())
            all_keys = time_keys + freq_keys
            self.all_feature_names = [k for k in all_keys if k in sel]

            rows = []
            for sig in self.data:
                td = self._time_features(sig)
                fd = self._freq_features(sig, self.fs)
                combined = {**td, **fd}
                rows.append([float(combined[k]) for k in self.all_feature_names])

            self.features_full = np.array(rows, dtype=float)
            self.features = self.features_full.copy()
            self.feature_names = list(self.all_feature_names)

            self._log(f"Feature extraction finished: {len(self.feature_names)} features")

            # schedule UI update to plot
            self.root.after(0, self._after_extract_plot)
        except Exception as e:
            self._log(f"Feature extraction failed: {e}\n{traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Feature extraction failed: {e}"))
            self._set_ui_busy(False)

    def _after_extract_plot(self):
        try:
            if self.features is None:
                messagebox.showerror("Error", "No features produced.")
                self._set_ui_busy(False)
                return

            # prepare box_list
            self.box_list = list(self.feature_names)
            self.box_index = 0
            if self.box_list:
                self._render_boxplot_item()
            else:
                self.box_ax.clear()
                self.box_ax.text(0.5,0.5,"No features selected", ha='center', va='center')
                self.box_canvas.draw()

            # PCA
            if self.features.shape[1] >= 2:
                try:
                    pca = PCA(n_components=2)
                    proj = pca.fit_transform(self.features)
                    plt.figure(figsize=(6,5))
                    classes = np.unique(self.labels)
                    for cls in classes:
                        idxs = np.where(self.labels==cls)[0]
                        plt.scatter(proj[idxs,0], proj[idxs,1], label=f"Class {int(cls)}", alpha=0.7)
                    plt.legend()
                    plt.title("PCA (2D) of selected features")
                    plt.xlabel("PC1"); plt.ylabel("PC2")
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    self._log(f"PCA failed: {e}")

            msg = f"Extracted {len(self.feature_names)} features."
            self.features_status.config(text=msg)
            self.status_label.config(text="Features extracted")
            self._log(msg)
        finally:
            self._set_ui_busy(False)

    # Boxplot navigation
    def _box_refresh_list(self):
        if not self.feature_names:
            messagebox.showinfo("Info", "No features extracted yet.")
            return
        self.box_list = list(self.feature_names)
        self.box_index = 0
        self._render_boxplot_item()
        self._log("Boxplot list refreshed")

    def _box_prev(self):
        if not self.box_list:
            return
        self.box_index = (self.box_index - 1) % len(self.box_list)
        self._render_boxplot_item()

    def _box_next(self):
        if not self.box_list:
            return
        self.box_index = (self.box_index + 1) % len(self.box_list)
        self._render_boxplot_item()

    def _render_boxplot_item(self):
        if not self.box_list or self.features is None:
            self.box_ax.clear()
            self.box_ax.text(0.5,0.5,"No feature boxplot to show", ha='center', va='center')
            self.box_canvas.draw()
            return
        fname = self.box_list[self.box_index]
        idx = self.feature_names.index(fname)
        classes = np.unique(self.labels)
        data_by_class = [self.features[self.labels==cls, idx] for cls in classes]
        self.box_ax.clear()
        self.box_ax.boxplot(data_by_class, labels=[f"Class {int(c)}" for c in classes])
        self.box_ax.set_title(f"Boxplot: {fname}")
        if fname in ['fft_power','band_power']:
            try:
                self.box_ax.set_yscale('log')
            except Exception:
                pass
        self.box_canvas.draw()

    # ----------------------------
    # Train & Evaluation Tab
    # ----------------------------
    def _build_train_tab(self, frame):
        ttk.Label(frame, text="Train & Evaluate Model", font=('Arial', 11)).pack(anchor='w', padx=8, pady=6)
        cfg = ttk.Frame(frame)
        cfg.pack(anchor='w', padx=8)

        ttk.Label(cfg, text="Test size (0-1):").grid(row=0, column=0, sticky='w', padx=6, pady=4)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Entry(cfg, textvariable=self.test_size_var, width=6).grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(cfg, text="SelectKBest k (0=off):").grid(row=0, column=2, sticky='w', padx=6, pady=4)
        self.k_best_var = tk.IntVar(value=0)
        ttk.Entry(cfg, textvariable=self.k_best_var, width=6).grid(row=0, column=3, padx=6, pady=4)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=8)
        self.train_button = ttk.Button(btn_frame, text="Train SVM (split & SelectKBest)", command=self._train_model_threaded)
        self.train_button.pack(side='left', padx=6)
        self.ui_buttons.append(self.train_button)

        self.inc_button = ttk.Button(btn_frame, text="Incremental feature eval", command=self._incremental_feature_eval_threaded)
        self.inc_button.pack(side='left', padx=6)
        self.ui_buttons.append(self.inc_button)

        self.show_conf_button = ttk.Button(btn_frame, text="Show Last Confusion", command=self._show_last_confusion)
        self.show_conf_button.pack(side='left', padx=6)
        self.ui_buttons.append(self.show_conf_button)

        self.train_status = ttk.Label(frame, text="", foreground='green')
        self.train_status.pack(anchor='w', padx=8, pady=6)

    def _train_model_threaded(self):
        if self._check_busy(): return
        if self.features_full is None or self.labels is None:
            messagebox.showerror("Error", "Extract features first and ensure labels exist.")
            return
        thread = threading.Thread(target=self._train_model_worker, daemon=True)
        thread.start()

    def _train_model_worker(self):
        try:
            self._set_ui_busy(True)
            self._log("Training started (background)")

            X_full = np.array(self.features_full, dtype=float)
            y = np.array(self.labels, dtype=int)
            if len(np.unique(y)) < 2:
                self.root.after(0, lambda: messagebox.showerror("Error", "Need at least 2 classes to train."))
                self._set_ui_busy(False)
                return

            try:
                test_size = float(self.test_size_var.get())
            except Exception:
                test_size = 0.2
                self.test_size_var.set(0.2)
            if not (0.0 < test_size < 0.9):
                test_size = 0.2
                self.test_size_var.set(0.2)

            X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=test_size, stratify=y, random_state=42)

            k = int(self.k_best_var.get())
            self.selector = None
            if k > 0:
                k = min(k, X_train_full.shape[1])
                if k < 1: k = 1
                selector = SelectKBest(score_func=f_classif, k=k)
                X_train = selector.fit_transform(X_train_full, y_train)
                X_test = selector.transform(X_test_full)
                keep_idx = selector.get_support(indices=True)
                self.feature_names = [self.all_feature_names[i] for i in keep_idx]
                try:
                    self.features = selector.transform(self.features_full)
                except Exception:
                    self.features = self.features_full[:, keep_idx].copy()
                self.selector = selector
                self._log(f"SelectKBest applied: kept top {k} features")
            else:
                X_train = X_train_full
                X_test = X_test_full
                self.feature_names = list(self.all_feature_names)
                self.features = self.features_full.copy()
                self.selector = None

            # scale
            self.scaler = StandardScaler()
            X_train_s = self.scaler.fit_transform(X_train)
            X_test_s = self.scaler.transform(X_test)

            # train SVM
            self.model = SVC(probability=True, kernel='rbf', random_state=42)
            self.model.fit(X_train_s, y_train)

            # evaluate
            y_pred = self.model.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            self._last_cm = cm
            self.test_X = X_test_s
            self.test_y = y_test

            self._log(f"Training finished. Test accuracy: {acc:.3f}")
            self.root.after(0, lambda: self._on_train_done(acc, report, cm))
        except Exception as e:
            self._log(f"Training failed: {e}\n{traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
        finally:
            self._set_ui_busy(False)

    def _on_train_done(self, acc, report, cm):
        try:
            plt.figure(figsize=(5,4))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self._log(f"Confusion plot error: {e}")
        try:
            self.train_status.config(text=f"Trained. Test accuracy: {acc:.3f}")
        except Exception:
            pass
        messagebox.showinfo("Train Report", f"Accuracy: {acc:.3f}\n\nReport:\n{report}")

    # ----------------------------
    # Incremental feature evaluation
    # ----------------------------
    def _incremental_feature_eval_threaded(self):
        if self._check_busy(): return
        if self.features_full is None or self.labels is None:
            messagebox.showerror("Error", "Extract features first.")
            return
        thread = threading.Thread(target=self._incremental_feature_eval_worker, daemon=True)
        thread.start()

    def _incremental_feature_eval_worker(self):
        try:
            self._set_ui_busy(True)
            self._log("Incremental evaluation started (background)")

            X = np.array(self.features_full, dtype=float)
            y = np.array(self.labels, dtype=int)
            if len(np.unique(y)) < 2:
                self.root.after(0, lambda: messagebox.showerror("Error", "Need at least 2 classes to evaluate."))
                self._set_ui_busy(False)
                return

            try:
                scores, pvals = f_classif(X, y)
            except Exception:
                scores = np.var(X, axis=0)
            order = np.argsort(-scores)
            ordered_names = [self.all_feature_names[i] for i in order]

            try:
                test_size = float(self.test_size_var.get())
            except Exception:
                test_size = 0.2
            if not (0.0 < test_size < 0.9):
                test_size = 0.2

            X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

            accs = []
            feat_counts = list(range(1, X.shape[1]+1))
            for k in feat_counts:
                idxs = order[:k]
                Xt = X_train_full[:, idxs]
                Xs = X_test_full[:, idxs]
                scaler = StandardScaler()
                try:
                    Xt_s = scaler.fit_transform(Xt)
                    Xs_s = scaler.transform(Xs)
                    clf = SVC(probability=False, kernel='rbf', random_state=42)
                    clf.fit(Xt_s, y_train)
                    y_pred = clf.predict(Xs_s)
                    accs.append(accuracy_score(y_test, y_pred))
                except Exception as e:
                    self._log(f"Incremental eval error at k={k}: {e}")
                    accs.append(0.0)

            self.root.after(0, lambda: self._after_incremental_eval(feat_counts, accs, ordered_names))
            self._log("Incremental evaluation finished")
        except Exception as e:
            self._log(f"Incremental feature eval failed: {e}\n{traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Error", "Incremental feature eval failed"))
        finally:
            self._set_ui_busy(False)

    def _after_incremental_eval(self, feat_counts, accs, ordered_names):
        try:
            plt.figure(figsize=(8,4))
            plt.plot(feat_counts, accs, marker='o')
            plt.xlabel('Number of top features (ANOVA-ranked)')
            plt.ylabel('Test accuracy')
            plt.title('Incremental feature addition evaluation')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            best_idx = int(np.argmax(accs))
            best_acc = accs[best_idx]
            best_k = feat_counts[best_idx]
            best_feats = ordered_names[:best_k]
            messagebox.showinfo("Incremental Eval", f"Best accuracy {best_acc:.3f} with {best_k} features:\n{', '.join(best_feats)}")
            self._log(f"Incremental eval: best acc={best_acc:.3f} with {best_k} features")
        except Exception as e:
            self._log(f"After incremental eval error: {e}")

    # ----------------------------
    # Test Tab (generate & classify)
    # ----------------------------
    def _build_test_tab(self, frame):
        ttk.Label(frame, text="Test signals & Model export", font=('Arial', 11)).pack(anchor='w', padx=8, pady=6)
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(anchor='w', padx=8, pady=4)
        ttk.Button(btn_frame, text="Generate 3 Test Signals", command=self._generate_test_signals).pack(side='left', padx=6)
        ttk.Button(btn_frame, text="Export model (pickle)", command=self._export_model).pack(side='left', padx=6)
        ttk.Button(btn_frame, text="Import model (pickle)", command=self._import_model).pack(side='left', padx=6)

        # test buttons area
        self.test_buttons_frame = ttk.Frame(frame)
        self.test_buttons_frame.pack(padx=8, pady=8)

        self.test_status = ttk.Label(frame, text="", foreground='blue')
        self.test_status.pack(anchor='w', padx=8, pady=4)

    def _generate_test_signals(self):
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Train model first.")
            return
        # clear old
        for w in list(self.test_buttons_frame.winfo_children()):
            w.destroy()
        self.test_signals = []
        length = self.signal_length
        n_classes = len(np.unique(self.labels)) if self.labels is not None else 2
        for i in range(3):
            t = np.linspace(0, 1, length)
            freq = np.random.uniform(0.5, 3 + 2*n_classes)
            amp = np.random.uniform(0.7, 1.3)
            noise_level = np.random.uniform(0.2, 0.7)
            mix = np.random.rand()
            sig_sin = np.sin(2*np.pi*freq*t)
            sig_square = np.sign(sig_sin)
            sig = amp * ((1-mix)*sig_sin + mix*sig_square) + noise_level * np.random.randn(length)
            self.test_signals.append(sig)
            btn = ttk.Button(self.test_buttons_frame, text=f"Test {i+1}", command=partial(self._show_test_signal, i))
            btn.pack(side='left', padx=6)
        self.test_status.config(text="3 test signals generated")
        self._log("Generated 3 test signals")

    def _show_test_signal(self, idx):
        if idx >= len(self.test_signals):
            return
        sig = self.test_signals[idx]
        plt.figure(figsize=(7,3))
        plt.plot(sig)
        plt.title(f"Test Signal {idx+1}")
        plt.tight_layout()
        plt.show()
        if not self.all_feature_names:
            messagebox.showerror("Error", "No feature definitions available. Extract features first.")
            self._log("Attempt to test without extracted features")
            return
        fs = float(self.fs_var.get()) if self.fs_var.get() else DEFAULT_FS
        feat_dict = {**self._time_features(sig), **self._freq_features(sig, fs)}
        try:
            full_feats = np.array([feat_dict[k] for k in self.all_feature_names], dtype=float).reshape(1, -1)
        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction mismatch: {e}")
            self._log(f"Feature extraction mismatch when testing: {e}")
            return
        X_test = full_feats
        if self.selector is not None:
            try:
                X_test = self.selector.transform(X_test)
            except Exception as e:
                messagebox.showerror("Error", f"Selector transform failed: {e}")
                self._log(f"Selector transform failed when testing: {e}")
                return
        else:
            if len(self.feature_names) != X_test.shape[1]:
                try:
                    idxs = [self.all_feature_names.index(fn) for fn in self.feature_names]
                    X_test = full_feats[:, idxs]
                except Exception as e:
                    self._log(f"Feature name mismatch when testing: {e}")
                    X_test = full_feats
        try:
            X_test_s = self.scaler.transform(X_test)
        except Exception as e:
            messagebox.showerror("Error", f"Scaler/model not compatible: {e}")
            self._log(f"Scaler transform failed when testing: {e}")
            return
        probs = self.model.predict_proba(X_test_s)[0]
        pred_class = int(np.argmax(probs))
        sorted_probs = np.sort(probs)[::-1]
        conf = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs)>1 else 0.0)
        p_true = 0.5 + conf/2
        p_true = min(max(p_true,0),1)
        other_classes = [c for c in range(len(probs)) if c!=pred_class]
        if other_classes:
            prob_others = (1-p_true)/len(other_classes)
            true_label_val = np.random.choice([pred_class] + other_classes, p=[p_true] + [prob_others]*len(other_classes))
        else:
            true_label_val = pred_class
        display_names = list(self.feature_names) if self.feature_names else list(self.all_feature_names)
        vals = X_test.flatten()
        info = f"Predicted: {pred_class} (Simulated True = {true_label_val})\nProbs: {np.round(probs,3)}\n\nFeatures:\n"
        info += "\n".join([f"{display_names[i]}: {float(vals[i]):.4f}" for i in range(len(vals))])
        messagebox.showinfo("Prediction", info)
        self._log(f"Tested signal {idx+1}: predicted {pred_class}, Simulated True = {true_label_val}")

    # ----------------------------
    # Export/import model & features
    # ----------------------------
    def _export_features(self):
        if self.features is None or not self.feature_names:
            messagebox.showerror("Error", "No features to export. Extract features first.")
            return
        if self.labels is None or len(self.labels) != len(self.features):
            if not messagebox.askyesno("Warning", "Labels length does not match features. Export without labels?"):
                return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files','*.csv')])
        if not path:
            return
        try:
            df = pd.DataFrame(self.features, columns=self.feature_names)
            if self.labels is not None and len(self.labels) == len(self.features):
                df['label'] = self.labels
            df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Features saved to {path}")
            self._log(f"Exported features to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save features: {e}")
            self._log(f"Failed to export features: {e}")

    def _export_model(self):
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Train model first.")
            return
        if not messagebox.askyesno("Security warning", "Saving model with pickle can be insecure. Continue?"):
            return
        path = filedialog.asksaveasfilename(defaultextension='.pkl', filetypes=[('Pickle','*.pkl')])
        if not path:
            return
        try:
            meta = {
                'saved_at': datetime.datetime.now().isoformat(),
                'all_feature_names': self.all_feature_names,
                'feature_names': self.feature_names,
                'fs': float(self.fs_var.get()) if hasattr(self, 'fs_var') else DEFAULT_FS
            }
            package = {'model': self.model, 'scaler': self.scaler, 'selector': self.selector, 'meta': meta}
            with open(path, 'wb') as f:
                pickle.dump(package, f)
            messagebox.showinfo("Saved", f"Model package saved to {path}")
            self._log(f"Exported model to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {e}")
            self._log(f"Failed to export model: {e}")

    def _import_model(self):
        path = filedialog.askopenfilename(filetypes=[('Pickle','*.pkl')])
        if not path:
            return
        try:
            with open(path, 'rb') as f:
                package = pickle.load(f)
            self.model = package.get('model', None)
            self.scaler = package.get('scaler', None)
            self.selector = package.get('selector', None)
            meta = package.get('meta', {})
            self.all_feature_names = meta.get('all_feature_names', self.all_feature_names)
            self.feature_names = meta.get('feature_names', self.feature_names)
            fs_val = meta.get('fs', DEFAULT_FS)
            try:
                self.fs_var.set(float(fs_val))
            except Exception:
                pass
            messagebox.showinfo("Loaded", f"Model imported from {os.path.basename(path)}")
            self._log(f"Imported model from {path}")
            self._update_button_states()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import model: {e}")
            self._log(f"Import model failed: {e}\n{traceback.format_exc()}")

    # ----------------------------
    # Help & Log Tab
    # ----------------------------
    def _build_help_tab(self, frame):
        left = ttk.Frame(frame)
        left.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        right = ttk.Frame(frame)
        right.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        help_text = ScrolledText(left, width=80, height=40, wrap='word')
        help_text.insert('1.0', self._help_content())
        help_text.configure(state='disabled')
        help_text.pack(fill='both', expand=True)

        ttk.Label(right, text="Activity Log (latest at bottom):").pack(anchor='w')
        self.log_widget = ScrolledText(right, width=55, height=32, wrap='word')
        self.log_widget.pack(fill='both', expand=True)
        log_btns = ttk.Frame(right)
        log_btns.pack(pady=6)
        ttk.Button(log_btns, text="Clear Log", command=self._clear_log).pack(side='left', padx=6)
        ttk.Button(log_btns, text="Save Log", command=self._save_log).pack(side='left', padx=6)

    def _help_content(self):
        return (
            "Signal Analyzer & Classifier - Help\\n\\n"
            "Workflow:\\n"
            "1) Input data: generate synthetic, load folder of CSVs (subfolders = classes), or load .mat file.\\n"
            "2) Preview: use the preview browser to inspect individual waveforms.\\n"
            "3) Features: choose presets and extract. Boxplots (single feature view) and PCA displayed.\\n"
            "4) Train & Eval: optionally select top-k features (ANOVA), then train SVM.\\n"
            "5) Incremental Eval: see how accuracy evolves as you add features.\\n"
            "6) Test: generate test signals and classify using trained model.\\n"
            "Export/Import: features CSV and model pickle supported.\\n"
            "Notes: set sampling rate (fs) to convert frequency-domain features to Hz.\\n"
        )

    # ----------------------------
    # Logging helpers
    # ----------------------------
    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] {msg}\n"
        try:
            if hasattr(self, 'log_widget'):
                self.log_widget.insert('end', full)
                self.log_widget.see('end')
        except Exception:
            pass
        print(full, end='')

    def _clear_log(self):
        try:
            self.log_widget.delete('1.0', 'end')
            self._log("Cleared log")
        except Exception:
            pass

    def _save_log(self):
        try:
            txt = self.log_widget.get('1.0', 'end')
            path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text','*.txt')])
            if not path: return
            with open(path, 'w', encoding='utf-8') as f:
                f.write(txt)
            messagebox.showinfo("Saved", f"Log saved to {path}")
            self._log(f"Saved log to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {e}")

    # ----------------------------
    # Load / Generate / Clear Data
    # ----------------------------
    def _clear_data(self):
        self.data = None
        self.labels = None
        self.features_full = None
        self.features = None
        self.feature_names = []
        self.all_feature_names = []
        self.model = None
        self.scaler = None
        self.selector = None
        self.test_signals = []
        self._log("Cleared all data, features and model")
        self.input_status.config(text="Cleared data")
        self._update_button_states()

    def _handle_input(self):
        mode = self.input_mode.get()
        if mode == 'synthetic':
            self._generate_synthetic()
        elif mode == 'folder':
            self._load_from_folder()
        elif mode == 'mat':
            self._load_mat_file()
        else:
            messagebox.showerror("Error", "Select an input mode")
        self._update_button_states()

    def _generate_synthetic(self):
        n_classes = int(self.num_classes.get())
        total = int(self.num_samples.get())
        target = int(self.target_class.get()) - 1
        if target < 0 or target >= n_classes:
            messagebox.showwarning("Warning", "Target class out of range: set to 1")
            target = 0
            self.target_class.set(1)

        length = self.signal_length
        focus_strength = float(self.focus_multiplier.get())
        target_count = int(round(total * focus_strength))
        remaining = total - target_count
        other_classes = [i for i in range(n_classes) if i != target]
        counts_others = np.zeros(len(other_classes), dtype=int)
        if remaining > 0 and len(other_classes) > 0:
            props = np.random.dirichlet(np.ones(len(other_classes))) * remaining
            counts_others = np.round(props).astype(int)
            diff = remaining - counts_others.sum()
            i = 0
            while diff != 0:
                counts_others[i % len(counts_others)] += np.sign(diff)
                diff = remaining - counts_others.sum()
                i += 1
        counts = np.zeros(n_classes, dtype=int)
        counts[target] = target_count
        for idx, cls in enumerate(other_classes):
            counts[cls] = counts_others[idx]

        signals, labels = [], []
        for cls_idx in range(n_classes):
            for _ in range(counts[cls_idx]):
                t = np.linspace(0,1,length)
                base_freq = 3 + cls_idx*4 + np.random.uniform(-1.0, 1.0)
                sig = 0
                n_harmonics = np.random.randint(1,4)
                for h in range(1, n_harmonics+1):
                    amp = np.random.uniform(0.5,1.0)
                    phase = np.random.uniform(0,2*np.pi)
                    sig += amp * np.sin(2*np.pi*base_freq*h*t + phase)
                sig += 0.2 * np.random.randn(length)
                if np.random.rand() < 0.3:
                    sig += 0.3 * np.sign(np.sin(2*np.pi*base_freq*t + np.random.rand()*2*np.pi))
                signals.append(sig)
                labels.append(cls_idx)

        if len(signals) == 0:
            messagebox.showerror("Error", "No synthetic samples generated.")
            return

        self.data, self.labels = shuffle(np.array(signals), np.array(labels), random_state=42)
        percentages = (counts / max(1, counts.sum())) * 100
        msg = f"Synthetic generated: {len(self.data)} signals, distribution: " + ", ".join([f"C{i}:{percentages[i]:.1f}%" for i in range(len(counts))])
        self.input_status.config(text=msg)
        self._log(msg)
        self._update_button_states()

    def _load_from_folder(self):
        folder = filedialog.askdirectory()
        if not folder: return
        class_dirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
        signals, labels = [], []
        for idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(folder, class_name)
            for file in sorted(os.listdir(class_path)):
                if file.lower().endswith('.csv'):
                    try:
                        arr = pd.read_csv(os.path.join(class_path, file), header=None).values.flatten()
                        if arr.size != self.signal_length:
                            if arr.size > self.signal_length:
                                arr = arr[:self.signal_length]
                            else:
                                arr = np.pad(arr, (0, self.signal_length-arr.size), 'constant')
                        signals.append(arr.astype(float))
                        labels.append(idx)
                    except Exception as e:
                        self._log(f"Failed to load {file}: {e}")
        if len(signals) == 0:
            messagebox.showerror("Error", "No CSV signals found in subfolders.")
            return
        self.data = np.array(signals)
        self.labels = np.array(labels)
        msg = f"Loaded {len(self.data)} signals from {len(class_dirs)} classes."
        self.input_status.config(text=msg)
        self._log(msg)
        self._update_button_states()

    def _load_mat_file(self):
        path = filedialog.askopenfilename(filetypes=[("MAT files","*.mat"), ("All files","*.*")])
        if not path: return
        try:
            mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load .mat: {e}")
            self._log(f"Failed to load .mat: {e}")
            return

        keys = [k for k in mat.keys() if not k.startswith("__")]
        x_key_user = self.mat_x_key.get().strip()
        y_key_user = self.mat_y_key.get().strip()

        # 1) user-provided key
        if x_key_user and x_key_user in mat:
            try:
                X_raw = np.array(mat[x_key_user])
                X2d = ensure_2d_signals(X_raw, target_len=self.signal_length)
                y_found = None
                if y_key_user and y_key_user in mat:
                    try:
                        y_cand = np.array(mat[y_key_user]).squeeze()
                        if y_cand.ndim == 1 and y_cand.size == X2d.shape[0]:
                            y_found = y_cand.astype(int).flatten()
                    except Exception:
                        pass
                if y_found is None:
                    for kk in keys:
                        if kk == x_key_user: continue
                        try:
                            cand = np.array(mat[kk]).squeeze()
                            if cand.ndim == 1 and cand.size == X2d.shape[0]:
                                y_found = cand.astype(int).flatten()
                                break
                        except Exception:
                            continue
                if y_found is None:
                    y_found = np.zeros(X2d.shape[0], dtype=int)
                    messagebox.showinfo("Info", "No label vector found in .mat; created dummy labels (zeros).")
                self.data = X2d.astype(float)
                self.labels = y_found
                msg = f"Loaded .mat using key '{x_key_user}' ({self.data.shape[0]} samples)."
                self.input_status.config(text=msg)
                self._log(msg)
                self._update_button_states()
                return
            except Exception as e:
                self._log(f"User-specified key attempt failed: {e}")

        # 2) detect X1, X2, ...
        x_keys = [k for k in keys if re.fullmatch(r"[Xx]\d+", k)]
        if not x_keys:
            x_keys = [k for k in keys if re.match(r"^[Xx].*\d+$", k)]
        if x_keys:
            try:
                def skey(s):
                    nums = re.findall(r'\d+', s)
                    return int(nums[0]) if nums else s
                x_sorted = sorted(x_keys, key=skey)
            except Exception:
                x_sorted = x_keys
            signals_list, labels_list = [], []
            for cls_idx, k in enumerate(x_sorted):
                try:
                    arr = np.array(mat[k])
                    arr2d = ensure_2d_signals(arr, target_len=self.signal_length)
                    signals_list.append(arr2d)
                    labels_list.append(np.full(arr2d.shape[0], cls_idx, dtype=int))
                except Exception as e:
                    self._log(f"Failed processing key {k}: {e}")
            if signals_list:
                X_all = np.vstack(signals_list)
                y_all = np.concatenate(labels_list)
                perm = np.random.permutation(X_all.shape[0])
                self.data = X_all[perm].astype(float)
                self.labels = y_all[perm]
                msg = f"Loaded {self.data.shape[0]} signals from keys: {', '.join(x_sorted)}"
                self.input_status.config(text=msg)
                self._log(msg)
                self._update_button_states()
                return

        # 3) single 2D matrix fallback
        for k in keys:
            try:
                arr = np.array(mat[k])
            except Exception:
                continue
            if arr.ndim == 2 and (arr.shape[0] == self.signal_length or arr.shape[1] == self.signal_length):
                try:
                    X2d = arr if arr.shape[1] == self.signal_length else arr.T
                    X2d = ensure_2d_signals(X2d, target_len=self.signal_length)
                    y_found = None
                    if y_key_user and y_key_user in mat:
                        try:
                            y_cand = np.array(mat[y_key_user]).squeeze()
                            if y_cand.ndim == 1 and y_cand.size == X2d.shape[0]:
                                y_found = y_cand.astype(int).flatten()
                        except Exception:
                            pass
                    if y_found is None:
                        for kk in keys:
                            if kk == k: continue
                            try:
                                arr2 = np.array(mat[kk]).squeeze()
                                if arr2.ndim == 1 and arr2.size == X2d.shape[0]:
                                    y_found = arr2.astype(int).flatten()
                                    break
                            except Exception:
                                continue
                    if y_found is None:
                        y_found = np.zeros(X2d.shape[0], dtype=int)
                        messagebox.showinfo("Info", f"No label vector found; created dummy labels for '{k}'.")
                    self.data = X2d.astype(float)
                    self.labels = y_found
                    msg = f"Loaded matrix '{k}' with {self.data.shape[0]} samples."
                    self.input_status.config(text=msg)
                    self._log(msg)
                    self._update_button_states()
                    return
                except Exception as e:
                    self._log(f"Warning processing key {k}: {e}")

        guesses = ", ".join(keys)
        messagebox.showinfo("Info", f"Could not automatically load signals. Available keys: {guesses}\nSet Signals key (X) manually and try again.")
        self._log(f"Could not auto-load .mat; keys: {guesses}")

    # ----------------------------
    # Busy state & UI helpers
    # ----------------------------
    def _set_ui_busy(self, busy: bool):
        with self._busy_lock:
            self._is_busy = busy
            try:
                if busy:
                    self.progress.start(10)
                    self.status_label.config(text="Busy...")
                else:
                    self.progress.stop()
                    self.status_label.config(text="Ready")
            except Exception:
                pass
            for w in self.ui_buttons:
                try:
                    w.configure(state='disabled' if busy else 'normal')
                except Exception:
                    pass

    def _check_busy(self) -> bool:
        with self._busy_lock:
            if self._is_busy:
                messagebox.showinfo("Please wait", "Another operation is running. Please wait until it finishes.")
                return True
            return False

    def _update_button_states(self):
        has_data = (self.data is not None and self.labels is not None)
        try:
            self.extract_button.configure(state='normal' if has_data else 'disabled')
        except Exception:
            pass
        try:
            self.train_button.configure(state='normal' if self.features_full is not None else 'disabled')
        except Exception:
            pass

    # ----------------------------
    # Feature info dialog
    # ----------------------------
    def _show_feature_info(self):
        info = ("Feature explanations:\n\n"
                "Time-domain:\n"
                " - mean: average value\n"
                " - variance: spread\n"
                " - skewness: asymmetry\n"
                " - kurtosis: tailedness\n"
                " - rms: root-mean-square\n"
                " - ptp: peak-to-peak\n\n"
                "Frequency-domain:\n"
                " - spec_centroid: spectral centroid (Hz)\n"
                " - dom_freq: dominant frequency (Hz)\n"
                " - spec_entropy: spectral entropy (flatness)\n"
                " - fft_power: total spectral power\n"
                " - band_power: power in [band_low, band_high]\n")
        messagebox.showinfo("Feature Info", info)

# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalAnalyzerApp(root)
    root.mainloop()
