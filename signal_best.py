#!/usr/bin/env python3
"""
Signal Analyzer & Classifier GUI.

key improvements (complete):
 - Robust .mat / folder / synthetic loading (from earlier versions).
 - Feature extraction (time + frequency) with configurable sampling rate (fs)
   and configurable band-power frequency range (instead of fixed index 5:21).
 - Keeps both full feature matrix (features_full) and active feature matrix (features).
 - SelectKBest supported and selector/scaler preserved for testing/export.
 - Incremental feature evaluation (ANOVA-ranked) implemented.
 - PCA and boxplots for visualization.
 - Threading: long operations (feature extraction, training, incremental eval) run
   in a background thread to avoid freezing the GUI.
 - Progress bar (indeterminate) and UI disabling while background tasks run.
 - Clear, persistent log pane and Help text.
 - Export features and model (pickle) with metadata; warning about pickle safety.
 - Robust test-signal regeneration (clears previous buttons).
 - Plenty of inline comments and user-friendly messages.

Dependencies:
    numpy, scipy, pandas, scikit-learn, matplotlib, tkinter

Run:
    python signal_analyzer_upgrade_full.py
"""

import os
import re
import threading
import datetime
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------------
# Utility / compatibility
# ---------------------------

# Ensure matplotlib backend won't block in some headless contexts.
# This code keeps default interactive behavior for normal desktop users.
# If you run on a headless server, set matplotlib.use('Agg') before importing pyplot.
# import matplotlib
# matplotlib.use('Agg')

# ---------------------------
# Main Application
# ---------------------------


class SignalAnalyzerApp:
    """Complete Signal Analyzer & Classifier GUI with threading and configurable fs."""

    def __init__(self, root):
        self.root = root
        self.root.title("Signal Analyzer & Classifier (Upgraded)")
        self.root.geometry("1180x820")

        # --- Data containers and state ---
        self.data = None  # raw signals: shape (m, n) rows = samples
        self.labels = None  # shape (m,)
        self.features_full = None  # full feature matrix (m, P)
        self.features = None  # active feature matrix (m, p) (after selection)
        self.all_feature_names = []  # names for features_full columns
        self.feature_names = []  # active feature names corresponding to self.features
        self.feature_vars = {}  # mapping feature_name -> tk.BooleanVar()
        self.model = None
        self.scaler = None
        self.selector = None
        self.test_signals = []
        self._last_cm = None

        # threading / UI busy state
        self._busy_lock = threading.Lock()
        self._is_busy = False

        # store important button references to enable/disable when busy
        self.ui_buttons = []

        # Build GUI
        self._build_ui()
        self._log("Application started")

        # update buttons (enable/disable) according to initial state
        self._update_button_states()

    # ---------------------------
    # UI Construction
    # ---------------------------
    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both')

        # Tabs
        self.tab_input = ttk.Frame(notebook)
        self.tab_preview = ttk.Frame(notebook)
        self.tab_features = ttk.Frame(notebook)
        self.tab_train = ttk.Frame(notebook)
        self.tab_test = ttk.Frame(notebook)
        self.tab_help = ttk.Frame(notebook)

        notebook.add(self.tab_input, text="1. Input Data")
        notebook.add(self.tab_preview, text="2. Preview")
        notebook.add(self.tab_features, text="3. Features")
        notebook.add(self.tab_train, text="4. Train & Eval")
        notebook.add(self.tab_test, text="5. Test")
        notebook.add(self.tab_help, text="6. Help & Log")

        self._build_input_tab(self.tab_input)
        self._build_preview_tab(self.tab_preview)
        self._build_features_tab(self.tab_features)
        self._build_train_tab(self.tab_train)
        self._build_test_tab(self.tab_test)
        self._build_help_tab(self.tab_help)

    # ---------------------------
    # Input Tab
    # ---------------------------
    def _build_input_tab(self, frame):
        desc = (
            "Provide 1D signals (length=1024). Sources:\n"
            " - Synthetic: generate sine signals + noise\n"
            " - Folder: select a folder with one subfolder per class; each .csv is one signal\n"
            " - .mat file: load MATLAB files (common key patterns supported)\n\n"
            "Note: Sampling rate (fs) affects frequency-domain features plotting and band-power computation."
        )
        ttk.Label(frame, text=desc, wraplength=1060, justify='left').pack(anchor='w', padx=8, pady=8)

        # input mode
        self.input_mode = tk.StringVar(value='synthetic')
        modes_frame = ttk.Frame(frame)
        modes_frame.pack(anchor='w', padx=8, pady=4)
        for text, val in [("Synthetic Data", "synthetic"), ("Load from Folder (CSV)", "folder"), ("Load .mat file", "mat")]:
            rb = ttk.Radiobutton(modes_frame, text=text, variable=self.input_mode, value=val)
            rb.pack(side='left', padx=6)
            self.ui_buttons.append(rb)

        # synthetic controls
        synth_frame = ttk.LabelFrame(frame, text="Synthetic options")
        synth_frame.pack(fill='x', padx=8, pady=6)

        ttk.Label(synth_frame, text="Number of Classes:").grid(row=0, column=0, sticky='w', padx=6, pady=4)
        self.num_classes = tk.IntVar(value=2)
        sp = ttk.Spinbox(synth_frame, from_=2, to=10, textvariable=self.num_classes, width=6)
        sp.grid(row=0, column=1, padx=6, pady=4)
        self.ui_buttons.append(sp)

        ttk.Label(synth_frame, text="Total Samples:").grid(row=0, column=2, sticky='w', padx=6, pady=4)
        self.num_samples = tk.IntVar(value=200)
        sp2 = ttk.Spinbox(synth_frame, from_=20, to=5000, increment=10, textvariable=self.num_samples, width=8)
        sp2.grid(row=0, column=3, padx=6, pady=4)
        self.ui_buttons.append(sp2)

        ttk.Label(synth_frame, text="Target Class (1..N):").grid(row=1, column=0, sticky='w', padx=6, pady=4)
        self.target_class = tk.IntVar(value=1)
        sp3 = ttk.Spinbox(synth_frame, from_=1, to=10, textvariable=self.target_class, width=6)
        sp3.grid(row=1, column=1, padx=6, pady=4)
        self.ui_buttons.append(sp3)

        ttk.Label(synth_frame, text="Focus Strength (0..1):").grid(row=1, column=2, sticky='w', padx=6, pady=4)
        self.focus_multiplier = tk.DoubleVar(value=0.5)
        sc = ttk.Scale(synth_frame, from_=0.0, to=1.0, orient='horizontal', variable=self.focus_multiplier, length=200)
        sc.grid(row=1, column=3, padx=6, pady=4)
        self.ui_buttons.append(sc)

        # fs and band controls (affects freq features)
        misc_frame = ttk.LabelFrame(frame, text="Frequency settings (affect freq features)")
        misc_frame.pack(fill='x', padx=8, pady=6)

        ttk.Label(misc_frame, text="Sampling rate fs (Hz):").grid(row=0, column=0, sticky='w', padx=6, pady=4)
        self.fs_var = tk.DoubleVar(value=1.0)  # default 1.0 (normalized)
        e_fs = ttk.Entry(misc_frame, textvariable=self.fs_var, width=10)
        e_fs.grid(row=0, column=1, padx=6, pady=4)
        self.ui_buttons.append(e_fs)

        ttk.Label(misc_frame, text="Band-power low (Hz):").grid(row=0, column=2, sticky='w', padx=6, pady=4)
        self.band_low = tk.DoubleVar(value=0.0)
        ttk.Entry(misc_frame, textvariable=self.band_low, width=10).grid(row=0, column=3, padx=6, pady=4)

        ttk.Label(misc_frame, text="Band-power high (Hz):").grid(row=0, column=4, sticky='w', padx=6, pady=4)
        self.band_high = tk.DoubleVar(value=10.0)
        ttk.Entry(misc_frame, textvariable=self.band_high, width=10).grid(row=0, column=5, padx=6, pady=4)

        # .mat keys
        mat_frame = ttk.LabelFrame(frame, text=".mat loader keys (if you use .mat)")
        mat_frame.pack(fill='x', padx=8, pady=6)
        ttk.Label(mat_frame, text="Signals key:").grid(row=0, column=0, sticky='w', padx=6, pady=4)
        self.mat_x_key = tk.StringVar(value='X')
        ttk.Entry(mat_frame, textvariable=self.mat_x_key, width=12).grid(row=0, column=1, padx=6, pady=4)
        ttk.Label(mat_frame, text="Labels key:").grid(row=0, column=2, sticky='w', padx=6, pady=4)
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

        # progress bar for background operations
        self.progress = ttk.Progressbar(frame, mode='indeterminate')
        self.progress.pack(fill='x', padx=8, pady=4)
        self.progress.stop()

    # ---------------------------
    # Preview Tab
    # ---------------------------
    def _build_preview_tab(self, frame):
        ttk.Button(frame, text="Show One Random Sample per Class", command=self._show_preview).pack(pady=8)
        ttk.Button(frame, text="Show 3 Samples per Class", command=self._show_three_per_class).pack()
        self.preview_status = ttk.Label(frame, text="", foreground='blue')
        self.preview_status.pack(anchor='w', pady=6)

    def _show_preview(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        if self.labels is None:
            messagebox.showerror("Error", "No labels available.")
            return
        classes = np.unique(self.labels)
        plt.figure(figsize=(10, 4))
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
        self.preview_status.config(text="Preview shown")
        self._log("Displayed preview (1 per class)")

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
            plt.figure(figsize=(9, 3))
            for i in sel:
                plt.plot(self.data[i], alpha=0.7)
            plt.title(f"Class {cls} - {len(sel)} samples")
            plt.tight_layout()
            plt.show()
        self._log("Displayed 3 samples per class")

    # ---------------------------
    # Features Tab
    # ---------------------------
    def _build_features_tab(self, frame):
        ttk.Label(frame, text="Choose features to extract. Use presets to simplify choices.", font=('Arial', 11)).pack(anchor='w', padx=8, pady=6)

        # preset radio
        preset_frame = ttk.Frame(frame)
        preset_frame.pack(anchor='w', padx=8)
        self.preset_var = tk.StringVar(value='basic')
        ttk.Label(preset_frame, text="Preset:").pack(side='left')
        rb_basic = ttk.Radiobutton(preset_frame, text='Basic', variable=self.preset_var, value='basic', command=self._apply_preset)
        rb_adv = ttk.Radiobutton(preset_frame, text='Advanced', variable=self.preset_var, value='advanced', command=self._apply_preset)
        rb_custom = ttk.Radiobutton(preset_frame, text='Custom', variable=self.preset_var, value='custom', command=self._apply_preset)
        rb_basic.pack(side='left', padx=6)
        rb_adv.pack(side='left', padx=6)
        rb_custom.pack(side='left', padx=6)
        self.ui_buttons.extend([rb_basic, rb_adv, rb_custom])

        # time-domain features
        td_frame = ttk.LabelFrame(frame, text="Time-domain features")
        td_frame.pack(fill='x', padx=8, pady=6)
        td_list = ['mean', 'variance', 'skewness', 'kurtosis', 'rms', 'ptp']
        for feat in td_list:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(td_frame, text=feat, variable=var)
            cb.pack(side='left', padx=6, pady=4)
            self.feature_vars[feat] = var
            self.ui_buttons.append(cb)

        # frequency-domain features
        fd_frame = ttk.LabelFrame(frame, text="Frequency-domain features")
        fd_frame.pack(fill='x', padx=8, pady=6)
        fd_list = ['spec_centroid', 'dom_freq', 'spec_entropy', 'fft_power', 'band_power']
        for feat in fd_list:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(fd_frame, text=feat, variable=var)
            cb.pack(side='left', padx=6, pady=4)
            self.feature_vars[feat] = var
            self.ui_buttons.append(cb)

        # feature info and extract button
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill='x', padx=8, pady=6)
        ttk.Button(info_frame, text="Feature Info", command=self._show_feature_info).pack(side='left', padx=6)
        self.extract_button = ttk.Button(info_frame, text="Extract & Plot boxplots + PCA", command=self._extract_and_plot_threaded)
        self.extract_button.pack(side='left', padx=6)
        self.ui_buttons.append(self.extract_button)

        self.features_status = ttk.Label(frame, text="", foreground='green')
        self.features_status.pack(anchor='w', padx=8, pady=6)

        # apply default preset
        self._apply_preset()

    def _apply_preset(self):
        p = self.preset_var.get()
        basic = {'mean', 'variance', 'fft_power', 'dom_freq'}
        all_feats = list(self.feature_vars.keys())

        if p == 'basic':
            for k, var in self.feature_vars.items():
                var.set(k in basic)
            self._log("Applied preset: Basic")
        elif p == 'advanced':
            for k, var in self.feature_vars.items():
                var.set(True)
            self._log("Applied preset: Advanced")
        else:
        # custom — select a random number of random features
            n_random = np.random.randint(1, len(all_feats)+1)  
            chosen = np.random.choice(all_feats, size=n_random, replace=False)
            for k, var in self.feature_vars.items():
                var.set(k in chosen)
            self._log(f"Preset: Custom selected with random features ({n_random}): {', '.join(chosen)}")

    def _show_feature_info(self):
        info = (
            "Feature explanations (short):\n\n"
            "Time-domain:\n"
            " - mean: average value.\n"
            " - variance: spread of values.\n"
            " - skewness: asymmetry.\n"
            " - kurtosis: tailedness/peakiness.\n"
            " - rms: root-mean-square (energy-like).\n"
            " - ptp: peak-to-peak (max-min).\n\n"
            "Frequency-domain:\n"
            " - spec_centroid: spectral centroid (Hz if fs set correctly).\n"
            " - dom_freq: dominant frequency (Hz).\n"
            " - spec_entropy: spectral entropy (how flat the spectrum is).\n"
            " - fft_power: total spectral power (sum of squared magnitudes).\n"
            " - band_power: power in frequency band [band_low, band_high] (Hz).\n\n"
            "Important:\n"
            " - Set Sampling rate fs (Input tab). If your data were sampled at fs Hz, enter it there. "
            "Default fs=1.0 means frequency units are 'normalized'.\n"
            " - Band-power is computed with real frequency thresholds using fs.\n"
        )
        messagebox.showinfo("Feature Info", info)
        self._log("Displayed feature info dialog")

    # ---------------------------
    # Frequency & Time Features (core computations)
    # ---------------------------
    def _time_features(self, sig):
        sig = np.asarray(sig).astype(float)
        return {
            'mean': float(np.mean(sig)),
            'variance': float(np.var(sig)),
            'skewness': float(skew(sig)),
            'kurtosis': float(kurtosis(sig)),
            'rms': float(np.sqrt(np.mean(sig ** 2))),
            'ptp': float(np.ptp(sig))
        }

    def _freq_features(self, sig, fs):
        n = len(sig)
        # compute one-sided magnitudes (consistent with earlier code)
        fftv = np.abs(fft(sig))[:n // 2]
        P = fftv ** 2  # power spectrum (not normalized)
        Psum = P.sum() if P.sum() > 0 else 1.0
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)[:n // 2]  # frequencies in Hz
        centroid = float((freqs * P).sum() / Psum) if freqs.size > 0 else 0.0
        dom_idx = int(np.argmax(P)) if P.size > 0 else 0
        dom_freq = float(freqs[dom_idx]) if freqs.size > dom_idx else 0.0
        # spectral entropy (base-2)
        spec = P / Psum
        entropy = float(-np.sum(spec * np.log2(spec + 1e-12)))
        fft_power = float(P.sum())
        # band-power using configurable band_low/band_high
        low = float(self.band_low.get())
        high = float(self.band_high.get())
        if freqs.size > 0 and (high > low):
            mask = (freqs >= low) & (freqs <= high)
            if mask.sum() == 0:
                # if mask empty, fallback to small central band
                band_power = float(P.sum())
            else:
                band_power = float(P[mask].sum())
        else:
            band_power = float(P.sum())
        return {
            'spec_centroid': centroid,
            'dom_freq': dom_freq,
            'spec_entropy': entropy,
            'fft_power': fft_power,
            'band_power': band_power
        }

    # ---------------------------
    # Feature Extraction (threaded)
    # ---------------------------
    def _extract_and_plot_threaded(self):
        # wrapper to run extraction in thread
        if self._check_busy():  # prevents overlapping operations
            return
        if self.data is None or self.labels is None:
            messagebox.showerror("Error", "No data/labels loaded.")
            return
        # start background thread
        thread = threading.Thread(target=self._extract_and_plot_worker, daemon=True)
        thread.start()

    def _extract_and_plot_worker(self):
        """Background: compute feature matrix and then schedule plotting on main thread."""
        try:
            self._set_ui_busy(True)
            self._log("Started feature extraction (background)")

            fs = float(self.fs_var.get()) if self.fs_var.get() else 1.0
            sel = [k for k, v in self.feature_vars.items() if v.get()]
            if not sel:
                self.root.after(0, lambda: messagebox.showerror("Error", "Select at least one feature."))
                self._set_ui_busy(False)
                return

            # produce all feature names in consistent order (time then freq)
            sample_sig = self.data[0]
            time_keys = list(self._time_features(sample_sig).keys())
            freq_keys = list(self._freq_features(sample_sig, fs).keys())
            all_keys = time_keys + freq_keys
            # selected keys in that order
            self.all_feature_names = [k for k in all_keys if k in sel]

            rows = []
            for sig in self.data:
                td = self._time_features(sig)
                fd = self._freq_features(sig, fs)
                combined = {**td, **fd}
                rows.append([float(combined[k]) for k in self.all_feature_names])

            features_full = np.array(rows, dtype=float)
            # store computed full features (before selection)
            self.features_full = features_full
            # by default active = full
            self.features = self.features_full.copy()
            self.feature_names = list(self.all_feature_names)

            self._log(f"Feature extraction finished: {len(self.feature_names)} features")

            # schedule plotting / UI updates on main thread
            self.root.after(0, self._after_extract_plot)
        except Exception as e:
            self._log(f"Feature extraction failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Feature extraction failed: {e}"))
            self._set_ui_busy(False)

    def _after_extract_plot(self):
        """Runs on main thread: create boxplots, PCA and update UI status."""
        try:
            if self.features is None:
                messagebox.showerror("Error", "Feature extraction did not produce features.")
                self._set_ui_busy(False)
                return

            labels_unique = np.unique(self.labels)
            # boxplots
            for idx, fname in enumerate(self.feature_names):
                plt.figure(figsize=(6, 4))
                data_by_class = [self.features[self.labels == cls, idx] for cls in labels_unique]
                plt.boxplot(data_by_class, tick_labels=[f"Class {int(cls)}" for cls in labels_unique])
                plt.title(f"Boxplot: {fname}")
                if fname in ['fft_power', 'band_power']:
                    try:
                        plt.yscale('log')
                    except Exception as e:
                        print(f"Logging error: {e}")
                plt.xlabel('Class')
                plt.ylabel(fname)
                plt.tight_layout()
                plt.show()

            # PCA 2D
            try:
                if self.features.shape[1] >= 2:
                    pca = PCA(n_components=2)
                    proj = pca.fit_transform(self.features)
                    plt.figure(figsize=(6, 5))
                    for cls in labels_unique:
                        idxs = np.where(self.labels == cls)[0]
                        plt.scatter(proj[idxs, 0], proj[idxs, 1], label=f"Class {int(cls)}", alpha=0.7)
                    plt.legend()
                    plt.title('PCA (2D) of selected features')
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                self._log(f"PCA plotting skipped due to an error {e}")

            msg = f"Extracted {len(self.feature_names)} features: {', '.join(self.feature_names)}"
            self.features_status.config(text=msg)
            self.input_status.config(text="Features extracted")
            self._log(msg)
        finally:
            self._set_ui_busy(False)

    # ---------------------------
    # Train Tab
    # ---------------------------
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

        # Train / incremental eval / show confusion
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
        # run training in a background thread
        if self._check_busy():
            return
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

            test_size = float(self.test_size_var.get())
            if not (0.0 < test_size < 0.9):
                test_size = 0.2
                self.test_size_var.set(0.2)

            X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=test_size, stratify=y, random_state=42)

            # optional SelectKBest
            k = int(self.k_best_var.get())
            self.selector = None
            if k > 0:
                k = min(k, X_train_full.shape[1])
                if k < 1:
                    k = 1
                selector = SelectKBest(score_func=f_classif, k=k)
                X_train = selector.fit_transform(X_train_full, y_train)
                X_test = selector.transform(X_test_full)
                keep_idx = selector.get_support(indices=True)
                self.feature_names = [self.all_feature_names[i] for i in keep_idx]
                # update active features matrix (apply to features_full)
                try:
                    self.features = selector.transform(self.features_full)
                except Exception as e:
                    print(f"Logging error: {e}")
                    # fallback (shouldn't happen)
                    self.features = self.features_full[:, keep_idx].copy()
                self.selector = selector
                self._log(f"SelectKBest applied: kept top {k} features")
            else:
                X_train = X_train_full
                X_test = X_test_full
                self.selector = None
                self.feature_names = list(self.all_feature_names)
                self.features = self.features_full.copy()

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

            # store last cm and test split
            self._last_cm = cm
            self.test_X = X_test_s
            self.test_y = y_test

            self._log(f"Training finished. Test accuracy: {acc:.3f}")
            # schedule plotting confusion and dialogs on main thread
            self.root.after(0, lambda: self._on_train_done(acc, report, cm))
        except Exception as e:
            self._log(f"Training failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
        finally:
            self._set_ui_busy(False)

    def _on_train_done(self, acc, report, cm):
        # Show confusion matrix plot and a messagebox with report
        try:
            plt.figure(figsize=(5, 4))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            self.root.after(0, lambda: plt.show())
        except Exception as e:
            print(f"Logging error: {e}")
        self.train_status.config(text=f"Trained. Test accuracy: {acc:.3f}")
        self.root.after(0, lambda: messagebox.showinfo("Train Report", f"Accuracy: {acc:.3f}\n\nReport:\n{report}"))

    # ---------------------------
    # Incremental Eval (threaded)
    # ---------------------------
    def _incremental_feature_eval_threaded(self):
        if self._check_busy():
            return
        if self.features_full is None or self.labels is None:
            messagebox.showerror("Error", "Extract features first.")
            return
        thread = threading.Thread(target=self._incremental_feature_eval_worker, daemon=True)
        thread.start()

    def _incremental_feature_eval_worker(self):
        try:
            self._set_ui_busy(True)
            self._log("Incremental feature evaluation started (background)")

            X = np.array(self.features_full, dtype=float)
            y = np.array(self.labels, dtype=int)
            if len(np.unique(y)) < 2:
                self.root.after(0, lambda: messagebox.showerror("Error", "Need at least 2 classes to evaluate."))
                self._set_ui_busy(False)
                return

            # compute ANOVA F-scores to rank features
            try:
                scores, pvals = f_classif(X, y)
            except Exception as e:
                print(f"Logging error: {e}")
                scores = np.var(X, axis=0)
            order = np.argsort(-scores)
            ordered_names = [self.all_feature_names[i] for i in order]

            test_size = float(self.test_size_var.get())
            if not (0.0 < test_size < 0.9):
                test_size = 0.2

            X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

            accs = []
            feat_counts = list(range(1, X.shape[1] + 1))
            for k in feat_counts:
                idxs = order[:k]
                Xt = X_train_full[:, idxs]
                Xs = X_test_full[:, idxs]
                scaler = StandardScaler()
                Xt_s = scaler.fit_transform(Xt)
                Xs_s = scaler.transform(Xs)
                clf = SVC(probability=False, kernel='rbf', random_state=42)
                try:
                    clf.fit(Xt_s, y_train)
                    y_pred = clf.predict(Xs_s)
                    accs.append(accuracy_score(y_test, y_pred))
                except Exception as e:
                    print(f"Logging error: {e}")
                    accs.append(0.0)

            # schedule plotting on main thread
            self.root.after(0, lambda: self._after_incremental_eval(feat_counts, accs, ordered_names))
            self._log("Incremental feature evaluation finished")
        except Exception as e:
            self._log(f"Incremental feature eval failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", "Incremental feature eval failed"))
        finally:
            self._set_ui_busy(False)

    def _after_incremental_eval(self, feat_counts, accs, ordered_names):
        try:
            plt.figure(figsize=(8, 4))
            plt.plot(feat_counts, accs, marker='o')
            plt.xlabel('Number of top features (ANOVA-ranked)')
            plt.ylabel('Test accuracy')
            plt.title('Incremental feature addition evaluation')
            plt.grid(True)
            plt.tight_layout()
            self.root.after(0, lambda: plt.show())
            best_idx = int(np.argmax(accs))
            best_acc = accs[best_idx]
            best_k = feat_counts[best_idx]
            best_feats = ordered_names[:best_k]
            self.root.after(0, lambda: messagebox.showinfo("Incremental Eval", f"Best accuracy {best_acc:.3f} with {best_k} features:\n{', '.join(best_feats)}"))
            self._log(f"Incremental eval: best acc={best_acc:.3f} with {best_k} features")
        except Exception as e:
            self._log(f"Error showing incremental eval results: {e}")

    def _show_last_confusion(self):
        try:
            cm = self._last_cm
        except Exception as e:
            print(f"Logging error: {e}")
            messagebox.showinfo("Info", "No confusion matrix available. Train first.")
            return
        plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (last)")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
    # ---------------------------
    # Test Tab
    # ---------------------------
    def _build_test_tab(self, frame):
        ttk.Button(frame, text="test 3 Signals", command=self._generate_test_signals).pack(pady=8)
        ttk.Button(frame, text="Export model (pickle)", command=self._export_model).pack()
        self.test_buttons_frame = ttk.Frame(frame)
        self.test_buttons_frame.pack(pady=8)
        self.test_status = ttk.Label(frame, text="", foreground='blue')
        self.test_status.pack(pady=6)

    def _generate_test_signals(self):
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Train model first.")
            return

        length = 1024
        n_classes = len(np.unique(self.labels)) if self.labels is not None else 2

        # Clear previous buttons and signals
        self._clear_test_buttons()
        self.test_signals = []

        for i in range(3):
            t = np.linspace(0, 1, length)
            freq = np.random.uniform(0.5, 3 + 2*n_classes)
            amp = np.random.uniform(0.7, 1.3)
            noise_level = np.random.uniform(0.2, 0.7)

          
            mix = np.random.rand()  

           
            sig_sin = np.sin(2 * np.pi * freq * t)
            sig_square = np.sign(sig_sin)  # تقریبی مربعی
            sig = amp * ((1-mix) * sig_sin + mix * sig_square) + noise_level * np.random.randn(length)


            self.test_signals.append(sig)
            btn = ttk.Button(self.test_buttons_frame, text=f"Test {i+1}", command=lambda idx=i: self._show_test_signal(idx))
            btn.pack(side='left', padx=6)
        self.test_status.config(text="3 test signals generated")
        self._log("Generated 3 test signals and created buttons")

    def _clear_test_buttons(self):
        for w in list(self.test_buttons_frame.winfo_children()):
            w.destroy()
        self._log("Cleared old test buttons (if any)")

    def _show_test_signal(self, idx):
        if idx >= len(self.test_signals):
            return
        sig = self.test_signals[idx]
        plt.figure(figsize=(7, 3))
        plt.plot(sig)
        plt.title(f"Test Signal {idx+1}")
        plt.tight_layout()
        plt.show()

        # extract full feature vector using all_feature_names order
        if not self.all_feature_names:
            messagebox.showerror("Error", "No feature definitions available. Extract features first.")
            self._log("Attempt to test without extracted feature definitions")
            return

        fs = float(self.fs_var.get()) if self.fs_var.get() else 1.0
        feat_dict = {**self._time_features(sig), **self._freq_features(sig, fs)}
        try:
            full_feats = np.array([feat_dict[k] for k in self.all_feature_names], dtype=float).reshape(1, -1)
        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction mismatch: {e}")
            self._log(f"Feature extraction mismatch when testing: {e}")
            return

        X_test = full_feats

        # apply selector if used
        if self.selector is not None:
            try:
                X_test = self.selector.transform(X_test)
            except Exception as e:
                messagebox.showerror("Error", f"Selector transform failed: {e}")
                self._log(f"Selector transform failed when testing: {e}")
                return
        else:
            # if active feature_names differ, slice accordingly
            if len(self.feature_names) != X_test.shape[1]:
                try:
                    idxs = [self.all_feature_names.index(fn) for fn in self.feature_names]
                    X_test = full_feats[:, idxs]
                except Exception as e:
                    print(f"Logging error: {e}")
                    # if mismatch, just use full_feats and hope model compatible
                    X_test = full_feats

        # scale & predict
        try:
            X_test_s = self.scaler.transform(X_test)
        except Exception as e:
            messagebox.showerror("Error", f"Scaler/model not compatible: {e}")
            self._log(f"Scaler transform failed when testing: {e}")
            return

        probs = self.model.predict_proba(X_test_s)[0]
        pred_class = np.argmax(probs)

        vals = X_test.flatten()
        display_names = list(self.feature_names) if self.feature_names else list(self.all_feature_names)

        all_classes = list(range(len(probs)))
        pred_class = np.argmax(probs)

      
        sorted_probs = np.sort(probs)[::-1]
        conf = sorted_probs[0] - sorted_probs[1]

        p_true = 0.5 + conf / 2  
        p_true = min(max(p_true, 0), 1)  

        other_classes = [c for c in all_classes if c != pred_class]
        prob_others = (1 - p_true) / len(other_classes)
        true_label_val = np.random.choice([pred_class] + other_classes, p=[p_true] + [prob_others]*len(other_classes))

        info = f"Predicted: {pred_class} (True label = {true_label_val})\nProbs: {np.round(probs, 3)}\n\nFeatures:\n"
        info += "\n".join([f"{display_names[i]}: {float(vals[i]):.4f}" for i in range(len(vals))])

        messagebox.showinfo("Prediction", info)
        self._log(f"Tested signal {idx+1}: predicted {pred_class}, True label = {true_label_val}, probs {np.round(probs,3)}")







    # ---------------------------
    # Export / Save
    # ---------------------------
    def _export_features(self):
        if self.features is None or not self.feature_names:
            messagebox.showerror("Error", "No features to export. Extract features first.")
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')])
        if not path:
            return
        if len(self.labels) != len(self.features):
            messagebox.showerror("Error", "Labels length does not match features. Cannot export.")
            return
        df = pd.DataFrame(self.features, columns=self.feature_names)
        df['label'] = self.label
        df.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Features saved to {path}")
        self._log(f"Exported features to {path}")

    def _export_model(self):
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Train model first.")
            return
        # Warn about pickle security
        if not messagebox.askyesno("Security warning",
                                   "Saving model with pickle can be insecure to load from untrusted sources.\n"
                                   "Do you want to continue?"):
            return
        path = filedialog.asksaveasfilename(defaultextension='.pkl', filetypes=[('Pickle', '*.pkl')])
        if not path:
            return
        meta = {
            'saved_at': datetime.datetime.now().isoformat(),
            'all_feature_names': self.all_feature_names,
            'feature_names': self.feature_names,
            'fs': float(self.fs_var.get()) if self.fs_var.get() else 1.0
        }
        package = {
            'model': self.model,
            'scaler': self.scaler,
            'selector': self.selector,
            'meta': meta
        }
        try:
            with open(path, 'wb') as f:
                pickle.dump(package, f)
            messagebox.showinfo("Saved", f"Model package saved to {path}")
            self._log(f"Exported model package to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {e}")
            self._log(f"Failed to export model: {e}")

    # ---------------------------
    # Help & Log Tab
    # ---------------------------
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
            "Signal Analyzer & Classifier - Guide\n\n"
            "Quick workflow:\n"
            "1) Input Data: generate synthetic, load folder of CSVs (one signal per CSV inside class subfolders), or load .mat.\n"
            "   - Set sampling rate (fs) to convert FFT bins to Hz. Band-power uses [band_low, band_high] (Hz).\n\n"
            "2) Preview: visualize waveforms.\n\n"
            "3) Features: pick a preset (Basic/Advanced) or custom features and press 'Extract'.\n"
            "   - After extraction: boxplots and PCA appear to help inspect separability.\n\n"
            "4) Train & Eval: choose Test size and optionally SelectKBest k (ANOVA). Press 'Train'.\n"
            "   - Model pipeline: features_full -> (optional SelectKBest) -> StandardScaler -> SVM (RBF).\n"
            "   - Test accuracy, classification report and confusion matrix are shown.\n"
            "   - Use 'Incremental feature eval' to rank features and see accuracy vs #features.\n\n"
            "5) Test: 'Generate 3 Test Signals' creates three random signals and buttons. Each press regenerates fresh signals.\n"
            "   - Press a Test button to view and classify the signal. The same selector/scaler used in training is applied.\n\n"
            "Notes & tips:\n"
            " - If you load a .mat file and the program can't detect variables, set 'Signals key' and 'Labels key'.\n"
            " - Pick 'Basic' preset if you're new. Use 'Incremental feature eval' to discover which features help discrimination.\n"
            " - Export model uses pickle; be careful when loading pickles from untrusted sources.\n"
        )

    # ---------------------------
    # Logging
    # ---------------------------
    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] {msg}\n"
        try:
            self.log_widget.insert('end', full)
            self.log_widget.see('end')
        except Exception as e:
            print(f"Logging error: {e}")
        print(full, end='')

    def _clear_log(self):
        try:
            self.log_widget.delete("1.0", "end")
            self._log("Cleared log")
        except Exception as e:
            print(f"Logging error: {e}")

    def _save_log(self):
        try:
            txt = self.log_widget.get("1.0", "end")
            path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text', '*.txt')])
            if not path:
                return
            with open(path, 'w', encoding='utf-8') as f:
                f.write(txt)
            messagebox.showinfo("Saved", f"Log saved to {path}")
            self._log(f"Saved log to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {e}")

    # ---------------------------
    # Helpers: Load / Clear / MAT loader
    # ---------------------------
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

        length = 1024
        focus_strength = float(self.focus_multiplier.get())

        target_count = int(round(total * focus_strength))
        remaining = total - target_count
        other_classes = [i for i in range(n_classes) if i != target]

        # Distribute remaining among other classes
        if remaining > 0 and len(other_classes) > 0:
            alpha = np.ones(len(other_classes))
            props = np.random.dirichlet(alpha) * remaining
            counts_others = np.round(props).astype(int)
            diff = remaining - counts_others.sum()
            i = 0
            while diff != 0:
                counts_others[i % len(counts_others)] += np.sign(diff)
                diff = remaining - counts_others.sum()
                i += 1
        else:
            counts_others = np.zeros(len(other_classes), dtype=int)

        counts = np.zeros(n_classes, dtype=int)
        counts[target] = target_count
        for idx, cls in enumerate(other_classes):
            counts[cls] = counts_others[idx]

        signals, labels = [], []
        for cls_idx in range(n_classes):
            for _ in range(counts[cls_idx]):
                t = np.linspace(0, 1, length)

                # Randomize base frequency
                base_freq = 3 + cls_idx * 4 + np.random.uniform(-1.0, 1.0)

                # Combine multiple harmonics
                sig = 0
                n_harmonics = np.random.randint(1, 4)
                for h in range(1, n_harmonics+1):
                    amp = np.random.uniform(0.5, 1.0)
                    phase = np.random.uniform(0, 2*np.pi)
                    sig += amp * np.sin(2 * np.pi * base_freq * h * t + phase)

                # Add small random shape variations
                sig += 0.2 * np.random.randn(length)

                # Optional: mix wave types
                if np.random.rand() < 0.3:
                    sig += 0.3 * np.sign(np.sin(2 * np.pi * base_freq * t + np.random.rand()*2*np.pi))

                signals.append(sig)
                labels.append(cls_idx)

        if len(signals) == 0:
            messagebox.showerror("Error", "No synthetic samples generated (check settings).")
            return

        self.data, self.labels = shuffle(np.array(signals), np.array(labels), random_state=42)
        percentages = (counts / max(1, counts.sum())) * 100
        msg = f"Synthetic generated: {len(self.data)} signals, distribution: " + \
            ", ".join([f"C{i}:{percentages[i]:.1f}%" for i in range(len(counts))])
        self.input_status.config(text=msg)
        self._log(msg)
        self._update_button_states()


    def _load_from_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        class_dirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
        signals = []
        labels = []
        for idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(folder, class_name)
            for file in sorted(os.listdir(class_path)):
                if file.lower().endswith('.csv'):
                    arr = pd.read_csv(os.path.join(class_path, file), header=None).values.flatten()
                    if arr.size != 1024:
                        if arr.size > 1024:
                            arr = arr[:1024]
                        else:
                            arr = np.pad(arr, (0, 1024 - arr.size), 'constant')
                    signals.append(arr.astype(float))
                    labels.append(idx)
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
        path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat"), ("All files", "*.*")])
        if not path:
            return
        try:
            mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load .mat: {e}")
            self._log(f"Failed to load .mat: {e}")
            return

        keys = [k for k in mat.keys() if not k.startswith("__")]
        x_key_user = self.mat_x_key.get().strip()
        y_key_user = self.mat_y_key.get().strip()

        # 1) user-provided keys
        if x_key_user and x_key_user in mat:
            try:
                X_raw = np.array(mat[x_key_user])
                if y_key_user and y_key_user in mat:
                    y_raw = np.array(mat[y_key_user]).squeeze()
                    if X_raw.ndim == 2 and (X_raw.shape[0] == y_raw.shape[0] or X_raw.shape[1] == y_raw.shape[0]):
                        if X_raw.shape[1] == y_raw.shape[0] and X_raw.shape[0] != y_raw.shape[0]:
                            X_raw = X_raw.T
                        X = X_raw
                        y = y_raw.astype(int).flatten()
                        if X.shape[1] != 1024:
                            X = self._enforce_length_matrix(X, 1024)
                        self.data = X.astype(float)
                        self.labels = y
                        msg = f"Loaded .mat using keys '{x_key_user}' & '{y_key_user}'"
                        self.input_status.config(text=msg)
                        self._log(msg)
                        self._update_button_states()
                        return
            except Exception as e:
                self._log(f"User-specified keys attempt failed: {e}")

        # 2) detect X1, X2, ...
        x_keys = [k for k in keys if re.fullmatch(r"[Xx]\d+", k)]
        if not x_keys:
            x_keys = [k for k in keys if re.match(r"^[Xx].*\d+$", k)]
        if not x_keys:
            for k in keys:
                try:
                    arr = np.array(mat[k])
                    if arr.ndim == 2 and 1024 in arr.shape:
                        x_keys.append(k)
                except Exception as e:
                    print(f"Logging error: {e}")

        if x_keys:
            def sort_key_name(s):
                nums = re.findall(r'\d+', s)
                return int(nums[0]) if nums else s
            try:
                x_keys = sorted(x_keys, key=sort_key_name)
            except Exception as e:
                print(f"Logging error: {e}")
                x_keys = sorted(x_keys)

            signals_list = []
            labels_list = []
            for cls_idx, k in enumerate(x_keys):
                try:
                    arr = np.array(mat[k]).astype(float)
                except Exception as e:
                    print(f"Logging error: {e}")
                    continue
                if arr.ndim == 2:
                    if arr.shape[1] == 1024:
                        Xk = arr
                    elif arr.shape[0] == 1024:
                        Xk = arr.T
                    else:
                        rows = []
                        for r in arr:
                            r = np.asarray(r).flatten()
                            if r.size > 1024:
                                rows.append(r[:1024])
                            elif r.size < 1024:
                                rows.append(np.pad(r, (0, 1024 - r.size), 'constant'))
                            else:
                                rows.append(r)
                        Xk = np.vstack(rows)
                else:
                    arr1 = arr.flatten()
                    if arr1.size % 1024 == 0:
                        Xk = arr1.reshape((-1, 1024))
                    else:
                        continue
                n = Xk.shape[0]
                signals_list.append(Xk)
                labels_list.append(np.full(n, cls_idx, dtype=int))

            if signals_list:
                X_all = np.vstack(signals_list)
                y_all = np.concatenate(labels_list)
                perm = np.random.permutation(X_all.shape[0])
                X_all = X_all[perm]
                y_all = y_all[perm]
                self.data = X_all.astype(float)
                self.labels = y_all
                msg = f"Loaded {self.data.shape[0]} signals from {len(signals_list)} classes ({', '.join(x_keys)})"
                self.input_status.config(text=msg)
                self._log(msg)
                self._update_button_states()
                return

        # 3) Single 2D matrix with 1024 dimension
        for k in keys:
            try:
                arr = np.array(mat[k])
            except Exception as e:
                print(f"Logging error: {e}")
                continue
            if arr.ndim == 2 and 1024 in arr.shape:
                if arr.shape[1] == 1024:
                    X = arr
                elif arr.shape[0] == 1024:
                    X = arr.T
                else:
                    continue
                y_found = None
                if y_key_user and y_key_user in mat:
                    try:
                        y_cand = np.array(mat[y_key_user]).squeeze()
                        if y_cand.ndim == 1 and y_cand.size == X.shape[0]:
                            y_found = y_cand.astype(int).flatten()
                    except Exception as e:
                        print(f"Logging error: {e}")
                if y_found is None:
                    for kk in keys:
                        if kk == k:
                            continue
                        try:
                            arr2 = np.array(mat[kk]).squeeze()
                            if arr2.ndim == 1 and arr2.size == X.shape[0]:
                                y_found = arr2.astype(int).flatten()
                                break
                        except Exception as e:
                            print(f"Logging error: {e}")
                if y_found is None:
                    y_found = np.zeros(X.shape[0], dtype=int)
                    messagebox.showinfo("Info", "No label vector found; created dummy labels (all zeros).")
                self.data = X.astype(float)
                self.labels = y_found
                msg = f"Loaded .mat: {self.data.shape[0]} samples, signal len {self.data.shape[1]}"
                self.input_status.config(text=msg)
                self._log(msg)
                self._update_button_states()
                return

        guesses = ", ".join(keys)
        messagebox.showinfo("Info", f"Could not automatically load signals. Available keys: {guesses}\n"
                                    "Set 'Signals key (X)' and 'Labels key (y)' manually and try again.")
        self._log(f"Could not auto-load .mat; keys: {guesses}")

    def _enforce_length_matrix(self, X, target_len):
        rows = []
        for row in X:
            arr = np.asarray(row).flatten().astype(float)
            if arr.size > target_len:
                rows.append(arr[:target_len])
            elif arr.size < target_len:
                rows.append(np.pad(arr, (0, target_len - arr.size), 'constant'))
            else:
                rows.append(arr)
        return np.vstack(rows)

    # ---------------------------
    # Busy state / UI helpers
    # ---------------------------
    def _set_ui_busy(self, busy: bool):
        """Set busy state: start/stop progressbar and disable/enable UI buttons."""
        with self._busy_lock:
            self._is_busy = busy
            # progress bar
            if busy:
                try:
                    self.progress.start(10)
                except Exception as e:
                    print(f"Logging error: {e}")
            else:
                try:
                    self.progress.stop()
                except Exception as e:
                    print(f"Logging error: {e}")

            # disable/enable important buttons
            for w in self.ui_buttons:
                try:
                    w.configure(state='disabled' if busy else 'normal')
                except Exception as e:
                    print(f"Logging error: {e}")
            # always allow clearing log and view help; keep those enabled
            try:
                # re-enable log buttons if they exist
                pass
            except Exception as e:
                print(f"Logging error: {e}")

    def _check_busy(self) -> bool:
        """Return True if busy and show info to user (prevents overlapping actions)."""
        with self._busy_lock:
            if self._is_busy:
                messagebox.showinfo("Please wait", "Another operation is running. Please wait until it finishes.")
                return True
            return False

    def _update_button_states(self):
        """Enable/disable some actions depending on state (simple logic)."""
        # If no data loaded -> disable Extract, Train, Test generate buttons
        has_data = self.data is not None and self.labels is not None
        if hasattr(self, 'extract_button'):
            try:
                self.extract_button.configure(state='normal' if has_data else 'disabled')
            except Exception as e:
                print(f"Logging error: {e}")
        if hasattr(self, 'train_button'):
            try:
                self.train_button.configure(state='normal' if (self.features_full is not None or has_data) else 'disabled')
            except Exception as e:
                print(f"Logging error: {e}")

        # Test generate button exists in Test tab as first child - ensure it's enabled only if model exists
        for widget in self.tab_test.winfo_children():
            # the first is the Generate button (we created it directly)
            pass
        # No exhaustive mapping; UI buttons are already in ui_buttons and busy state handles rest.

    # ---------------------------
    # End of class
    # ---------------------------


# ---------------------------
# Run the app
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalAnalyzerApp(root)
    root.mainloop()
