#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SignalToolkit - Tkinter desktop app for two-class signal feature-extraction,
selection, training and visualization (University project ready).

Expect .mat file with variables: X1 (m1 x 1024), X2 (m2 x 1024)
Rows = samples, cols = 1024 timepoints.

Features: time-domain (mean, var, skew, kurtosis, rms, zcr, energy, iqr, mad, ptp)
          freq-domain (Welch PSD -> centroid, bandwidth, rolloff95, flatness, entropy,
                       dominant freq/power, bandpower thirds)

Capabilities:
 - Load .mat interactively
 - Extract features
 - Manual selection (checkboxes)
 - Auto selection: mutual information (MI), ANOVA F, PCA ranking
 - Incremental feature addition & CV accuracy plot
 - Train multiple models (SVM linear/rbf, kNN, RF, Logistic, NaiveBayes)
 - Compare via CV and evaluate on holdout test
 - Boxplots, PCA scatter, t-SNE scatter, ROC & confusion
 - Test on random/noisy signals
 - Save report (JSON) and model (joblib)
"""
import os, sys, json, datetime
import numpy as np
import scipy.io as sio
import scipy.signal as sps
import scipy.stats as stats
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ----------------------- Config / Defaults -----------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PWELCH_WINDOW = 256
PWELCH_NOOVERLAP = 128
PWELCH_NFFT = 1024

# ----------------------- Utilities -----------------------
def ensure_row_matrix(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X

def size_str(X):
    try:
        return f"{X.shape}"
    except:
        return str(type(X))

# ----------------------- Feature extraction -----------------------
def extract_time_features(sig):
    # sig: 1D numpy array
    feats = {}
    feats['mean'] = float(np.mean(sig))
    feats['var'] = float(np.var(sig))
    feats['std'] = float(np.std(sig))
    feats['skew'] = float(stats.skew(sig))
    feats['kurt'] = float(stats.kurtosis(sig))
    feats['rms'] = float(np.sqrt(np.mean(sig**2)))
    zc = np.sum(np.abs(np.diff(np.sign(sig)))) / (2.0 * len(sig))
    feats['zcr'] = float(zc)
    feats['energy'] = float(np.sum(sig**2))
    feats['iqr'] = float(np.percentile(sig,75) - np.percentile(sig,25))
    feats['mad'] = float(np.mean(np.abs(sig - np.mean(sig))))
    feats['max'] = float(np.max(sig))
    feats['min'] = float(np.min(sig))
    feats['ptp'] = float(np.ptp(sig))
    return feats

def extract_freq_features(sig, fs=1.0, nperseg=PWELCH_WINDOW, noverlap=PWELCH_NOOVERLAP, nfft=PWELCH_NFFT):
    f, Pxx = sps.welch(sig, fs=fs, window='hamming', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    P = Pxx.copy()
    Psum = np.sum(P) + 1e-12
    Pn = P / Psum
    feats = {}
    feats['spec_centroid'] = float(np.sum(f * Pn))
    feats['spec_bandwidth'] = float(np.sqrt(np.sum(((f - feats['spec_centroid'])**2) * Pn)))
    cumsum = np.cumsum(Pn)
    feats['spec_rolloff95'] = float(f[np.searchsorted(cumsum, 0.95)])
    geo_mean = stats.gmean(P + 1e-12)
    feats['spec_flatness'] = float(geo_mean / (np.mean(P) + 1e-12))
    feats['spec_entropy'] = float(-np.sum(Pn * np.log2(Pn + 1e-12)))
    dom_idx = int(np.argmax(P))
    feats['dom_freq'] = float(f[dom_idx])
    feats['dom_power'] = float(P[dom_idx])
    n = len(P)
    feats['bandpower_low'] = float(np.sum(Pn[:n//3]))
    feats['bandpower_mid'] = float(np.sum(Pn[n//3:2*n//3]))
    feats['bandpower_high'] = float(np.sum(Pn[2*n//3:]))
    return feats

def extract_features_matrix(X, fs=1.0):
    X = ensure_row_matrix(X)
    feat_list = []
    for i in range(X.shape[0]):
        sig = X[i,:].astype(float)
        tfs = extract_time_features(sig)
        ffs = extract_freq_features(sig, fs=fs)
        allf = {**tfs, **ffs}
        feat_list.append(allf)
    feat_names = list(feat_list[0].keys())
    F = np.array([[d[n] for n in feat_names] for d in feat_list], dtype=float)
    return F, feat_names

# ----------------------- Feature ranking / selection -----------------------
def rank_mi(X, y):
    mi = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    idx = np.argsort(mi)[::-1]
    return idx, mi

def rank_anova(X, y):
    fvals, pvals = f_classif(X, y)
    idx = np.argsort(fvals)[::-1]
    return idx, fvals

def rank_pca(X):
    pca = PCA()
    pca.fit(X)
    comps = np.abs(pca.components_) * pca.explained_variance_[:,None]
    scores = comps.sum(axis=0)
    idx = np.argsort(scores)[::-1]
    return idx, scores

def incremental_cv_accuracy(X, y, order_idx, cv=5, model=None):
    if model is None:
        model = LogisticRegression(max_iter=2000)
    accs = []
    for k in range(1, len(order_idx)+1):
        sel = order_idx[:k]
        try:
            scores = cross_val_score(model, X[:, sel], y, cv=cv, scoring='accuracy', n_jobs=1)
            accs.append(float(np.mean(scores)))
        except Exception:
            accs.append(float('nan'))
    return accs

# ----------------------- Models / helpers -----------------------
def build_model(name):
    if name == 'SVM-linear':
        return SVC(kernel='linear', probability=True, random_state=RANDOM_SEED)
    if name == 'SVM-rbf':
        return SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
    if name == 'kNN':
        return KNeighborsClassifier(n_neighbors=5)
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    if name == 'Logistic':
        return LogisticRegression(max_iter=2000)
    if name == 'NaiveBayes':
        return GaussianNB()
    raise ValueError("Unknown model")

def compare_models_cv(X, y, model_names, cv=5):
    res = {}
    for name in model_names:
        clf = build_model(name)
        try:
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
            res[name] = {'mean_acc': float(np.mean(scores)), 'std_acc': float(np.std(scores))}
        except Exception as e:
            res[name] = {'mean_acc': float('nan'), 'std_acc': float('nan')}
    return res

# ----------------------- Plot helpers -----------------------
def clear_axis(ax):
    ax.clear()
    ax.set_xticks([]); ax.set_yticks([])
    ax.figure.tight_layout()

def plot_sample_signals(ax, X1, X2, nshow=3):
    ax.clear()
    n1 = min(nshow, X1.shape[0])
    n2 = min(nshow, X2.shape[0])
    t = np.arange(X1.shape[1])
    for i in range(n1):
        ax.plot(t, X1[i,:], color='tab:blue', alpha=0.6)
    for j in range(n2):
        ax.plot(t, X2[j,:], color='tab:orange', alpha=0.6)
    ax.set_title('Sample signals (class1:blue, class2:orange)')
    ax.set_xlabel('Sample index'); ax.set_ylabel('Amplitude')
    ax.figure.tight_layout()

def plot_boxplots(ax, F, y, feat_names, sel_idx):
    ax.clear()
    if len(sel_idx)==0:
        ax.text(0.5,0.5,"No features selected", ha='center')
        return
    # plot up to 6 features
    k = min(6, len(sel_idx))
    fig = ax.figure
    fig.clf()
    axes = fig.subplots(1,k)
    if k==1:
        axes = [axes]
    for i in range(k):
        fn = feat_names[sel_idx[i]]
        data = [F[y==c, sel_idx[i]] for c in np.unique(y)]
        axes[i].boxplot(data, labels=[f"Class {int(c)}" for c in np.unique(y)])
        axes[i].set_title(fn)
    fig.tight_layout()

def plot_pca(ax, F, y, sel_idx):
    ax.clear()
    if len(sel_idx)==0:
        ax.text(0.5,0.5,"No features selected", ha='center')
        return
    Xsel = F[:, sel_idx]
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xsel)
    for c in np.unique(y):
        ax.scatter(Z[y==c,0], Z[y==c,1], label=f"Class {int(c)}", s=20)
    ax.legend(); ax.set_title('PCA 2D'); ax.figure.tight_layout()

def plot_tsne(ax, F, y, sel_idx, perplexity=30, n_iter=1000, max_points=800):
    ax.clear()
    if len(sel_idx)==0:
        ax.text(0.5,0.5,"No features selected", ha='center')
        return
    Xsel = F[:, sel_idx]
    n = Xsel.shape[0]
    if n > max_points:
        idxs = np.random.choice(n, size=max_points, replace=False)
        Xsub = Xsel[idxs]; ysub = y[idxs]
    else:
        Xsub = Xsel; ysub = y
    tsne = TSNE(n_components=2, perplexity=min(50, max(5,perplexity)), n_iter=n_iter, random_state=RANDOM_SEED)
    Z = tsne.fit_transform(Xsub)
    for c in np.unique(ysub):
        ax.scatter(Z[ysub==c,0], Z[ysub==c,1], label=f"Class {int(c)}", s=20)
    ax.legend(); ax.set_title('t-SNE'); ax.figure.tight_layout()

def plot_roc(ax, clf, Xtest, ytest):
    ax.clear()
    if len(np.unique(ytest))!=2:
        ax.text(0.5,0.5,"ROC needs 2 classes", ha='center')
        return np.nan
    try:
        prob = clf.predict_proba(Xtest)[:,1]
    except Exception:
        # some classifiers don't have predict_proba
        try:
            prob = clf.decision_function(Xtest)
            prob = (prob - prob.min())/(prob.max()-prob.min()+1e-12)
        except Exception:
            ax.text(0.5,0.5,"Model has no probability output", ha='center')
            return np.nan
    ybin = (ytest==2).astype(int)  # treat class 2 as positive
    fpr, tpr, _ = roc_curve(ybin, prob)
    roc_auc = auc(fpr,tpr)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],'k--'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend()
    ax.set_title('ROC Curve'); ax.figure.tight_layout()
    return roc_auc

def plot_confusion(ax, ytrue, ypred):
    ax.clear()
    cm = confusion_matrix(ytrue, ypred)
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.figure.tight_layout()

# ----------------------- Main Tkinter App -----------------------
class SignalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signal Toolkit - Feature & Classifier Studio")
        self.geometry("1200x720")
        # data
        self.mat_path = None
        self.X1 = None; self.X2 = None
        self.F = None; self.feat_names = None; self.y = None
        self.selected_idx = []
        self.rankings = {}
        self.scaler = None
        self.fitted_models = {}
        self.best_model_name = None
        self.last_compare = {}
        self._build_ui()

    def _build_ui(self):
        # Top: menu bar buttons
        top_frame = ttk.Frame(self)
        top_frame.pack(side='top', fill='x', padx=6, pady=6)
        btn_load = ttk.Button(top_frame, text="Load .mat", command=self.load_mat)
        btn_load.pack(side='left', padx=4)
        btn_extract = ttk.Button(top_frame, text="Extract Features", command=self.extract_features)
        btn_extract.pack(side='left', padx=4)

        btn_save = ttk.Button(top_frame, text="Save Report & Model", command=self.save_report_and_model)
        btn_save.pack(side='right', padx=4)

        # Notebook with tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=6, pady=6)

        # Tab: Load & preview
        self.tab_load = ttk.Frame(self.nb); self.nb.add(self.tab_load, text="Load & Preview")
        self._build_tab_load()

        # Tab: Features
        self.tab_feats = ttk.Frame(self.nb); self.nb.add(self.tab_feats, text="Features")
        self._build_tab_features()

        # Tab: Selection
        self.tab_sel = ttk.Frame(self.nb); self.nb.add(self.tab_sel, text="Selection & Visualize")
        self._build_tab_selection()

        # Tab: Training
        self.tab_train = ttk.Frame(self.nb); self.nb.add(self.tab_train, text="Training & Compare")
        self._build_tab_training()

        # Tab: Testing
        self.tab_test = ttk.Frame(self.nb); self.nb.add(self.tab_test, text="Testing")
        self._build_tab_testing()

    # --------------- Tab builders ---------------
    def _build_tab_load(self):
        frm = self.tab_load
        lbl = ttk.Label(frm, text="Load a MATLAB .mat file containing X1 and X2 (rows = samples, cols = 1024)")
        lbl.pack(anchor='w', padx=8, pady=6)
        self.lbl_info = ttk.Label(frm, text="No file loaded", foreground='blue')
        self.lbl_info.pack(anchor='w', padx=8)

        # plot area
        plot_frame = ttk.Frame(frm)
        plot_frame.pack(fill='both', expand=True, padx=8, pady=8)
        self.fig_preview = plt.Figure(figsize=(8,4))
        self.ax_preview = self.fig_preview.add_subplot(111)
        self.canvas_preview = FigureCanvasTkAgg(self.fig_preview, master=plot_frame)
        self.canvas_preview.get_tk_widget().pack(fill='both', expand=True)

    def _build_tab_features(self):
        frm = self.tab_feats
        left = ttk.Frame(frm); left.pack(side='left', fill='y', padx=8, pady=8)
        right = ttk.Frame(frm); right.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        ttk.Label(left, text="Feature list (toggle to include):").pack(anchor='w')
        self.feature_canvas = tk.Canvas(left, width=320, height=420)
        self.feature_scroll = ttk.Scrollbar(left, orient='vertical', command=self.feature_canvas.yview)
        self.feature_frame = ttk.Frame(self.feature_canvas)
        self.feature_frame.bind("<Configure>", lambda e: self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all")))
        self.feature_canvas.create_window((0,0), window=self.feature_frame, anchor='nw')
        self.feature_canvas.configure(yscrollcommand=self.feature_scroll.set)
        self.feature_canvas.pack(side='left', fill='y'); self.feature_scroll.pack(side='right', fill='y')

        # right: quick stats and boxplot
        ttk.Label(right, text="Quick visual:").pack(anchor='w')
        self.fig_feats = plt.Figure(figsize=(6,4))
        self.ax_feats = self.fig_feats.add_subplot(111)
        self.canvas_feats = FigureCanvasTkAgg(self.fig_feats, master=right)
        self.canvas_feats.get_tk_widget().pack(fill='both', expand=True)

        btn_box = ttk.Button(right, text="Show Boxplots (selected)", command=self.show_boxplots)
        btn_box.pack(anchor='s', pady=6)

    def _build_tab_selection(self):
        frm = self.tab_sel
        top = ttk.Frame(frm); top.pack(fill='x', padx=8, pady=6)
        ttk.Label(top, text="Auto selection method:").pack(side='left')
        self.auto_var = tk.StringVar(value='mi')
        cb = ttk.Combobox(top, textvariable=self.auto_var, values=['mi','anova','pca','none'], state='readonly', width=8)
        cb.pack(side='left', padx=6)
        ttk.Label(top, text="Keep top K:").pack(side='left', padx=6)
        self.num_keep_var = tk.IntVar(value=10)
        sp = ttk.Spinbox(top, from_=1, to=100, textvariable=self.num_keep_var, width=6)
        sp.pack(side='left')
        ttk.Button(top, text="Apply Auto Selection", command=self.apply_auto_selection).pack(side='left', padx=6)

        # Incremental evaluation
        mid = ttk.Frame(frm); mid.pack(fill='x', padx=8, pady=6)
        ttk.Label(mid, text="Incremental (ranked) CV accuracy:").pack(side='left')
        self.kfold_inc = tk.IntVar(value=5)
        ttk.Label(mid, text="k-fold:").pack(side='left', padx=6)
        ttk.Spinbox(mid, from_=2, to=10, textvariable=self.kfold_inc, width=4).pack(side='left')
        ttk.Button(mid, text="Run Incremental", command=self.run_incremental).pack(side='left', padx=6)

        # Visual area for PCA/t-SNE
        bot = ttk.Frame(frm); bot.pack(fill='both', expand=True, padx=8, pady=6)
        self.fig_sel = plt.Figure(figsize=(6,4))
        self.ax_sel = self.fig_sel.add_subplot(111)
        self.canvas_sel = FigureCanvasTkAgg(self.fig_sel, master=bot)
        self.canvas_sel.get_tk_widget().pack(fill='both', expand=True)
        btn_pca = ttk.Button(bot, text="Show PCA (selected)", command=self.show_pca)
        btn_pca.pack(side='left', padx=6)
        btn_tsne = ttk.Button(bot, text="Show t-SNE (selected)", command=self.show_tsne_dialog)
        btn_tsne.pack(side='left', padx=6)

    def _build_tab_training(self):
        frm = self.tab_train
        top = ttk.Frame(frm); top.pack(fill='x', padx=8, pady=6)
        ttk.Label(top, text="Select models to train:").pack(anchor='w')
        self.models = ['SVM-linear','SVM-rbf','kNN','RandomForest','Logistic','NaiveBayes']
        self.model_vars = {}
        row = ttk.Frame(frm); row.pack(fill='x', padx=8)
        for i,m in enumerate(self.models):
            v = tk.BooleanVar(value=(m in ['SVM-linear','RandomForest']))
            cb = ttk.Checkbutton(row, text=m, variable=v)
            cb.grid(row=0, column=i, padx=6, sticky='w')
            self.model_vars[m] = v

        params = ttk.Frame(frm); params.pack(fill='x', padx=8, pady=6)
        ttk.Label(params, text="Holdout test fraction:").pack(side='left')
        self.holdout_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(params, from_=0.05, to=0.5, increment=0.05, textvariable=self.holdout_var, width=6).pack(side='left', padx=6)
        ttk.Label(params, text="CV folds:").pack(side='left', padx=8)
        self.kfold_var = tk.IntVar(value=5)
        ttk.Spinbox(params, from_=2, to=10, textvariable=self.kfold_var, width=4).pack(side='left')

        ttk.Button(frm, text="Train & Compare", command=self.train_and_compare).pack(padx=8, pady=6)

        # results text and plot
        bot = ttk.Frame(frm); bot.pack(fill='both', expand=True, padx=8, pady=6)
        self.txt_results = tk.Text(bot, height=10)
        self.txt_results.pack(side='left', fill='both', expand=True)
        self.fig_train = plt.Figure(figsize=(5,4))
        self.ax_train = self.fig_train.add_subplot(111)
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=bot)
        self.canvas_train.get_tk_widget().pack(side='left', fill='both', expand=True)

    def _build_tab_testing(self):
        frm = self.tab_test
        top = ttk.Frame(frm); top.pack(fill='x', padx=8, pady=6)
        ttk.Button(top, text="Test on random signals", command=self.test_on_random).pack(side='left', padx=6)
        ttk.Button(top, text="Show ROC & Confusion (best)", command=self.show_roc_conf).pack(side='left', padx=6)
        # plot area
        bot = ttk.Frame(frm); bot.pack(fill='both', expand=True, padx=8, pady=6)
        self.fig_test = plt.Figure(figsize=(8,4))
        self.ax_test = self.fig_test.add_subplot(121)
        self.ax_test2 = self.fig_test.add_subplot(122)
        self.canvas_test = FigureCanvasTkAgg(self.fig_test, master=bot)
        self.canvas_test.get_tk_widget().pack(fill='both', expand=True)

    # ----------------- Actions -----------------
    def load_mat(self):
        path = filedialog.askopenfilename(title="Select MATLAB .mat", filetypes=[("MAT files","*.mat")])
        if not path:
            return
        try:
            S = sio.loadmat(path)
            if 'X1' not in S or 'X2' not in S:
                messagebox.showerror("Error", "MAT file must contain X1 and X2 variables")
                return
            self.mat_path = path
            self.X1 = ensure_row_matrix(np.asarray(S['X1']))
            self.X2 = ensure_row_matrix(np.asarray(S['X2']))
            self.lbl_info.config(text=f"Loaded: {os.path.basename(path)} | X1={self.X1.shape}, X2={self.X2.shape}")
            self.log(f"Loaded MAT: {path} | X1={self.X1.shape} X2={self.X2.shape}")
            # preview signals
            plot_sample_signals(self.ax_preview, self.X1, self.X2, nshow=3)
            self.canvas_preview.draw()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def extract_features(self):
        if self.X1 is None or self.X2 is None:
            messagebox.showwarning("No data", "Load MAT file first")
            return
        self.log("Extracting features...")
        F1, names = extract_features_matrix(self.X1, fs=1.0)
        F2, _ = extract_features_matrix(self.X2, fs=1.0)
        F = np.vstack([F1, F2])
        y = np.hstack([np.ones(F1.shape[0], dtype=int), 2*np.ones(F2.shape[0], dtype=int)])
        self.F = F; self.feat_names = names; self.y = y
        self.log(f"Extracted features matrix: {F.shape} ; features: {len(names)}")
        # populate checkboxes
        for w in self.feature_frame.winfo_children():
            w.destroy()
        self.feat_vars = []
        for i,n in enumerate(names):
            v = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(self.feature_frame, text=f"{i+1:02d}. {n}", variable=v)
            cb.pack(anchor='w', pady=1)
            self.feat_vars.append(v)
        # compute rankings (try)
        try:
            idx_mi, mi = rank_mi(self.F, self.y)
            idx_an, fvals = rank_anova(self.F, self.y)
            idx_p, pscore = rank_pca(self.F)
            self.rankings['mi'] = (idx_mi, mi)
            self.rankings['anova'] = (idx_an, fvals)
            self.rankings['pca'] = (idx_p, pscore)
            self.log("Computed feature rankings (MI, ANOVA, PCA)")
        except Exception as e:
            self.log(f"Ranking failed: {e}")
        self.show_boxplots()

    def _get_selected_indices(self):
        if not hasattr(self, 'feat_vars'):
            return []
        return [i for i,v in enumerate(self.feat_vars) if v.get()]

    def apply_auto_selection(self):
        if self.F is None:
            messagebox.showwarning("No features", "Extract features first")
            return
        method = self.auto_var.get(); k = int(self.num_keep_var.get())
        if method == 'none':
            return
        if method not in self.rankings:
            messagebox.showwarning("No ranking", "Feature rankings not available")
            return
        idxs, scores = self.rankings[method]
        keep = idxs[:k].tolist()
        for i,v in enumerate(self.feat_vars):
            v.set(i in keep)
        self.log(f"Auto-selected top {k} by {method}")

    def run_incremental(self):
        if self.F is None:
            messagebox.showwarning("No features", "Extract features first")
            return
        method = self.auto_var.get()
        if method not in self.rankings:
            messagebox.showwarning("Ranking required", "Compute feature rankings first (extract features)")
            return
        order_idx = self.rankings[method][0]
        kfold = int(self.kfold_inc.get())
        self.log("Running incremental CV accuracy... (may take a while)")
        accs = incremental_cv_accuracy(self.F, self.y, order_idx, cv=kfold, model=LogisticRegression(max_iter=2000))
        # plot accs
        self.ax_sel.clear()
        self.ax_sel.plot(np.arange(1, len(accs)+1), accs, '-o')
        self.ax_sel.set_xlabel('#features')
        self.ax_sel.set_ylabel('CV accuracy')
        self.ax_sel.set_title('Incremental CV accuracy')

        # --- دقیق کردن محور X ---
        self.ax_sel.set_xticks(np.arange(1, len(accs)+1))   # یک tick برای هر تعداد ویژگی
        self.ax_sel.set_xticklabels([str(i) for i in range(1, len(accs)+1)], rotation=45)  # چرخش برای خوانایی

        self.ax_sel.grid(True, linestyle='--', alpha=0.5)
        self.canvas_sel.draw()
        self.log("Incremental done. Use Selection tab to pick top features.")

    def show_boxplots(self):
        sel = self._get_selected_indices()
        if self.F is None:
            messagebox.showwarning("No features", "Extract features first")
            return
        plot_boxplots(self.ax_feats, self.F, self.y, self.feat_names, sel)
        self.canvas_feats.draw()

    def show_pca(self):
        sel = self._get_selected_indices()
        plot_pca(self.ax_sel, self.F, self.y, sel)
        self.canvas_sel.draw()

    def show_tsne_dialog(self):
        sel = self._get_selected_indices()
        if self.F is None:
            messagebox.showwarning("No features", "Extract features first")
            return
        top = tk.Toplevel(self)
        top.title("t-SNE parameters")
        ttk.Label(top, text="Perplexity:").grid(row=0,column=0); pvar = tk.IntVar(value=30)
        ttk.Spinbox(top, from_=5, to=50, textvariable=pvar, width=6).grid(row=0,column=1)
        ttk.Label(top, text="Iterations:").grid(row=1,column=0); iv = tk.IntVar(value=800)
        ttk.Spinbox(top, from_=250, to=5000, textvariable=iv, width=8).grid(row=1,column=1)
        def run():
            plot_tsne(self.ax_sel, self.F, self.y, sel, perplexity=pvar.get(), n_iter=iv.get())
            self.canvas_sel.draw(); top.destroy()
        ttk.Button(top, text="Run t-SNE", command=run).grid(row=2,column=0,columnspan=2, pady=6)

    def train_and_compare(self):
        sel = self._get_selected_indices()
        if len(sel)==0:
            messagebox.showwarning("No features", "Select features first")
            return
        X = self.F[:, sel]; y = self.y
        # standardize
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        self.scaler = scaler
        # holdout split
        hold = float(self.holdout_var.get())
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=hold, stratify=y, random_state=RANDOM_SEED)
        # selected models
        sel_models = [m for m,v in self.model_vars.items() if v.get()]
        if not sel_models:
            messagebox.showwarning("No models", "Select models to train")
            return
        kfold = int(self.kfold_var.get())
        self.log("Starting cross-validation on training set...")
        cvres = compare_models_cv(Xtr, ytr, sel_models, cv=kfold)
        # fit final
        self.fitted_models = {}
        test_metrics = {}
        for name in sel_models:
            clf = build_model(name); clf.fit(Xtr, ytr)
            self.fitted_models[name] = clf
            ypred = clf.predict(Xte)
            acc = accuracy_score(yte, ypred)
            try:
                aucv = roc_auc_score((yte==2).astype(int), clf.predict_proba(Xte)[:,1])
            except Exception:
                aucv = float('nan')
            test_metrics[name] = {'test_acc': float(acc), 'test_auc': float(aucv)}
        # display results
        self.txt_results.delete(1.0, tk.END)
        self.txt_results.insert(tk.END, "Cross-val on train (mean ± std):\n")
        for n,r in cvres.items():
            self.txt_results.insert(tk.END, f"{n}: {r['mean_acc']:.4f} ± {r['std_acc']:.4f}\n")
        self.txt_results.insert(tk.END, "\nTest results:\n")
        for n,r in test_metrics.items():
            self.txt_results.insert(tk.END, f"{n}: test_acc={r['test_acc']:.4f}, test_auc={r['test_auc']:.4f}\n")
        self.last_compare = {'cv': cvres, 'test': test_metrics}
        # best model
        best = max(cvres.items(), key=lambda kv: kv[1]['mean_acc'])[0]
        self.best_model_name = best
        self.log(f"Training complete. Best by CV: {best}")
        # plot confusion for best
        self.ax_train.clear()
        plot_confusion(self.ax_train, yte, self.fitted_models[best].predict(Xte))
        self.canvas_train.draw()
        # training finished

    def show_roc_conf(self):
        if not hasattr(self, 'best_model_name') or self.best_model_name is None:
            messagebox.showwarning("No model", "Train models first")
            return
        best = self.best_model_name
        clf = self.fitted_models[best]
        # prepare test set from last training
        sel = self._get_selected_indices()
        X = self.F[:, sel]; Xs = self.scaler.transform(X)
        hold = float(self.holdout_var.get())
        _, Xte, _, yte = train_test_split(Xs, self.y, test_size=hold, stratify=self.y, random_state=RANDOM_SEED)
        # ROC left, Conf right
        self.ax_test.clear(); self.ax_test2.clear()
        aucv = plot_roc(self.ax_test, clf, Xte, yte)
        plot_confusion(self.ax_test2, yte, clf.predict(Xte))
        self.canvas_test.draw()
        self.log(f"ROC plotted for {best} (AUC={aucv})")

    def test_on_random(self):
        if self.X1 is None or self.X2 is None or self.F is None or not hasattr(self,'fitted_models'):
            messagebox.showwarning("Missing", "Load/extract/train first")
            return

        ntest = 8
        L = self.X1.shape[1]
        testsigs = []
        true = []
        for i in range(ntest):
            cls = 1 if i < ntest//2 else 2
            proto = self.X1[np.random.randint(0, self.X1.shape[0])] if cls==1 else self.X2[np.random.randint(0, self.X2.shape[0])]
            sig = proto + 0.05*np.random.randn(L)
            testsigs.append(sig)
            true.append(cls)
        testsigs = np.vstack(testsigs)

        # Extract features and select
        Ftest, _ = extract_features_matrix(testsigs, fs=1.0)
        sel = self._get_selected_indices()
        Xtest = Ftest[:, sel]
        if hasattr(self,'scaler') and self.scaler is not None:
            Xtest = self.scaler.transform(Xtest)

        # Predict
        best = getattr(self,'best_model_name', None)
        if best is None:
            messagebox.showwarning("No model", "Train models first")
            return
        clf = self.fitted_models[best]
        preds = clf.predict(Xtest)

        # --- Plot signals in separate subplots ---
        fig, axes = plt.subplots(ntest, 1, figsize=(12, 2*ntest), sharex=True)
        colors = ['tab:blue', 'tab:orange']
        for i in range(ntest):
            ax = axes[i]
            ax.plot(testsigs[i,:], color=colors[true[i]-1], alpha=0.7, label=f"Class {true[i]} {true[i] == preds[i]}")
            ax.scatter(np.arange(L), testsigs[i,:], c=[colors[p-1] for p in [preds[i]]*L], s=10, marker='x', label=f"Pred: {preds[i]}")
            ax.set_ylabel(f"S{i+1}")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True)
        axes[-1].set_xlabel("Sample index")
        fig.tight_layout()

        # Display in a new window
        win = tk.Toplevel(self)
        win.title("Random Signals Test - Individual Subplots")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

        # Log
        self.log(f"Random test (best={best}) preds: {preds.tolist()} true: {true}")



    def save_report_and_model(self):
        if self.F is None:
            messagebox.showwarning("Nothing to save", "Extract and train first")
            return
        outdir = filedialog.askdirectory(title="Select folder to save report & model")
        if not outdir:
            return
        rep = {
            'timestamp': datetime.datetime.now().isoformat(),
            'mat_file': self.mat_path,
            'data_shapes': {'X1': size_str(self.X1),'X2': size_str(self.X2)},
            'features': self.feat_names,
            'selected_features': self._get_selected_indices(),
            'cv_results': self.last_compare.get('cv', {}),
            'test_results': self.last_compare.get('test', {})
        }
        rep_path = os.path.join(outdir, 'signal_report.json')
        with open(rep_path, 'w') as f:
            json.dump(rep, f, indent=2)
        # save best model
        if self.best_model_name:
            model_path = os.path.join(outdir, f"model_{self.best_model_name}.joblib")
            joblib.dump(self.fitted_models[self.best_model_name], model_path)
            if hasattr(self,'scaler') and self.scaler is not None:
                joblib.dump(self.scaler, os.path.join(outdir,'scaler.joblib'))
        else:
            model_path = None
        messagebox.showinfo("Saved", f"Report saved: {rep_path}\nModel: {model_path if model_path else 'none'}")
        self.log(f"Saved report to {rep_path}; model: {model_path}")

    def log(self, s):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] {s}")
        # append to results widget if exists
        try:
            self.txt_results.insert(tk.END, f"[{t}] {s}\n")
            self.txt_results.see(tk.END)
        except:
            pass

# -------------------- Run --------------------
def main():
    app = SignalApp()
    app.mainloop()

if __name__ == "__main__":
    main()
