import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import scipy.io as sio
import scipy.stats as stats
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.svm import SVC
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import random
import os

# ==============================
# Feature extraction
# ==============================
time_features = ['Mean','Variance','Skewness','Kurtosis','ZCR','ShannonEntropy']
freq_features = ['TotalEnergy','SpectralEntropy','Band1','Band2','Band3']
all_features = time_features + freq_features

def signal_power(sig):
    freqs, psd = signal.welch(sig, nperseg=256)
    freqs = freqs / np.max(freqs)
    return freqs, psd

def extract_features(sig):
    feats = []
    # Time-domain
    feats.append(np.mean(sig))
    feats.append(np.var(sig))
    feats.append(stats.skew(sig))
    feats.append(stats.kurtosis(sig))
    zcr = ((sig[:-1]*sig[1:])<0).sum()
    feats.append(zcr/len(sig))
    hist,_ = np.histogram(sig,bins=50,density=True)
    hist = hist[hist>0]
    feats.append(-np.sum(hist*np.log2(hist)))
    # Frequency-domain
    freqs, psd = signal_power(sig)
    feats.append(np.sum(psd))
    feats.append(stats.entropy(psd))
    total_energy = np.sum(psd)
    bands = [(0,0.1),(0.1,0.3),(0.3,0.5)]
    for low,high in bands:
        band_energy = np.sum(psd[(freqs>=low)&(freqs<high)])
        feats.append(band_energy/total_energy if total_energy>0 else 0)
    return np.array(feats)

# ==============================
# Dashboard
# ==============================
class SignalDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Feature Classification Dashboard")
        self.X_signals = None
        self.y = None
        self.features = None
        self.selected_features = []
        self.model_trained = False
        self.trained_model = None
        self.scaler = None

        # --- Frames ---
        top_frame = tk.Frame(root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        left_frame = tk.Frame(root)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Top Frame ---
        tk.Button(top_frame, text="Load MATLAB File", command=self.load_file, bg='blue', fg='white').pack(side=tk.LEFT)
        self.file_label = tk.Label(top_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # --- Left Frame: Feature Selection ---
        tk.Label(left_frame, text="Available Features").pack()
        self.feature_listbox = tk.Listbox(left_frame, selectmode=tk.MULTIPLE, height=15)
        for f in all_features: self.feature_listbox.insert(tk.END, f)
        self.feature_listbox.pack()
        tk.Button(left_frame, text="Add Selected Features", command=self.add_features).pack(pady=5)
        tk.Button(left_frame, text="Clear Features", command=self.clear_features).pack(pady=5)
        tk.Button(left_frame, text="Auto Select Top Features", command=self.auto_select_features).pack(pady=5)
        tk.Button(left_frame, text="Incremental Feature Selection", command=self.incremental_feature_selection).pack(pady=5)
        tk.Label(left_frame, text="Selected Features:").pack(pady=(20,0))
        self.selected_label = tk.Label(left_frame, text="")
        self.selected_label.pack()
        tk.Button(left_frame, text="Random Features Boxplots", command=self.random_boxplots,bg='yellow').pack(pady=5)
        tk.Button(left_frame, text="Train Model", command=self.train_model, bg='green', fg='white').pack(pady=5)
        tk.Label(left_frame, text="Test a Signal:").pack(pady=(20,0))
        tk.Button(left_frame, text="Test Random Signal", command=self.test_random_signal, bg='orange').pack(pady=5)

        # --- Right Frame: Tabs ---
        self.tabs = ttk.Notebook(right_frame)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self.tab_data = tk.Frame(self.tabs); self.tabs.add(self.tab_data, text="Data Summary")
        self.tab_feature = tk.Frame(self.tabs); self.tabs.add(self.tab_feature, text="Feature Table")
        self.tab_plots = tk.Frame(self.tabs); self.tabs.add(self.tab_plots, text="Plots")
        self.tab_class = tk.Frame(self.tabs); self.tabs.add(self.tab_class, text="Classification Results")
        self.tab_test = tk.Frame(self.tabs); self.tabs.add(self.tab_test, text="Test Signal")

    # ==============================
    # Load MATLAB file
    # ==============================
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MATLAB files","*.mat")])
        if not file_path: return
        self.file_label.config(text=os.path.basename(file_path))
        mat_data = sio.loadmat(file_path)
        if 'X1' not in mat_data or 'X2' not in mat_data:
            messagebox.showerror("Error","MATLAB file must contain X1 and X2 matrices!")
            return
        X1 = mat_data['X1']; X2 = mat_data['X2']
        self.X_signals = np.vstack((X1,X2))
        labels1 = np.zeros(X1.shape[0]); labels2 = np.ones(X2.shape[0])
        self.y = np.hstack((labels1, labels2))
        self.features = np.array([extract_features(s) for s in self.X_signals])
        self.show_data_summary(X1, X2)

    def show_data_summary(self, X1, X2):
        for widget in self.tab_data.winfo_children(): widget.destroy()
        tk.Label(self.tab_data, text=f"Class1: {X1.shape[0]} samples, length {X1.shape[1]}").pack()
        tk.Label(self.tab_data, text=f"Class2: {X2.shape[0]} samples, length {X2.shape[1]}").pack()
        tk.Label(self.tab_data, text="Select Signal to View:").pack(pady=(10,0))
        # Sample combobox
        self.signal_selector = ttk.Combobox(self.tab_data, values=[f"C1-{i}" for i in range(X1.shape[0])] +
                                           [f"C2-{i}" for i in range(X2.shape[0])])
        self.signal_selector.pack()
        tk.Button(self.tab_data, text="Show Signal", command=self.show_selected_signal).pack(pady=5)
        messagebox.showinfo("Info","File loaded and features extracted!")

    def show_selected_signal(self):
        idx_str = self.signal_selector.get()
        if not idx_str: return
        if idx_str.startswith("C1-"):
            idx = int(idx_str.split("-")[1])
        else:
            idx = int(idx_str.split("-")[1]) + np.sum(self.y==0)
        sig = self.X_signals[idx]
        for widget in self.tab_data.winfo_children():
            if isinstance(widget, tk.Frame): widget.destroy()
        frame = tk.Frame(self.tab_data)
        frame.pack(fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(sig)
        ax.set_title(f"Signal {idx_str}")
        ax.set_xlabel("Sample"); ax.set_ylabel("Amplitude")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack()

    # ==============================
    # Feature selection
    # ==============================
    def add_features(self):
        sel = [all_features[i] for i in self.feature_listbox.curselection()]
        for f in sel:
            if f not in self.selected_features: self.selected_features.append(f)
        self.update_selected_label()
    def clear_features(self):
        self.selected_features = []; self.update_selected_label()
    def auto_select_features(self):
        self.selected_features = all_features[:6]; self.update_selected_label()
    def update_selected_label(self):
        self.selected_label.config(text=", ".join(self.selected_features))

    def incremental_feature_selection(self):
        if self.features is None or self.y is None:
            messagebox.showwarning("Warning", "Load a MATLAB file first!")
            return

        # یک بار فقط داده رو split می‌کنیم
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.y, test_size=0.3, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        accuracies = []
        temp_features = []
        self.selected_features = []

        for f in all_features:
            temp_features.append(all_features.index(f))
            X_train_sel = X_train[:, temp_features]
            X_test_sel = X_test[:, temp_features]

            model = SVC(kernel='linear', probability=True, random_state=42)
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            self.selected_features.append(f)

        # ساختن لیست کامل از فیچرها + دقت‌ها
        results = "\n".join([f"{f}: Accuracy={a:.3f}" for f, a in zip(self.selected_features, accuracies)])

        # پیدا کردن بهترین دقت
        best_idx = int(np.argmax(accuracies))
        best_feat = self.selected_features[:best_idx+1]
        best_acc = accuracies[best_idx]

        messagebox.showinfo(
            "Incremental Feature Selection",
            results + f"\n\nBest Accuracy = {best_acc:.3f} with features: {', '.join(best_feat)}"
        )

        self.update_selected_label()




    # ==============================
    # Train Model
    # ==============================
    def train_model(self):
        if self.features is None or self.y is None:
            messagebox.showwarning("Warning","Load a MATLAB file first!")
            return
        if len(self.selected_features)==0:
            messagebox.showwarning("Warning","Select at least one feature!")
            return
        idx = [all_features.index(f) for f in self.selected_features]
        X_sel = self.features[:,idx]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, self.y, test_size=0.3, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.trained_model = SVC(kernel='rbf', probability=True)
        self.trained_model.fit(X_train_scaled, y_train)
        self.model_trained = True
        self.plot_results(idx, X_sel, X_test_scaled, y_test)
        self.show_feature_table()
        messagebox.showinfo("Info","Model trained successfully!")

    def plot_results(self, idx, X_sel, X_test_scaled, y_test):
        for tab in [self.tab_plots, self.tab_class]:
            for widget in tab.winfo_children(): widget.destroy()
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sel)
        frame = tk.Frame(self.tab_plots)
        frame.pack(fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.scatter(X_pca[self.y==0,0], X_pca[self.y==0,1], label='Class1', alpha=0.7)
        ax.scatter(X_pca[self.y==1,0], X_pca[self.y==1,1], label='Class2', alpha=0.7)
        ax.set_title('PCA Projection'); ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, frame).update()
        # Boxplots 5-feature
        frame2 = tk.Frame(self.tab_plots)
        frame2.pack(fill=tk.BOTH, expand=True)
        if len(idx)>5: idx_to_plot = random.sample(idx,5)
        else: idx_to_plot = idx
        fig2, axes = plt.subplots(1,len(idx_to_plot), figsize=(4*len(idx_to_plot),4))
        if len(idx_to_plot)==1: axes=[axes]
        for i,j in enumerate(idx_to_plot):
            axes[i].boxplot([self.features[self.y==0,j], self.features[self.y==1,j]], labels=['Class1','Class2'])
            axes[i].set_title(all_features[j])
        canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
        canvas2.draw(); canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas2, frame2).update()
        # Confusion Matrix
        y_pred = self.trained_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        fig3, ax3 = plt.subplots(figsize=(4,4))
        ConfusionMatrixDisplay(cm, display_labels=['Class1','Class2']).plot(ax=ax3)
        frame3 = tk.Frame(self.tab_class)
        frame3.pack(fill=tk.BOTH, expand=True)
        canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
        canvas3.draw(); canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas3, frame3).update()
        # ROC Curve
        y_prob = self.trained_model.predict_proba(X_test_scaled)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig4, ax4 = plt.subplots(figsize=(4,4))
        ax4.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
        ax4.plot([0,1],[0,1],'--',color='gray')
        ax4.set_xlabel('FPR'); ax4.set_ylabel('TPR'); ax4.set_title('ROC Curve'); ax4.legend()
        frame4 = tk.Frame(self.tab_class)
        frame4.pack(fill=tk.BOTH, expand=True)
        canvas4 = FigureCanvasTkAgg(fig4, master=frame4)
        canvas4.draw(); canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas4, frame4).update()


    def random_boxplots(self):
        if self.features is None: return
        num = 3
        idx_to_plot = random.sample(range(len(all_features)), num)
        for widget in self.tab_plots.winfo_children(): widget.destroy()
        frame = tk.Frame(self.tab_plots); frame.pack(fill=tk.BOTH, expand=True)
        fig, axes = plt.subplots(1, num, figsize=(4*num,4))
        if num==1: axes=[axes]
        for i,j in enumerate(idx_to_plot):
            axes[i].boxplot([self.features[self.y==0,j], self.features[self.y==1,j]], labels=['Class1','Class2'])
            axes[i].set_title(all_features[j])
        canvas = FigureCanvasTkAgg(fig, master=frame); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, frame).update()
        
    # ==============================
    # Feature Table
    # ==============================
    def show_feature_table(self):
        for widget in self.tab_feature.winfo_children(): widget.destroy()
        container = tk.Frame(self.tab_feature)
        container.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(container, columns=all_features, show='headings')
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side='right', fill='y'); hsb.pack(side='bottom', fill='x'); tree.pack(fill=tk.BOTH, expand=True)
        for f in all_features: tree.heading(f, text=f)
        for row in self.features: tree.insert("", tk.END, values=list(row))

    # ==============================
    # Test Random Signal
    # ==============================
    def test_random_signal(self):
        if not self.model_trained:
            messagebox.showwarning("Warning","Train the model first!")
            return
        sig_len = 1024
        choice = np.random.choice([
            'gaussian', 'uniform_noise', 'sine', 'cosine', 'square', 
            'sawtooth', 'chirp', 'composite', 'impulse', 'brownian', 'mix'
        ])

        t = np.linspace(0, 1, sig_len)

        if choice == 'gaussian':
            test_sig = np.random.randn(sig_len)
        elif choice == 'uniform_noise':
            test_sig = np.random.rand(sig_len) - 0.5
        elif choice == 'sine':
            freq = np.random.choice([3, 5, 10, 20])
            phase = np.random.rand() * 2 * np.pi
            amp = np.random.uniform(0.5, 2.0)
            test_sig = amp * np.sin(2 * np.pi * freq * t + phase)
        elif choice == 'cosine':
            freq = np.random.choice([3, 7, 15])
            phase = np.random.rand() * 2 * np.pi
            amp = np.random.uniform(0.5, 2.0)
            test_sig = amp * np.cos(2 * np.pi * freq * t + phase)
        elif choice == 'square':
            freq = np.random.choice([5, 12])
            test_sig = signal.square(2 * np.pi * freq * t)
        elif choice == 'sawtooth':
            freq = np.random.choice([5, 15])
            test_sig = signal.sawtooth(2 * np.pi * freq * t)
        elif choice == 'chirp':
            f0, f1 = np.random.choice([2, 5]), np.random.choice([15, 25])
            test_sig = signal.chirp(t, f0=f0, f1=f1, t1=1, method='linear')
        elif choice == 'composite':
            test_sig = (np.sin(2*np.pi*5*t) + np.cos(2*np.pi*15*t)) / 2
        elif choice == 'impulse':
            test_sig = np.zeros(sig_len)
            test_sig[np.random.randint(0, sig_len)] = 1
        elif choice == 'brownian':
            test_sig = np.cumsum(np.random.randn(sig_len))
        elif choice == 'mix':
            sub_choices = np.random.choice(
                ['sine', 'cosine', 'square', 'sawtooth', 'chirp', 'gaussian'],
                size=np.random.randint(2, 5), replace=False
            )
            signals_list = []
            for sc in sub_choices:
                if sc == 'sine':
                    freq = np.random.choice([4, 8, 16])
                    phase = np.random.rand() * 2 * np.pi
                    signals_list.append(np.sin(2*np.pi*freq*t + phase))
                elif sc == 'cosine':
                    freq = np.random.choice([3, 6, 12])
                    phase = np.random.rand() * 2 * np.pi
                    signals_list.append(np.cos(2*np.pi*freq*t + phase))
                elif sc == 'square':
                    freq = np.random.choice([5, 10])
                    signals_list.append(signal.square(2*np.pi*freq*t))
                elif sc == 'sawtooth':
                    freq = np.random.choice([6, 14])
                    signals_list.append(signal.sawtooth(2*np.pi*freq*t))
                elif sc == 'chirp':
                    signals_list.append(signal.chirp(t, f0=1, f1=20, t1=1, method='quadratic'))
                elif sc == 'gaussian':
                    signals_list.append(np.random.randn(sig_len) * 0.3)

            weights = np.random.rand(len(signals_list))
            weights /= np.sum(weights)
            test_sig = np.sum([w*s for w, s in zip(weights, signals_list)], axis=0)

        feats = extract_features(test_sig)
        idx = [all_features.index(f) for f in self.selected_features]
        X_test_feat = feats[idx].reshape(1, -1)
        X_test_scaled = self.scaler.transform(X_test_feat)
        pred_class = self.trained_model.predict(X_test_scaled)[0]


        for widget in self.tab_test.winfo_children(): widget.destroy()
        tk.Label(self.tab_test, text=f"Predicted Class: {int(pred_class)+1}").pack()
        tk.Label(self.tab_test, text=f"Selected Feature values: {feats[idx]}").pack()
        frame = tk.Frame(self.tab_test)
        frame.pack(fill=tk.BOTH, expand=True)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(test_sig); ax.set_title(f"Test Signal ({choice})"); ax.set_xlabel("Sample"); ax.set_ylabel("Amplitude")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, frame).update()

# ==============================
# Run App
# ==============================
root = tk.Tk()
app = SignalDashboard(root)
root.mainloop()
