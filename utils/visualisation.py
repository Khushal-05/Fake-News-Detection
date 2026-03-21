
"""
Visualization utilities for fake news detection model evaluation
"""
import argparse
import glob
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelVisualizer:
    """Visualization tools for model evaluation"""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize visualizer
        
        Args:
            class_names: Names of classes (e.g., ['Fake', 'Real'])
        """
        self.class_names = class_names if class_names else ['Class 0', 'Class 1']
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title += ' (Normalized)'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
            ax=ax
        )
        
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_per_language_performance(
        self,
        per_lang_metrics: Dict[str, Dict[str, float]],
        metrics_to_plot: List[str] = None,
        title: str = 'Per-Language Performance',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot per-language performance comparison
        
        Args:
            per_lang_metrics: Dictionary with per-language metrics
            metrics_to_plot: List of metrics to plot
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        """
        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Prepare data
        languages = list(per_lang_metrics.keys())
        data = {metric: [] for metric in metrics_to_plot}
        
        for lang in languages:
            for metric in metrics_to_plot:
                data[metric].append(per_lang_metrics[lang][metric])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(languages))
        width = 0.8 / len(metrics_to_plot)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_to_plot)))
        
        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            offset = width * i - (width * len(metrics_to_plot) / 2 - width / 2)
            bars = ax.bar(
                x + offset,
                data[metric],
                width,
                label=metric.replace('_', ' ').title(),
                color=color,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,  # <-- comma here
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

                # ax.text(
                #     bar.get_x() + bar.get_width() / 2.,
                #     height,
                #     f'{height:.3f}',
                #     ha='center',
                #     va='bottom',
                #     fontsize=8
                # )
        
        ax.set_xlabel('Language', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([lang.upper() for lang in languages], rotation=0)
        ax.legend(loc='lower right', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Per-language performance plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        metrics: List[str] = None,
        title: str = 'Training History',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Plot training history
        
        Args:
            history: Dictionary with training history
            metrics: Metrics to plot
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
        
        if num_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in history:
                epochs = range(1, len(history[train_key]) + 1)
                ax.plot(epochs, history[train_key], 'b-o', label=f'Train {metric}', linewidth=2)
            
            if val_key in history:
                epochs = range(1, len(history[val_key]) + 1)
                ax.plot(epochs, history[val_key], 'r-s', label=f'Val {metric}', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} over Epochs', fontsize=12, fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training history plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'ROC Curve',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot ROC curve for binary classification
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})'
        )
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_class_distribution(
        self,
        labels: np.ndarray,
        languages: Optional[List[str]] = None,
        title: str = 'Class Distribution',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot class distribution overall and per language
        
        Args:
            labels: Label array
            languages: Language codes (optional)
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        """
        if languages is None:
            # Simple bar plot
            fig, ax = plt.subplots(figsize=(8, 6))
            unique, counts = np.unique(labels, return_counts=True)
            
            bars = ax.bar(
                [self.class_names[i] for i in unique],
                counts,
                color=['#ff6b6b', '#4ecdc4'],
                alpha=0.8,
                edgecolor='black'
            )
            
            # Add value labels (counts are integers)
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

            
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3)
            
        else:
            # Stacked bar plot per language
            df = pd.DataFrame({'label': labels, 'language': languages})
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Count per language and class
            counts = df.groupby(['language', 'label']).size().unstack(fill_value=0)
            
            counts.plot(
                kind='bar',
                stacked=False,
                ax=ax,
                color=['#ff6b6b', '#4ecdc4'],
                alpha=0.8,
                edgecolor='black',
                width=0.7
            )
            
            ax.set_xlabel('Language', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(self.class_names, title='Class', frameon=True, shadow=True)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Class distribution plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_prediction_confidence_distribution(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'Prediction Confidence Distribution',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot distribution of prediction confidence scores
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities [n_samples, n_classes]
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        """
        # Get max probability (confidence)
        y_pred_proba = np.max(y_proba, axis=1)
        y_pred = np.argmax(y_proba, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (y_pred == y_true)
        correct_proba = y_pred_proba[correct_mask]
        incorrect_proba = y_pred_proba[~correct_mask]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(correct_proba, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
        axes[0].hist(incorrect_proba, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        axes[0].set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        axes[0].legend(frameon=True, shadow=True)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Box plot
        data = [correct_proba, incorrect_proba]
        bp = axes[1].boxplot(
            data,
            labels=['Correct', 'Incorrect'],
            patch_artist=True,
            notch=True,
            showmeans=True
        )
        
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1].set_ylabel('Prediction Confidence', fontsize=11, fontweight='bold')
        axes[1].set_title('Confidence Box Plot', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confidence distribution plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def create_evaluation_dashboard(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        languages: Optional[List[str]] = None,
        per_lang_metrics: Optional[Dict] = None,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive evaluation dashboard
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            languages: Language codes
            per_lang_metrics: Per-language metrics
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix', fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Normalized Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title('Normalized Confusion Matrix', fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax3.plot([0, 1], [0, 1], 'k--', lw=2)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve', fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.grid(alpha=0.3)
        
        # 4. Class Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        unique, counts = np.unique(y_true, return_counts=True)
        ax4.bar([self.class_names[i] for i in unique], counts, 
                color=['#ff6b6b', '#4ecdc4'], alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Count')
        ax4.set_title('Class Distribution', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Confidence Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        y_pred_proba = np.max(y_proba, axis=1)
        y_pred_calc = np.argmax(y_proba, axis=1)
        correct_mask = (y_pred_calc == y_true)
        ax5.hist(y_pred_proba[correct_mask], bins=20, alpha=0.7, label='Correct', color='green')
        ax5.hist(y_pred_proba[~correct_mask], bins=20, alpha=0.7, label='Incorrect', color='red')
        ax5.set_xlabel('Prediction Confidence')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Confidence Distribution', fontweight='bold')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Per-Language Performance (if available)
        if per_lang_metrics:
            ax6 = fig.add_subplot(gs[1, 2])
            langs = list(per_lang_metrics.keys())
            accs = [per_lang_metrics[lang]['accuracy'] for lang in langs]
            ax6.barh(langs, accs, color=plt.cm.viridis(np.linspace(0, 1, len(langs))), 
                     alpha=0.8, edgecolor='black')
            ax6.set_xlabel('Accuracy')
            ax6.set_title('Per-Language Accuracy', fontweight='bold')
            ax6.grid(axis='x', alpha=0.3)
            for i, (lang, acc) in enumerate(zip(langs, accs)):
                ax6.text(acc, i, f'{acc:.3f}', va='center', ha='left', fontweight='bold')
        
        # 7-9. Metrics Summary (Text)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        metrics_text = f"""
        OVERALL PERFORMANCE METRICS
        
        Accuracy:           {acc:.4f}
        Precision (Weighted): {prec:.4f}
        Recall (Weighted):    {rec:.4f}
        F1-Score (Weighted):  {f1:.4f}
        ROC-AUC:             {roc_auc:.4f}
        
        Total Samples:       {len(y_true)}
        Correct Predictions: {np.sum(y_pred == y_true)} ({np.sum(y_pred == y_true)/len(y_true)*100:.2f}%)
        Incorrect Predictions: {np.sum(y_pred != y_true)} ({np.sum(y_pred != y_true)/len(y_true)*100:.2f}%)
        """
        
        ax7.text(0.5, 0.5, metrics_text, ha='center', va='center', 
                fontsize=11, family='monospace', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Evaluation dashboard saved to {save_path}")
        
        plt.show()
        plt.close()

def find_latest_run(runs_root="outputs/runs"):
    if not os.path.exists(runs_root):
        raise FileNotFoundError(f"No runs root found at {runs_root}")
    runs = sorted(glob.glob(os.path.join(runs_root, "*")), key=os.path.getmtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No runs found in {runs_root}")
    return runs[0]

def load_run(run_dir):
    """Load history and predictions from a run folder. Returns numpy arrays for numeric data."""
    # load history
    hist_path = os.path.join(run_dir, "history.json")
    history = None
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as fh:
            history = json.load(fh)

    # load predictions (prefer npz)
    npz_path = os.path.join(run_dir, "predictions.npz")
    if os.path.exists(npz_path):
        npz = np.load(npz_path, allow_pickle=True)
        y_true = npz["y_true"].tolist() if "y_true" in npz else []
        y_pred = npz["y_pred"].tolist() if "y_pred" in npz else []
        y_proba = npz["y_proba"]           # keep as numpy array (shape: N x C)
        languages = npz["languages"].tolist() if "languages" in npz else []
    else:
        csv_path = os.path.join(run_dir, "predictions.csv")
        if os.path.exists(csv_path):
            import csv
            y_true, y_pred, y_proba, languages = [], [], [], []
            with open(csv_path, newline="", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                for r in reader:
                    y_true.append(int(r["y_true"]))
                    y_pred.append(int(r["y_pred"]))
                    y_proba.append(float(r["y_proba"]))
                    languages.append(r.get("language", ""))
            # convert scalar-prob list to column vector shape (N,1)
            y_proba = np.array(y_proba)[:, None] if len(y_proba) > 0 else np.zeros((0, 1))
        else:
            raise FileNotFoundError(f"No predictions found in {run_dir}")

    # Normalize types: ensure numpy arrays for numeric ops
    y_true = np.array(y_true, dtype=int) if len(y_true) > 0 else np.array([], dtype=int)
    y_pred = np.array(y_pred, dtype=int) if len(y_pred) > 0 else np.array([], dtype=int)
    y_proba = np.array(y_proba)
    if y_proba.ndim == 1:
        y_proba = y_proba[:, None]
    languages = list(languages)

    return history, y_true, y_pred, y_proba, languages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", "-r", type=str, default=None, help="Path to run folder (overrides auto-latest)")
    parser.add_argument("--runs-root", type=str, default="outputs/runs", help="Root containing run folders")
    parser.add_argument("--save-path", type=str, default=None, help="Optional save path for dashboard image")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(args.runs_root)

    print("Visualising run:", run_dir)
    history, y_true, y_pred, y_proba, languages = load_run(run_dir)

    # instantiate your ModelVisualizer (use local class if present)
    MV = globals().get("ModelVisualizer", None)
    if MV is None:
        tried = []
        for mod_path in ("visualisation", "utils.visualisation"):
            try:
                m = __import__(mod_path, fromlist=["ModelVisualizer"])
                MV = getattr(m, "ModelVisualizer", None)
                if MV:
                    break
            except Exception as e:
                tried.append((mod_path, str(e)))
    if MV is None:
        raise RuntimeError(
            "ModelVisualizer class not found. Define it in this file or make it importable as visualisation.ModelVisualizer. "
            f"Tried: {tried}"
        )

    try:
        viz = MV()
    except TypeError:
        viz = MV(class_names=['Fake', 'Real'])

    save_path = args.save_path or os.path.join(run_dir, "dashboard.png")
    viz.create_evaluation_dashboard(y_true=y_true,
                                    y_pred=y_pred,
                                    y_proba=y_proba,
                                    languages=languages,
                                    per_lang_metrics=None,
                                    save_path=save_path)
    
    # --- also save individual plots (7 images) ---
    def safe_call(name, *a, filename=None, **kw):
        if hasattr(viz, name):
            try:
                if filename:
                    kw['save_path'] = os.path.join(run_dir, filename)
                getattr(viz, name)(*a, **kw)
                print(f"Saved: {filename or name}")
            except Exception as e:
                print(f"Warning: {name} failed -> {e}")
        else:
            print(f"Skipping {name} (not implemented)")

    # build per-language metrics if needed
    per_lang_metrics = None
    if len(languages) > 0:
        per_lang_metrics = {}
        unique_langs = sorted(set(languages))
        for lang in unique_langs:
            idxs = [i for i, L in enumerate(languages) if L == lang]
            if not idxs:
                continue
            yt = np.array(y_true)[idxs]
            yp = np.array(y_pred)[idxs]
            acc = (yt == yp).mean() if len(yt) > 0 else 0.0
            # compute precision/recall/f1 per-language using sklearn (safe fallback)
            try:
                from sklearn.metrics import precision_recall_fscore_support
                p, r, f, _ = precision_recall_fscore_support(yt, yp, average='weighted', zero_division=0)
            except Exception:
                p = r = f = 0.0
            per_lang_metrics[lang] = {'accuracy': float(acc), 'precision': float(p), 'recall': float(r), 'f1_score': float(f)}

    # 1. Confusion matrix (raw)
    safe_call('plot_confusion_matrix', y_true, y_pred, normalize=False, filename='cm_raw.png')
    # 2. Confusion matrix (normalized)
    safe_call('plot_confusion_matrix', y_true, y_pred, normalize=True, filename='cm_normalized.png')
    # 3. ROC curve (binary expected; use positive-class probs if present)
    if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
        pos_probs = y_proba[:, 1]
    else:
        pos_probs = y_proba.ravel()
    safe_call('plot_roc_curve', y_true, pos_probs, filename='roc_curve.png')
    # 4. Class distribution
    safe_call('plot_class_distribution', y_true, languages, filename='class_dist.png')
    # 5. Confidence distribution (hist + box)
    safe_call('plot_prediction_confidence_distribution', y_true, y_proba, filename='confidence_dist.png')
    # 6. Per-language performance
    if per_lang_metrics:
        safe_call('plot_per_language_performance', per_lang_metrics, filename='per_lang_perf.png')
    # 7. Training history (if history exists)
    if history:
        safe_call('plot_training_history', history, filename='training_history.png')

    # print("Saved individual plots (if methods available) into", run_dir)

    print("Saved dashboard to", save_path)

# # Example usage — see __main__ block above for the live entry point.
