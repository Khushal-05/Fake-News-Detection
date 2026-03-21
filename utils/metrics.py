"""
Comprehensive metrics computation for fake news detection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    average_precision_score
)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics for classification"""
    
    def __init__(self, num_classes: int = 2, class_names: List[str] = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes (e.g., ['Fake', 'Real'])
        """
        self.num_classes = num_classes
        
        if class_names is None:
            self.class_names = [f"Class_{i}" for i in range(num_classes)]
        else:
            self.class_names = class_names
    
    def compute_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute basic classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        return metrics
    
    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with per-class metrics
        """
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(np.sum(y_true == i))
            }
        
        return per_class_metrics
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: None, 'true', 'pred', or 'all'
        
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
        
        return cm
    
    def compute_roc_auc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute ROC-AUC scores
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities [n_samples, n_classes]
        
        Returns:
            Dictionary with ROC-AUC scores
        """
        roc_metrics = {}
        
        try:
            if self.num_classes == 2:
                # Binary classification
                roc_metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                roc_metrics['avg_precision'] = average_precision_score(y_true, y_proba[:, 1])
            else:
                # Multi-class classification
                roc_metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                )
                roc_metrics['roc_auc_ovo'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovo', average='weighted'
                )
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
            roc_metrics['roc_auc'] = None
        
        return roc_metrics
    
    def compute_per_language_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        languages: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each language
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            languages: List of language codes
        
        Returns:
            Dictionary with per-language metrics
        """
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'language': languages
        })
        
        per_lang_metrics = {}
        
        for lang in df['language'].unique():
            lang_df = df[df['language'] == lang]
            lang_y_true = lang_df['y_true'].values
            lang_y_pred = lang_df['y_pred'].values
            
            if len(lang_y_true) == 0:
                continue
            
            per_lang_metrics[lang] = {
                'accuracy': accuracy_score(lang_y_true, lang_y_pred),
                'precision': precision_score(lang_y_true, lang_y_pred, average='weighted', zero_division=0),
                'recall': recall_score(lang_y_true, lang_y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(lang_y_true, lang_y_pred, average='weighted', zero_division=0),
                'samples': len(lang_y_true)
            }
        
        return per_lang_metrics
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        languages: Optional[List[str]] = None
    ) -> Dict:
        """
        Compute all available metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            languages: Language codes (optional)
        
        Returns:
            Comprehensive metrics dictionary
        """
        all_metrics = {
            'basic_metrics': self.compute_basic_metrics(y_true, y_pred),
            'per_class_metrics': self.compute_per_class_metrics(y_true, y_pred),
            'confusion_matrix': self.compute_confusion_matrix(y_true, y_pred).tolist(),
            'confusion_matrix_normalized': self.compute_confusion_matrix(
                y_true, y_pred, normalize='true'
            ).tolist()
        }
        
        # Add ROC-AUC if probabilities provided
        if y_proba is not None:
            all_metrics['roc_metrics'] = self.compute_roc_auc(y_true, y_proba)
        
        # Add per-language metrics if languages provided
        if languages is not None:
            all_metrics['per_language_metrics'] = self.compute_per_language_metrics(
                y_true, y_pred, languages
            )
        
        return all_metrics
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False
    ):
        """
        Get sklearn classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: Return as dictionary
        
        Returns:
            Classification report
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=output_dict,
            zero_division=0
        )
    
    def print_metrics_summary(self, metrics: Dict):
        """
        Print formatted metrics summary
        
        Args:
            metrics: Metrics dictionary from compute_all_metrics
        """
        print("\n" + "="*70)
        print("METRICS SUMMARY")
        print("="*70)
        
        # Basic metrics
        print("\n Overall Performance:")
        basic = metrics['basic_metrics']
        print(f"  Accuracy:           {basic['accuracy']:.4f}")
        print(f"  Precision (Macro):  {basic['precision_macro']:.4f}")
        print(f"  Recall (Macro):     {basic['recall_macro']:.4f}")
        print(f"  F1-Score (Macro):   {basic['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted):{basic['f1_weighted']:.4f}")
        print(f"  Balanced Accuracy:  {basic['balanced_accuracy']:.4f}")
        print(f"  Cohen's Kappa:      {basic['cohen_kappa']:.4f}")
        print(f"  Matthews Corr:      {basic['matthews_corrcoef']:.4f}")
        
        # Per-class metrics
        print("\n Per-Class Performance:")
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            print(f"\n  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall:    {class_metrics['recall']:.4f}")
            print(f"    F1-Score:  {class_metrics['f1_score']:.4f}")
            print(f"    Support:   {class_metrics['support']}")
        
        # ROC metrics
        if 'roc_metrics' in metrics and metrics['roc_metrics'].get('roc_auc'):
            print("\n📉ROC Metrics:")
            print(f"  ROC-AUC: {metrics['roc_metrics']['roc_auc']:.4f}")
            if 'avg_precision' in metrics['roc_metrics']:
                print(f"  Avg Precision: {metrics['roc_metrics']['avg_precision']:.4f}")
        
        # Per-language metrics
        if 'per_language_metrics' in metrics:
            print("\n Per-Language Performance:")
            for lang, lang_metrics in sorted(
                metrics['per_language_metrics'].items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            ):
                print(f"\n  {lang.upper()} ({lang_metrics['samples']} samples):")
                print(f"    Accuracy:  {lang_metrics['accuracy']:.4f}")
                print(f"    Precision: {lang_metrics['precision']:.4f}")
                print(f"    Recall:    {lang_metrics['recall']:.4f}")
                print(f"    F1-Score:  {lang_metrics['f1_score']:.4f}")
        
        print("\n" + "="*70)


# Example usage
if __name__ == "__main__":
    # Simulate predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.copy(y_true)
    # Add some errors
    error_indices = np.random.choice(1000, 100, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    y_proba = np.random.rand(1000, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    languages = np.random.choice(['hi', 'ta', 'gu', 'mr'], 1000)
    
    # Calculate metrics
    calculator = MetricsCalculator(num_classes=2, class_names=['Fake', 'Real'])
    metrics = calculator.compute_all_metrics(y_true, y_pred, y_proba, languages.tolist())
    
    # Print summary
    calculator.print_metrics_summary(metrics)
    
    # Print classification report
    print("\nClassification Report:")
    print(calculator.get_classification_report(y_true, y_pred))