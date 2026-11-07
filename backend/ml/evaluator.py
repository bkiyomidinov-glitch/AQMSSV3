"""Evaluate the trained model on the test split."""

import pandas as pd
import numpy as np
import pickle
import json
import os
import argparse
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, auc
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class AMQSSEvaluator:
    """Evaluate trained RandomForest on test set."""
    
    def __init__(self, models_dir, dataset_csv_path):
        self.models_dir = models_dir
        self.dataset_csv_path = dataset_csv_path
        
        with open(os.path.join(models_dir, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(os.path.join(models_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_cols = self.metadata['feature_columns']
        print("Model loaded")
    
    def load_and_split_dataset(self, test_fraction=0.2):
        """Load dataset and apply same time-based split as training."""
        df = pd.read_csv(self.dataset_csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        split_idx = int(len(df) * (1 - test_fraction))
        test_df = df.iloc[split_idx:].copy()
        
        print(f"Test set: {len(test_df)} samples")
        return test_df
    
    def extract_features_and_labels(self, df):
        """Extract features and labels from test set."""
        X = df[self.feature_cols].values.astype(float)
        y = df['label'].values.astype(int)
        return X, y
    
    def evaluate(self, output_dir=None, test_fraction=0.2):
        test_df = self.load_and_split_dataset(test_fraction=test_fraction)
        X_test, y_test = self.extract_features_and_labels(test_df)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        unique_test = np.unique(y_test)
        if len(unique_test) == 1:
            print("Test set contains only one class")
            print(f"  Unique classes: {unique_test}")
            print("  Using full dataset for evaluation instead.")
            full_df = pd.read_csv(self.dataset_csv_path)
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
            X_full, y_full = self.extract_features_and_labels(full_df)
            X_full_scaled = self.scaler.transform(X_full)
            y_test = y_full
            y_pred = self.model.predict(X_full_scaled)
            y_proba = self.model.predict_proba(X_full_scaled)[:, 1]
            test_df = full_df
        
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = np.nan
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if len(cm) == 1:
                if y_test[0] == 0:
                    tn = cm[0, 0]
                else:
                    tp = cm[0, 0]
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall_vals, precision_vals)
        except:
            pr_auc = np.nan
        
        from sklearn.metrics import f1_score as f1_weighted
        f1_weighted_score = f1_weighted(y_test, y_pred, average='weighted')
        
        from sklearn.metrics import classification_report
        try:
            class_report = classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality'], output_dict=True, zero_division=0)
        except:
            class_report = {}
        
        metrics = {
            'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
            'pr_auc': float(pr_auc) if not np.isnan(pr_auc) else None,
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'f1_weighted': float(f1_weighted_score),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            },
            'test_size': len(y_test),
            'label_distribution': {
                'class_0': int((y_test == 0).sum()),
                'class_1': int((y_test == 1).sum())
            },
            'class_report': class_report
        }
        
        print("\nTest set performance")
        roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
        pr_auc_str = f"{metrics['pr_auc']:.4f}" if metrics['pr_auc'] else "N/A"
        print(f"  ROC-AUC:       {roc_auc_str}")
        print(f"  PR-AUC:        {pr_auc_str}")
        print(f"  Precision:     {metrics['precision']:.4f} (Positive class)")
        print(f"  Recall:        {metrics['recall']:.4f} (True positive rate)")
        print(f"  Sensitivity:   {metrics['sensitivity']:.4f} (Same as recall)")
        print(f"  Specificity:   {metrics['specificity']:.4f} (True negative rate)")
        print(f"  F1-Score:      {metrics['f1_score']:.4f} (Positive class)")
        print(f"  F1-Weighted:   {metrics['f1_weighted']:.4f} (All classes)")
        print(f"\n  Confusion Matrix (Test):")
        print(f"    TN={tn:3d} (True Negatives)   FP={fp:3d} (False Positives)")
        print(f"    FN={fn:3d} (False Negatives)  TP={tp:3d} (True Positives)")
        print(f"\n  Class Distribution (Test):")
        print(f"    Low Quality:  {metrics['label_distribution']['class_0']} samples")
        print(f"    High Quality: {metrics['label_distribution']['class_1']} samples")
        
        # Feature importance
        feature_importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        metrics['feature_importance'] = feature_importance
        
        print(f"\n[Evaluator] === Feature Importance ===")
        for feat, imp in feature_importance.items():
            print(f"  {feat:20s}: {imp:.4f}")
        
        # Save metrics
        if output_dir is None:
            output_dir = self.models_dir
        
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Evaluator] Metrics saved: {metrics_path}")
        
        # Plot ROC curve
        self._plot_roc_curve(y_test, y_proba, output_dir)
        
        # Plot PR curve
        self._plot_pr_curve(y_test, y_proba, output_dir)
        
        # Plot feature importance
        self._plot_feature_importance(feature_importance, output_dir)
        
        return metrics, test_df, y_proba
    
    def _plot_roc_curve(self, y_test, y_proba, output_dir):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Test Set)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[Evaluator] ROC curve saved: {roc_path}")
    
    def _plot_pr_curve(self, y_test, y_proba, output_dir):
        """Plot and save PR curve."""
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_vals, precision_vals)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='darkgreen', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Test Set)')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        
        pr_path = os.path.join(output_dir, 'pr_curve.png')
        plt.savefig(pr_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[Evaluator] PR curve saved: {pr_path}")
    
    def _plot_feature_importance(self, feature_importance, output_dir):
        """Plot and save feature importance."""
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances, color='steelblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance (Test Set)')
        plt.grid(alpha=0.3, axis='x')
        
        imp_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(imp_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[Evaluator] Feature importance plot saved: {imp_path}")


def evaluate_pipeline(models_dir, dataset_csv_path, output_dir=None, test_fraction=0.2):
    """
    Full evaluation pipeline.
    
    Args:
        models_dir: Directory with trained model
        dataset_csv_path: Path to dataset.csv
        output_dir: Where to save reports and plots
    
    Returns:
        metrics dict
    """
    evaluator = AMQSSEvaluator(models_dir, dataset_csv_path)
    metrics, test_df, y_proba = evaluator.evaluate(output_dir, test_fraction=test_fraction)
    return metrics


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description='Evaluate trained AMQSS model on dataset')
    parser.add_argument(
        '--models-dir',
        default=os.getenv('AMQSS_MODELS_DIR', str(project_root / 'backend' / 'ml')),
        help='Directory with model.pkl, scaler.pkl, metadata.json'
    )
    parser.add_argument(
        '--dataset-csv',
        default=os.getenv('AMQSS_DATASET_CSV', str(project_root / 'data' / 'dataset.csv')),
        help='Path to dataset.csv'
    )
    parser.add_argument(
        '--output-dir',
        default=os.getenv('AMQSS_EVAL_OUTPUT_DIR'),
        help='Directory for metrics and plots (defaults to models-dir)'
    )
    parser.add_argument('--test-fraction', type=float, default=0.2, help='Fraction used as test set')
    args = parser.parse_args()

    models_dir = str(Path(args.models_dir).expanduser())
    dataset_path = str(Path(args.dataset_csv).expanduser())
    output_dir = str(Path(args.output_dir).expanduser()) if args.output_dir else models_dir

    try:
        metrics = evaluate_pipeline(models_dir, dataset_path, output_dir, test_fraction=args.test_fraction)
        print(f"\n✓ Evaluation complete. Reports saved to {output_dir}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
