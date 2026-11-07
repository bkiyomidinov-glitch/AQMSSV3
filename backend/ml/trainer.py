"""
Trainer for AMQSS AI Layer
Purpose: Strict chronological train/test split, train RandomForest, save model + metadata.
Constraints: No data leakage, strict chronological timeline (zero lookahead bias).
"""

import os
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report


def time_based_train_test_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Strict chronological train/test split without separate per-class stratification.
    Guarantees that train timestamp < test timestamp for all samples (zero lookahead bias).
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - test_fraction))

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print("Strict chronological split (Walk-forward, zero lookahead leakage):")
    print(f"  Train: {len(train_df)} samples ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"    Class 0: {(train_df['label'] == 0).sum()} | Class 1: {(train_df['label'] == 1).sum()}")
    print(f"  Test:  {len(test_df)} samples ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    print(f"    Class 0: {(test_df['label'] == 0).sum()} | Class 1: {(test_df['label'] == 1).sum()}")

    return train_df, test_df


def extract_features_and_labels(df: pd.DataFrame, feature_cols=None):
    """Return selected feature matrix and labels."""
    if feature_cols is None:
        feature_cols = [
            'volatility_regime', 'true_liquidity', 'structure',
            'momentum', 'volatility_level', 'current_atr'
        ]

    X = df[feature_cols].values.astype(float)
    y = df['label'].values.astype(int)

    return X, y, feature_cols


def train_model(X_train, y_train, feature_cols, n_estimators=200, random_state=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    classes = np.unique(y_train)
    if len(classes) > 1:
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {c: float(w) for c, w in zip(classes, class_weights)}
    else:
        class_weight_dict = {classes[0]: 1.0}

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight_dict,
        bootstrap=True,
        oob_score=True
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_train_scaled)

    print("Model trained:")
    print(f"  Trees: {n_estimators}, Depth: 12, Min samples split: 8")
    print(f"  Train accuracy: {model.score(X_train_scaled, y_train):.4f}")
    print(f"  OOB score: {model.oob_score_:.4f}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    print(f"  Class weights: {class_weight_dict}")
    print("\n  Classification Report (Train):")
    print(classification_report(y_train, y_pred, target_names=['Low Quality', 'High Quality'] if len(classes) > 1 else None))

    return model, scaler


def save_model(model, scaler, feature_cols, output_dir, metadata=None):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[Trainer] Model saved: {model_path}")

    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[Trainer] Scaler saved: {scaler_path}")

    meta = {
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'trained_at': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        **(metadata or {})
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[Trainer] Metadata saved: {meta_path}")


def train_pipeline(
    dataset_csv_path: str,
    models_dir: str,
    test_fraction: float = 0.2,
    n_estimators: int = 100
):
    print(f"[Trainer] Loading dataset from {dataset_csv_path}...")
    df = pd.read_csv(dataset_csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Loaded {len(df)} samples")

    print("[Trainer] Splitting data (strict chronological, no shuffle, no lookahead)...")
    train_df, test_df = time_based_train_test_split(df, test_fraction=test_fraction)

    print("[Trainer] Extracting features...")
    X_train, y_train, feature_cols = extract_features_and_labels(train_df)
    print(f"  Features: {feature_cols}")
    print(f"  Shape: {X_train.shape}")

    print("[Trainer] Training RandomForest...")
    model, scaler = train_model(X_train, y_train, feature_cols, n_estimators=n_estimators)

    print("[Trainer] Saving artifacts...")
    metadata = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'test_fraction': test_fraction,
        'train_end_time': train_df['timestamp'].max().isoformat(),
        'test_start_time': test_df['timestamp'].min().isoformat(),
    }
    save_model(model, scaler, feature_cols, models_dir, metadata=metadata)

    return model, scaler, train_df, test_df


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description='Train AMQSS model from dataset CSV')
    parser.add_argument(
        '--dataset-csv',
        default=os.getenv('AMQSS_DATASET_CSV', str(project_root / 'data' / 'dataset.csv')),
        help='Path to dataset.csv'
    )
    parser.add_argument(
        '--models-dir',
        default=os.getenv('AMQSS_MODELS_DIR', str(project_root / 'backend' / 'ml')),
        help='Directory to save model artifacts'
    )
    parser.add_argument('--test-fraction', type=float, default=0.2, help='Fraction reserved for test')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees for RandomForest')
    args = parser.parse_args()

    dataset_path = str(Path(args.dataset_csv).expanduser())
    models_path = str(Path(args.models_dir).expanduser())

    try:
        model, scaler, train_df, test_df = train_pipeline(
            dataset_path,
            models_path,
            test_fraction=args.test_fraction,
            n_estimators=args.n_estimators
        )
        print(f"\n✓ Training complete. Model saved to {models_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
