"""
Trainer for AMQSS AI Layer
Purpose: Time-based train/test split, train RandomForest, save model + metadata.
Constraints: No shuffling, strict chronological split.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os
from datetime import datetime
import argparse
from pathlib import Path


def time_based_train_test_split(df, test_fraction=0.2):
    """Split each class chronologically so both sets contain each label."""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    train_dfs = []
    test_dfs = []
    
    for label in df['label'].unique():
        class_df = df[df['label'] == label].reset_index(drop=True)
        split_idx = int(len(class_df) * (1 - test_fraction))
        
        train_dfs.append(class_df.iloc[:split_idx])
        test_dfs.append(class_df.iloc[split_idx:])
    
    train_df = pd.concat(train_dfs).sort_values('timestamp').reset_index(drop=True)
    test_df = pd.concat(test_dfs).sort_values('timestamp').reset_index(drop=True)
    
    print("Time-based stratified split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"    Class 0: {(train_df['label'] == 0).sum()}")
    print(f"    Class 1: {(train_df['label'] == 1).sum()}")
    print(f"  Test:  {len(test_df)} samples")
    print(f"    Class 0: {(test_df['label'] == 0).sum()}")
    print(f"    Class 1: {(test_df['label'] == 1).sum()}")
    
    return train_df, test_df


def extract_features_and_labels(df, feature_cols=None):
    """Return the selected feature matrix and labels."""
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
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
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
        oob_score=True  # Out-of-bag scoring for better validation
    )
    model.fit(X_train_scaled, y_train)
    
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_train_scaled)
    
    print("Model trained:")
    print(f"  Trees: {n_estimators}, Depth: 12, Min samples split: 8")
    print(f"  Train accuracy: {model.score(X_train_scaled, y_train):.4f}")
    print(f"  OOB score: {model.oob_score_:.4f}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    print(f"  Class weights: {class_weight_dict}")
    print(f"\n  Classification Report (Train):")
    print(classification_report(y_train, y_pred, target_names=['Low Quality', 'High Quality']))
    
    return model, scaler


def save_model(model, scaler, feature_cols, output_dir, metadata=None):
    """
    Save model, scaler, and metadata to disk.
    
    Args:
        model: Trained RandomForest
        scaler: StandardScaler
        feature_cols: Feature names
        output_dir: Directory to save
        metadata: Dict with additional info
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[Trainer] Model saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[Trainer] Scaler saved: {scaler_path}")
    
    # Save metadata
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
    dataset_csv_path,
    models_dir,
    test_fraction=0.2,
    n_estimators=100
):
    """
    Full training pipeline: load dataset, split, train, save.
    
    Args:
        dataset_csv_path: Path to dataset.csv (from dataset_builder)
        models_dir: Directory to save model artifacts
        test_fraction: Fraction for test set
        n_estimators: Number of trees
    
    Returns:
        model, scaler, train_df, test_df
    """
    print("[Trainer] Loading dataset...")
    df = pd.read_csv(dataset_csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Loaded {len(df)} samples")
    
    print("[Trainer] Splitting data (time-based, no shuffle)...")
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
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Train AMQSS model from dataset CSV')
    parser.add_argument(
        '--dataset-csv',
        default=os.getenv('AMQSS_DATASET_CSV', str(project_root / 'data' / 'dataset.csv')),
        help='Path to dataset.csv'
    )
    parser.add_argument(
        '--models-dir',
        default=os.getenv('AMQSS_MODELS_DIR', str(project_root / 'models')),
        help='Directory to save model artifacts'
    )
    parser.add_argument('--test-fraction', type=float, default=0.2, help='Fraction of dataset reserved for test')
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
