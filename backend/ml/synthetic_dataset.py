"""
Synthetic dataset generator for testing AI layer when real data is limited.
Creates realistic AMQSS features + forward performance labels.
IMPROVED: Balanced classes, larger dataset, better feature engineering.
"""

import pandas as pd
import numpy as np
import os


def create_synthetic_dataset(output_path, n_samples=1000):
    """
    Generate synthetic AMQSS data with balanced classes and engineered features.
    
    Args:
        output_path: Where to save the synthetic dataset
        n_samples: Number of samples (will be split 50/50 balanced)
    """
    np.random.seed(42)
    
    # Generate timestamps (hourly, spanning n_samples hours)
    start_time = pd.Timestamp('2025-01-01')
    timestamps = [start_time + pd.Timedelta(hours=i) for i in range(n_samples)]
    
    # Create TWO classes explicitly for balance
    n_per_class = n_samples // 2
    
    # CLASS 0: Low quality markets (190 samples)
    # Features: low values, high instability, low structure
    vol_reg_0 = np.clip(np.random.normal(3.5, 1.5, n_per_class), 0, 10)
    true_liq_0 = np.clip(np.random.normal(3, 1.2, n_per_class), 0, 10)
    struct_0 = np.clip(np.random.normal(2.5, 1, n_per_class), 0, 10)
    momentum_0 = np.clip(np.random.normal(3, 1.5, n_per_class), 0, 10)
    vol_level_0 = np.clip(np.random.normal(7, 1.5, n_per_class), 0, 10)  # High instability
    atr_0 = np.abs(np.random.normal(0.0005, 0.0002, n_per_class))
    label_0 = np.zeros(n_per_class, dtype=int)
    
    # CLASS 1: High quality markets (190 samples)
    # Features: high values, stable, good structure, high momentum
    vol_reg_1 = np.clip(np.random.normal(7.5, 1.5, n_per_class), 0, 10)
    true_liq_1 = np.clip(np.random.normal(7.5, 1.2, n_per_class), 0, 10)
    struct_1 = np.clip(np.random.normal(7.5, 1, n_per_class), 0, 10)
    momentum_1 = np.clip(np.random.normal(7, 1.5, n_per_class), 0, 10)
    vol_level_1 = np.clip(np.random.normal(3, 1.5, n_per_class), 0, 10)  # Low instability (stable)
    atr_1 = np.abs(np.random.normal(0.0015, 0.0003, n_per_class))
    label_1 = np.ones(n_per_class, dtype=int)
    
    # Combine both classes
    data = {
        'volatility_regime': np.concatenate([vol_reg_0, vol_reg_1]),
        'true_liquidity': np.concatenate([true_liq_0, true_liq_1]),
        'structure': np.concatenate([struct_0, struct_1]),
        'momentum': np.concatenate([momentum_0, momentum_1]),
        'volatility_level': np.concatenate([vol_level_0, vol_level_1]),
        'current_atr': np.concatenate([atr_0, atr_1]),
        'label': np.concatenate([label_0, label_1]),
        'timestamp': timestamps
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle for randomness (keep time order in indices, but mix classes)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[SyntheticDataset] Generated {len(df)} balanced synthetic samples")
    print(f"  Label distribution:")
    print(f"    Class 0 (Low quality):  {(df['label'] == 0).sum()} samples")
    print(f"    Class 1 (High quality): {(df['label'] == 1).sum()} samples")
    print(f"    Balance: {(df['label'].value_counts() / len(df) * 100).round(1).to_dict()}%")
    print(f"  Feature stats (Class 0):")
    print(f"    volatility_regime: {df[df['label']==0]['volatility_regime'].mean():.2f} ± {df[df['label']==0]['volatility_regime'].std():.2f}")
    print(f"    momentum: {df[df['label']==0]['momentum'].mean():.2f} ± {df[df['label']==0]['momentum'].std():.2f}")
    print(f"  Feature stats (Class 1):")
    print(f"    volatility_regime: {df[df['label']==1]['volatility_regime'].mean():.2f} ± {df[df['label']==1]['volatility_regime'].std():.2f}")
    print(f"    momentum: {df[df['label']==1]['momentum'].mean():.2f} ± {df[df['label']==1]['momentum'].std():.2f}")
    print(f"  Saved to: {output_path}")
    
    return df


if __name__ == '__main__':
    output_path = r'C:\Users\bkiyo\Desktop\Project.X\data\dataset.csv'
    create_synthetic_dataset(output_path, n_samples=1000)
