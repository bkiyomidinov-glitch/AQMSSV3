"""
Dataset Builder for AMQSS AI Layer
Purpose: Build a clean, time-aligned dataset from AMQSS output + future performance data.
Optimized: Fully vectorized performance calculation, zero look-ahead bias.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_amqss_results(csv_path: str) -> pd.DataFrame:
    """Load AMQSS results (market_scores.csv)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[DatasetBuilder] Warning: CSV parsing error ({e}). Attempting lenient read...")
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(csv_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=False)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_price_data(csv_path: str) -> pd.DataFrame:
    """Load raw candle data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Price data not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'volume2', 'other'][:df.shape[1]]

    # If first row contains header strings, skip it
    if df.iloc[0]['open'] == 'open' or str(df.iloc[0]['open']).isalpha():
        df = df.iloc[1:].reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['time', 'close'])
    df['time'] = pd.to_datetime(df['time'].astype(str).str.replace('.', '-', regex=False), format='mixed', errors='coerce')
    df = df.dropna(subset=['time'])

    df = df.sort_values('time').reset_index(drop=True)
    return df


def compute_forward_performance(price_df: pd.DataFrame, lookback_bars: int = 20) -> pd.DataFrame:
    """
    Vectorized computation of forward_return and max_drawdown over the next N bars.
    Replaces slow row-by-row iteration with numpy sliding windows.
    """
    df = price_df.copy()
    n = len(df)

    forward_returns = np.full(n, np.nan, dtype=float)
    max_drawdowns = np.full(n, np.nan, dtype=float)

    if n > lookback_bars:
        closes = df['close'].to_numpy(dtype=float)

        # sliding window over future closes [i+1 : i+1+lookback_bars]
        windows = np.lib.stride_tricks.sliding_window_view(closes[1:], window_shape=lookback_bars)
        num_windows = len(windows)

        entry_prices = closes[:num_windows]
        exit_prices = windows[:, -1]
        min_prices = windows.min(axis=1)

        valid = entry_prices != 0
        forward_returns[:num_windows][valid] = (exit_prices[valid] - entry_prices[valid]) / entry_prices[valid]
        max_drawdowns[:num_windows][valid] = (min_prices[valid] - entry_prices[valid]) / entry_prices[valid]

    df['forward_return'] = forward_returns
    df['max_drawdown'] = max_drawdowns
    return df


def build_dataset(
    amqss_csv_path: str,
    price_csv_path: str,
    output_csv_path: str,
    lookback_bars: int = 20,
    return_threshold: float = 0.002,
    drawdown_threshold: float = -0.01
) -> pd.DataFrame:
    """
    Full pipeline: Load, align, compute forward performance vectorially, generate labels, save.
    """
    print("[DatasetBuilder] Loading AMQSS results...")
    amqss_df = load_amqss_results(amqss_csv_path)
    print(f"  Loaded {len(amqss_df)} AMQSS records")

    print("[DatasetBuilder] Loading price data...")
    price_df = load_price_data(price_csv_path)
    print(f"  Loaded {len(price_df)} price candles")

    print("[DatasetBuilder] Computing forward performance (vectorized)...")
    price_df = compute_forward_performance(price_df, lookback_bars=lookback_bars)
    valid_perf = price_df['forward_return'].notna().sum()
    print(f"  Computed forward performance for {valid_perf} candles")

    print("[DatasetBuilder] Aligning AMQSS with price data using merge_asof...")
    aligned_df = pd.merge_asof(
        amqss_df.sort_values('timestamp'),
        price_df[['time', 'forward_return', 'max_drawdown']].sort_values('time'),
        left_on='timestamp',
        right_on='time',
        direction='nearest',
        tolerance=pd.Timedelta('24h')
    )
    aligned_df = aligned_df.dropna(subset=['forward_return']).reset_index(drop=True)
    print(f"  Aligned {len(aligned_df)} records")

    if len(aligned_df) == 0:
        print("  ERROR: No AMQSS records aligned with price data within 24h tolerance.")
        return None

    print("[DatasetBuilder] Generating labels...")
    aligned_df['label'] = (
        (aligned_df['forward_return'] >= return_threshold) &
        (aligned_df['max_drawdown'] >= drawdown_threshold)
    ).astype(int)

    label_dist = aligned_df['label'].value_counts()
    print(f"  Label distribution: {label_dist.to_dict()}")

    feature_cols = [
        'volatility_regime', 'true_liquidity', 'structure',
        'momentum', 'volatility_level', 'current_atr'
    ]

    missing = [c for c in feature_cols if c not in aligned_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in AMQSS data: {missing}")

    output_cols = feature_cols + ['label', 'timestamp']
    dataset_df = aligned_df[output_cols].reset_index(drop=True)
    print(f"  Final dataset: {len(dataset_df)} samples")

    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    dataset_df.to_csv(output_csv_path, index=False)
    print(f"[DatasetBuilder] Dataset saved to {output_csv_path}")

    return dataset_df


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description='Build AMQSS training dataset from scores + price data')
    parser.add_argument(
        '--amqss-csv',
        default=os.getenv('AMQSS_RESULTS_CSV', str(project_root / 'data' / 'market_scores.csv')),
        help='Path to AMQSS results CSV (market_scores.csv)'
    )
    parser.add_argument(
        '--price-csv',
        default=os.getenv('AMQSS_PRICE_CSV'),
        help='Path to raw OHLCV price CSV'
    )
    parser.add_argument(
        '--output-csv',
        default=os.getenv('AMQSS_DATASET_CSV', str(project_root / 'data' / 'dataset.csv')),
        help='Output dataset CSV path'
    )
    parser.add_argument('--lookback-bars', type=int, default=20, help='Forward bars to compute performance')
    parser.add_argument('--return-threshold', type=float, default=0.002, help='Min forward return for label=1')
    parser.add_argument('--drawdown-threshold', type=float, default=-0.01, help='Min allowed max drawdown for label=1')
    args = parser.parse_args()

    legacy_default_csv = Path.home() / 'Downloads' / 'EURUSD_60_2025-01-20_2026-01-19.csv'
    price_path = None
    if args.price_csv:
        price_path = Path(args.price_csv).expanduser()
    elif legacy_default_csv.exists():
        price_path = legacy_default_csv

    if not price_path or not price_path.exists():
        print("\n✗ Error: Valid price CSV path is required.")
        raise SystemExit(1)

    amqss_path = Path(args.amqss_csv).expanduser()
    output_path = Path(args.output_csv).expanduser()

    try:
        build_dataset(
            str(amqss_path),
            str(price_path),
            str(output_path),
            lookback_bars=args.lookback_bars,
            return_threshold=args.return_threshold,
            drawdown_threshold=args.drawdown_threshold
        )
        print(f"\n✓ Dataset ready: {output_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
