"""
AMQSS CLI Runner
Executes core market scoring algorithm and records results to CSV.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.scoring import calculate_amqss_scores

try:
    from backend.ml.predictor import load_predictor
except Exception:
    load_predictor = None


def parse_runtime_args():
    """Parse runtime CLI arguments."""
    parser = argparse.ArgumentParser(description='AMQSS main scoring core with adaptive AI weights')
    parser.add_argument(
        '--price-csv',
        default=os.getenv('AMQSS_PRICE_CSV'),
        help='Path to raw price CSV (OHLCV, no header expected)'
    )
    parser.add_argument(
        '--models-dir',
        default=os.getenv('AMQSS_MODELS_DIR', str(PROJECT_ROOT / 'backend' / 'ml')),
        help='Directory with AI model artifacts (model.pkl, scaler.pkl, metadata.json)'
    )
    parser.add_argument(
        '--results-dir',
        default=os.getenv('AMQSS_RESULTS_DIR', str(PROJECT_ROOT / 'data')),
        help='Directory where market_scores.csv will be saved'
    )
    return parser.parse_args()


def main():
    args = parse_runtime_args()
    price_csv_path = Path(args.price_csv).expanduser() if args.price_csv else None
    models_dir = str(Path(args.models_dir).expanduser())
    results_dir = str(Path(args.results_dir).expanduser())

    legacy_default_csv = Path.home() / 'Downloads' / 'EURUSD_60_2025-01-20_2026-01-19.csv'
    if price_csv_path is None and legacy_default_csv.exists():
        price_csv_path = legacy_default_csv

    # Also check local data directory if available
    fallback_data_csv = PROJECT_ROOT / 'data' / 'EURUSD_60_2025-01-20_2026-01-19.csv'
    if price_csv_path is None and fallback_data_csv.exists():
        price_csv_path = fallback_data_csv

    if price_csv_path is None:
        print("Error: price CSV path is required.")
        print("Use --price-csv <path> or set AMQSS_PRICE_CSV environment variable.")
        sys.exit(1)

    if not price_csv_path.exists():
        print(f"Error: price CSV not found: {price_csv_path}")
        sys.exit(1)

    try:
        raw_df = pd.read_csv(price_csv_path, header=None)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    predictor = None
    if load_predictor is not None and os.path.exists(models_dir):
        try:
            predictor = load_predictor(models_dir)
        except Exception as e:
            print(f"Note: Predictor not initialized from {models_dir}: {e}")

    try:
        res = calculate_amqss_scores(raw_df, predictor=predictor)
    except Exception as e:
        print(f"Error calculating AMQSS scores: {e}")
        sys.exit(1)

    scores = res['scores']
    adaptive_weights = res['adaptive_weights']
    total_score = res['total_score']
    market_condition = res['market_condition']
    metrics = res['metrics']

    print("=== Market Quality Score (AMQSS v2 - Architecture Clean) ===")
    print("\nScores per factor:")
    for k, v in scores.items():
        print(f"  {k}: {v:.2f}/10")

    print("\nAdaptive weights (AI + trend):")
    for k, v in adaptive_weights.items():
        print(f"  {k}: {v:.3f}")

    print(f"\nTotal score: {total_score:.2f}/10")
    print(f"Market condition: {market_condition}")

    os.makedirs(results_dir, exist_ok=True)
    results_df = pd.DataFrame([{
        'timestamp': res['timestamp'],
        'total_score': total_score,
        'volatility_regime': scores['volatility_regime'],
        'true_liquidity': scores['true_liquidity'],
        'structure': scores['structure'],
        'momentum': scores['momentum'],
        'volatility_level': scores['volatility_level'],
        'market_condition': market_condition,
        'current_atr': metrics['current_atr'],
        'mean_atr': metrics['mean_atr'],
        'atr_std': metrics['atr_std'],
        'recent_volume': metrics['recent_volume'],
        'trend_strength': metrics['trend_strength'],
        'direction_bias': metrics['direction_bias'],
        'w_volatility_regime': adaptive_weights['volatility_regime'],
        'w_true_liquidity': adaptive_weights['true_liquidity'],
        'w_structure': adaptive_weights['structure'],
        'w_momentum': adaptive_weights['momentum'],
        'w_volatility_level': adaptive_weights['volatility_level'],
        'ai_high_quality_prob': res['ai_high_quality_prob']
    }])

    csv_path = os.path.join(results_dir, 'market_scores.csv')
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            results_df = pd.concat([existing, results_df], ignore_index=True)
        except Exception:
            pass

    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")


if __name__ == '__main__':
    main()
