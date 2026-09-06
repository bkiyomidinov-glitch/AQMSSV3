import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import argparse
from pathlib import Path

try:
    from predictor import load_predictor
except Exception:
    load_predictor = None


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_runtime_args():
    """Parse runtime paths so main.py can run as portable project core."""
    parser = argparse.ArgumentParser(description='AMQSS main scoring core with adaptive AI weights')
    parser.add_argument(
        '--price-csv',
        default=os.getenv('AMQSS_PRICE_CSV'),
        help='Path to raw price CSV (OHLCV, no header expected)'
    )
    parser.add_argument(
        '--models-dir',
        default=os.getenv('AMQSS_MODELS_DIR', str(PROJECT_ROOT / 'models')),
        help='Directory with AI model artifacts (model.pkl, scaler.pkl, metadata.json)'
    )
    parser.add_argument(
        '--results-dir',
        default=os.getenv('AMQSS_RESULTS_DIR', str(PROJECT_ROOT / 'results')),
        help='Directory where market_scores.csv will be saved'
    )
    return parser.parse_args()


args = parse_runtime_args()
price_csv_path = Path(args.price_csv).expanduser() if args.price_csv else None
models_dir = str(Path(args.models_dir).expanduser())
results_dir = str(Path(args.results_dir).expanduser())

legacy_default_csv = Path.home() / 'Downloads' / 'EURUSD_60_2025-01-20_2026-01-19.csv'
if price_csv_path is None and legacy_default_csv.exists():
    price_csv_path = legacy_default_csv

if price_csv_path is None:
    print("Error: price CSV path is required.")
    print("Use --price-csv <path> or set AMQSS_PRICE_CSV environment variable.")
    exit()

if not price_csv_path.exists():
    print(f"Error: price CSV not found: {price_csv_path}")
    exit()

base_weights = {
    "volatility_regime": 0.25,
    "true_liquidity": 0.15,
    "structure": 0.30,
    "momentum": 0.20,
    "volatility_level": 0.10
}


def normalize_weights(raw_weights, min_weight=0.03):
    """Normalize factor weights so they are positive and sum to 1."""
    cleaned = {k: max(float(v), min_weight) for k, v in raw_weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        equal = 1.0 / len(cleaned)
        return {k: equal for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


def build_adaptive_weights(
    static_weights,
    scores,
    trend_strength,
    direction_bias,
    ai_feature_weights=None,
    ai_high_quality_prob=None
):
    """
    Build adaptive AMQSS weights from:
    1) static baseline,
    2) current trend regime,
    3) AI feature importances (if trained model exists).
    """
    adaptive = dict(static_weights)

    if trend_strength >= 0.55:
        adaptive['structure'] += 0.07
        adaptive['momentum'] += 0.05 + (0.02 * abs(direction_bias))
        adaptive['volatility_level'] += 0.01
        adaptive['true_liquidity'] -= 0.06
        adaptive['volatility_regime'] -= 0.07
    else:
        adaptive['true_liquidity'] += 0.07
        adaptive['volatility_level'] += 0.06
        adaptive['structure'] += 0.02
        adaptive['momentum'] -= 0.07
        adaptive['volatility_regime'] -= 0.08

    if scores.get('volatility_regime', 0) > 8.0:
        adaptive['volatility_level'] += 0.04
        adaptive['momentum'] -= 0.02
        adaptive['structure'] -= 0.02

    if ai_feature_weights is not None:
        confidence = 0.2
        if ai_high_quality_prob is not None:
            confidence = float(np.clip(abs(ai_high_quality_prob - 0.5) * 2.0, 0, 1))

        ai_blend = 0.15 + (0.35 * confidence)
        for factor in adaptive:
            adaptive[factor] = ((1 - ai_blend) * adaptive[factor]) + (ai_blend * ai_feature_weights.get(factor, adaptive[factor]))

    return normalize_weights(adaptive)

thresholds = {
    "NO_TRADE": 4,      # Bottom 25%
    "WEAK": 6,          # 25-50%
    "NORMAL": 7.5,      # 50-75%
    "HIGH": 10          # Top 25%
}
structure_window = 20
momentum_window = 5
atr_window = 50
liquidity_window = 10
time_weighted = True

try:
    df = pd.read_csv(price_csv_path, header=None)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'volume2', 'other']
    df = df.iloc[1:].reset_index(drop=True)
except Exception as e:
    print("Error reading CSV:", e)
    exit()

if len(df) < 50:
    print("Error: Not enough data. Require at least 50 candles.")
    exit()

df = df.dropna()

df['open'] = pd.to_numeric(df['open'], errors='coerce')
df['high'] = pd.to_numeric(df['high'], errors='coerce')
df['low'] = pd.to_numeric(df['low'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
df['volume2'] = pd.to_numeric(df['volume2'], errors='coerce')
df = df.dropna()

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if (df['high'] < df['low']).any():
    print("Warning: Found candles where high < low. Removing corrupted data.")
    df = df[df['high'] >= df['low']]

if len(df) < 50:
    print("Error: Not enough valid data after integrity check.")
    exit()

df['HL'] = df['high'] - df['low']
df['HC'] = abs(df['high'] - df['close'].shift(1))
df['LC'] = abs(df['low'] - df['close'].shift(1))
df['TR'] = df[['HL','HC','LC']].max(axis=1)
df['ATR'] = df['TR'].rolling(atr_window).mean()

mean_atr = df['ATR'].mean()
atr_std = df['ATR'].std()

df = df.dropna()

required_rows = max(structure_window, momentum_window)
if len(df) < required_rows:
    print(f"Error: Not enough data after processing. Have {len(df)}, need {required_rows}")
    exit()

current_atr = df['ATR'].iloc[-1]

atr_ratio = current_atr / mean_atr if mean_atr > 0 else 0
volatility_regime_raw = atr_ratio * 10
volatility_regime_score = np.clip(volatility_regime_raw, 0, 10)

recent_volume = df['volume'].iloc[-liquidity_window:].values
mean_volume = df['volume'].mean()
std_volume = df['volume'].std()

if mean_volume > 0 and len(recent_volume) > 0:
    volume_ratio = recent_volume.mean() / mean_volume
    volume_magnitude = np.clip(volume_ratio * 10, 0, 10)
    
    if recent_volume.mean() > 0:
        volume_cv = std_volume / recent_volume.mean()
        consistency_score = 10 - np.clip(volume_cv * 10, 0, 10)  # Invert: low CV = high score
    else:
        consistency_score = 5
    
    true_liquidity_raw = (volume_magnitude * 0.6) + (consistency_score * 0.4)
    true_liquidity_score = np.clip(true_liquidity_raw, 0, 10)
else:
    true_liquidity_score = 5  # Neutral if no volume data

if mean_atr > 0:
    volatility_stability_ratio = atr_std / mean_atr
    volatility_level_raw = 10 - np.clip(volatility_stability_ratio * 10, 0, 10)
else:
    volatility_level_raw = 0
volatility_level_score = np.clip(volatility_level_raw, 0, 10)

highs = df['high'].iloc[-structure_window:].values
lows = df['low'].iloc[-structure_window:].values
x = np.arange(structure_window)

coeffs_high = np.polyfit(x, highs, 1)
coeffs_low = np.polyfit(x, lows, 1)
fit_high = np.polyval(coeffs_high, x)
fit_low = np.polyval(coeffs_low, x)

ss_res_high = np.sum((highs - fit_high) ** 2)
ss_tot_high = np.sum((highs - np.mean(highs)) ** 2)
r2_high = 1 - (ss_res_high / ss_tot_high) if ss_tot_high > 0 else 0

ss_res_low = np.sum((lows - fit_low) ** 2)
ss_tot_low = np.sum((lows - np.mean(lows)) ** 2)
r2_low = 1 - (ss_res_low / ss_tot_low) if ss_tot_low > 0 else 0

r2_avg = (r2_high + r2_low) / 2
structure_raw = np.clip(r2_avg * 10, 0, 10)
structure_score = structure_raw

momentum_range = (df['high'].iloc[-momentum_window:] - df['low'].iloc[-momentum_window:]).values
momentum_atr = df['ATR'].iloc[-momentum_window:].values
momentum_closes = df['close'].iloc[-momentum_window:].values
momentum_opens = df['open'].iloc[-momentum_window:].values

if np.any(momentum_atr == 0) or np.any(np.isnan(momentum_atr)):
    momentum_score = 5  # Neutral if data invalid
else:
    if time_weighted:
        weights_array = np.linspace(0.5, 1.0, momentum_window)
        ratio_weighted = (momentum_range / momentum_atr) * weights_array
        momentum_magnitude = (ratio_weighted.sum() / weights_array.sum()) * 10
        
        direction_raw = (momentum_closes - momentum_opens) * weights_array
        direction_normalized = (direction_raw.sum() / momentum_range.sum()) if momentum_range.sum() > 0 else 0
        direction_component = np.clip(direction_normalized * 3, -2, 2)  # ±2 adjustment
        
        momentum_raw = np.clip(momentum_magnitude + direction_component, 0, 10)
    else:
        momentum_magnitude = (momentum_range / momentum_atr).mean() * 10
        direction_raw = (momentum_closes - momentum_opens).mean()
        direction_normalized = direction_raw / momentum_range.mean() if momentum_range.mean() > 0 else 0
        direction_component = np.clip(direction_normalized * 3, -2, 2)
        momentum_raw = np.clip(momentum_magnitude + direction_component, 0, 10)
    
    momentum_score = np.clip(momentum_raw, 0, 10)

scores = {
    "volatility_regime": volatility_regime_score,
    "true_liquidity": true_liquidity_score,
    "structure": structure_score,
    "momentum": momentum_score,
    "volatility_level": volatility_level_score
}

for k, v in scores.items():
    if np.isnan(v) or np.isinf(v):
        print(f"Warning: {k} score is invalid ({v}), setting to 0")
        scores[k] = 0

ai_high_quality_prob = None
ai_feature_weights = None

current_features = {
    'volatility_regime': float(volatility_regime_score),
    'true_liquidity': float(true_liquidity_score),
    'structure': float(structure_score),
    'momentum': float(momentum_score),
    'volatility_level': float(volatility_level_score),
    'current_atr': float(current_atr),
    'recent_volume': float(df['volume'].iloc[-liquidity_window:].mean() if 'volume' in df.columns else 0)
}

if load_predictor is not None:
    try:
        predictor = load_predictor(models_dir)
        ai_high_quality_prob = predictor.predict(current_features)

        if hasattr(predictor, 'model') and hasattr(predictor.model, 'feature_importances_'):
            feature_cols = getattr(predictor, 'feature_cols', [])
            feature_importances = predictor.model.feature_importances_
            raw_ai_weights = {
                col: float(imp)
                for col, imp in zip(feature_cols, feature_importances)
                if col in scores
            }

            if raw_ai_weights and sum(raw_ai_weights.values()) > 0:
                ai_feature_weights = normalize_weights({k: raw_ai_weights.get(k, 0.0) for k in scores}, min_weight=0.001)

        print(f"\n🤖 AI Market Environment Score: {ai_high_quality_prob:.4f}")
    except Exception as e:
        print(f"\n⚠️  AI Predictor unavailable: {e}")

trend_strength = float(np.clip(r2_avg, 0, 1))
direction_bias = float(np.clip((momentum_score - 5.0) / 5.0, -1, 1))

adaptive_weights = build_adaptive_weights(
    static_weights=base_weights,
    scores=scores,
    trend_strength=trend_strength,
    direction_bias=direction_bias,
    ai_feature_weights=ai_feature_weights,
    ai_high_quality_prob=ai_high_quality_prob
)

total_score = sum(scores[k] * adaptive_weights[k] for k in scores)

# Ensure total_score is valid
if np.isnan(total_score) or np.isinf(total_score):
    print("Error: Total score is invalid. Check data quality.")
    total_score = 0

if total_score < thresholds['NO_TRADE']:
    market_condition = "NO TRADE"
elif total_score < thresholds['WEAK']:
    market_condition = "WEAK CONDITIONS"
elif total_score < thresholds['NORMAL']:
    market_condition = "NORMAL MARKET"
else:
    market_condition = "HIGH QUALITY MARKET"

print("=== Market Quality Score (AMQSS v2 - Fixed) ===")
print("\nScores per factor:")
print(f"  Volatility Regime (ATR ratio): {volatility_regime_score:.2f}/10")
print(f"  True Liquidity (Volume-based): {true_liquidity_score:.2f}/10")
print(f"  Structure (Trend strength): {structure_score:.2f}/10")
print(f"  Momentum (Candle range): {momentum_score:.2f}/10")
print(f"  Volatility Level (Stability): {volatility_level_score:.2f}/10")
print("\nAdaptive weights (AI + trend):")
for factor_name, factor_weight in adaptive_weights.items():
    print(f"  {factor_name}: {factor_weight:.3f}")
print(f"\nTotal score: {total_score:.2f}/10")
print(f"Market condition: {market_condition}")
print("\nNote: thresholds should be backtested for your strategy")
print("\nLast 10 candles:")
print(df[['time', 'open', 'high', 'low', 'close', 'volume', 'ATR']].tail(10).to_string(index=False))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

safe_idx = max(0, len(df) - 30)
amqss_timestamp = pd.to_datetime(df['time'].iloc[safe_idx])

results_df = pd.DataFrame({
    'timestamp': [amqss_timestamp],
    'total_score': [total_score],
    'volatility_regime': [volatility_regime_score],
    'true_liquidity': [true_liquidity_score],
    'structure': [structure_score],
    'momentum': [momentum_score],
    'volatility_level': [volatility_level_score],
    'market_condition': [market_condition],
    'current_atr': [current_atr],
    'mean_atr': [mean_atr],
    'atr_std': [atr_std],
    'recent_volume': [df['volume'].iloc[-liquidity_window:].mean() if 'volume' in df.columns else 0],
    'trend_strength': [trend_strength],
    'direction_bias': [direction_bias],
    'w_volatility_regime': [adaptive_weights['volatility_regime']],
    'w_true_liquidity': [adaptive_weights['true_liquidity']],
    'w_structure': [adaptive_weights['structure']],
    'w_momentum': [adaptive_weights['momentum']],
    'w_volatility_level': [adaptive_weights['volatility_level']]
})

results_df['ai_high_quality_prob'] = [ai_high_quality_prob]

# Save to CSV (append mode with proper header handling)
csv_path = os.path.join(results_dir, 'market_scores.csv')
if os.path.exists(csv_path):
    # Read existing data, append new row, write back
    try:
        existing_df = pd.read_csv(csv_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    except Exception as e:
        print(f"Warning: Could not append to existing CSV ({e}). Overwriting.")
    results_df.to_csv(csv_path, index=False)
else:
    results_df.to_csv(csv_path, index=False)

print(f"\n💾 Results saved to: {csv_path}")