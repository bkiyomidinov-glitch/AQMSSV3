"""
Core AMQSS (Adaptive Market Quality Scoring System) Engine.
Pure functional logic independent of CLI and transport layers.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


DEFAULT_BASE_WEIGHTS = {
    "volatility_regime": 0.25,
    "true_liquidity": 0.15,
    "structure": 0.30,
    "momentum": 0.20,
    "volatility_level": 0.10
}

DEFAULT_THRESHOLDS = {
    "NO_TRADE": 4.0,
    "WEAK": 6.0,
    "NORMAL": 7.5,
    "HIGH": 10.0
}


def normalize_weights(raw_weights: Dict[str, float], min_weight: float = 0.03) -> Dict[str, float]:
    """Normalize factor weights so they are positive and sum to 1, guaranteeing each >= min_weight."""
    n = len(raw_weights)
    if n == 0:
        return {}
    if min_weight * n >= 1.0:
        equal = 1.0 / n
        return {k: equal for k in raw_weights}

    non_negative = {k: max(float(v), 0.0) for k, v in raw_weights.items()}
    total = sum(non_negative.values())
    if total <= 0:
        equal = 1.0 / n
        return {k: equal for k in raw_weights}

    remaining = 1.0 - (n * min_weight)
    return {k: min_weight + (v / total) * remaining for k, v in non_negative.items()}


def build_adaptive_weights(
    static_weights: Dict[str, float],
    scores: Dict[str, float],
    trend_strength: float,
    direction_bias: float,
    ai_feature_weights: Optional[Dict[str, float]] = None,
    ai_high_quality_prob: Optional[float] = None
) -> Dict[str, float]:
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


def clean_price_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean OHLCV price dataframe."""
    df = df_raw.copy()
    if df.shape[1] >= 8 and ('open' not in df.columns or df.columns[0] != 'time'):
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'volume2', 'other'][:df.shape[1]]
        if df.iloc[0]['open'] == 'open' or str(df.iloc[0]['open']).isalpha():
            df = df.iloc[1:].reset_index(drop=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    if 'volume2' in df.columns:
        numeric_cols.append('volume2')

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['high', 'low', 'close'])

    # Sanity checks
    if (df['high'] < df['low']).any():
        df = df[df['high'] >= df['low']]

    return df.reset_index(drop=True)


def calculate_amqss_scores(
    df_raw: pd.DataFrame,
    predictor: Any = None,
    structure_window: int = 20,
    momentum_window: int = 5,
    atr_window: int = 50,
    liquidity_window: int = 10,
    time_weighted: bool = True,
    base_weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compute AMQSS quality scores and regime metrics on cleaned price data.
    """
    weights_baseline = base_weights or DEFAULT_BASE_WEIGHTS
    thresh = thresholds or DEFAULT_THRESHOLDS

    df = clean_price_dataframe(df_raw)
    required_rows = max(structure_window, momentum_window, atr_window)
    if len(df) < required_rows:
        raise ValueError(f"Not enough data after cleaning. Have {len(df)}, require at least {required_rows} rows.")

    # True Range and ATR
    df['HL'] = df['high'] - df['low']
    df['HC'] = (df['high'] - df['close'].shift(1)).abs()
    df['LC'] = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(atr_window).mean()

    df = df.dropna().reset_index(drop=True)
    if len(df) < max(structure_window, momentum_window):
        raise ValueError("Not enough valid data after ATR computation.")

    mean_atr = float(df['ATR'].mean())
    atr_std = float(df['ATR'].std())
    current_atr = float(df['ATR'].iloc[-1])

    # 1. Volatility Regime
    atr_ratio = current_atr / mean_atr if mean_atr > 0 else 0.0
    volatility_regime_score = float(np.clip(atr_ratio * 10, 0, 10))

    # 2. True Liquidity
    recent_volume = df['volume'].iloc[-liquidity_window:].values if 'volume' in df.columns else np.array([])
    mean_volume = float(df['volume'].mean()) if 'volume' in df.columns else 0.0
    std_volume = float(df['volume'].std()) if 'volume' in df.columns else 0.0

    if mean_volume > 0 and len(recent_volume) > 0 and recent_volume.mean() > 0:
        volume_ratio = recent_volume.mean() / mean_volume
        volume_magnitude = np.clip(volume_ratio * 10, 0, 10)
        volume_cv = std_volume / recent_volume.mean()
        consistency_score = 10 - np.clip(volume_cv * 10, 0, 10)
        true_liquidity_raw = (volume_magnitude * 0.6) + (consistency_score * 0.4)
        true_liquidity_score = float(np.clip(true_liquidity_raw, 0, 10))
    else:
        true_liquidity_score = 5.0

    # 3. Volatility Level (Stability)
    if mean_atr > 0:
        volatility_stability_ratio = atr_std / mean_atr
        volatility_level_score = float(np.clip(10 - (volatility_stability_ratio * 10), 0, 10))
    else:
        volatility_level_score = 0.0

    # 4. Structure (Trend strength via linear regression)
    highs = df['high'].iloc[-structure_window:].values
    lows = df['low'].iloc[-structure_window:].values
    x = np.arange(structure_window)

    coeffs_high = np.polyfit(x, highs, 1)
    coeffs_low = np.polyfit(x, lows, 1)
    fit_high = np.polyval(coeffs_high, x)
    fit_low = np.polyval(coeffs_low, x)

    ss_res_high = np.sum((highs - fit_high) ** 2)
    ss_tot_high = np.sum((highs - np.mean(highs)) ** 2)
    r2_high = 1 - (ss_res_high / ss_tot_high) if ss_tot_high > 0 else 0.0

    ss_res_low = np.sum((lows - fit_low) ** 2)
    ss_tot_low = np.sum((lows - np.mean(lows)) ** 2)
    r2_low = 1 - (ss_res_low / ss_tot_low) if ss_tot_low > 0 else 0.0

    r2_avg = max(0.0, float((r2_high + r2_low) / 2))
    structure_score = float(np.clip(r2_avg * 10, 0, 10))

    # 5. Momentum
    momentum_range = (df['high'].iloc[-momentum_window:] - df['low'].iloc[-momentum_window:]).values
    momentum_atr = df['ATR'].iloc[-momentum_window:].values
    momentum_closes = df['close'].iloc[-momentum_window:].values
    momentum_opens = df['open'].iloc[-momentum_window:].values

    if np.any(momentum_atr == 0) or np.any(np.isnan(momentum_atr)):
        momentum_score = 5.0
    else:
        if time_weighted:
            weights_array = np.linspace(0.5, 1.0, momentum_window)
            ratio_weighted = (momentum_range / momentum_atr) * weights_array
            momentum_magnitude = (ratio_weighted.sum() / weights_array.sum()) * 10

            direction_raw = (momentum_closes - momentum_opens) * weights_array
            direction_normalized = (direction_raw.sum() / momentum_range.sum()) if momentum_range.sum() > 0 else 0.0
            direction_component = np.clip(direction_normalized * 3, -2, 2)
            momentum_raw = np.clip(momentum_magnitude + direction_component, 0, 10)
        else:
            momentum_magnitude = (momentum_range / momentum_atr).mean() * 10
            direction_raw = (momentum_closes - momentum_opens).mean()
            direction_normalized = direction_raw / momentum_range.mean() if momentum_range.mean() > 0 else 0.0
            direction_component = np.clip(direction_normalized * 3, -2, 2)
            momentum_raw = np.clip(momentum_magnitude + direction_component, 0, 10)

        momentum_score = float(np.clip(momentum_raw, 0, 10))

    scores = {
        "volatility_regime": volatility_regime_score,
        "true_liquidity": true_liquidity_score,
        "structure": structure_score,
        "momentum": momentum_score,
        "volatility_level": volatility_level_score
    }

    # Predictive AI enhancement
    ai_high_quality_prob = None
    ai_feature_weights = None

    current_features = {
        'volatility_regime': volatility_regime_score,
        'true_liquidity': true_liquidity_score,
        'structure': structure_score,
        'momentum': momentum_score,
        'volatility_level': volatility_level_score,
        'current_atr': current_atr,
        'recent_volume': float(recent_volume.mean()) if len(recent_volume) > 0 else 0.0
    }

    if predictor is not None:
        try:
            ai_high_quality_prob = float(predictor.predict(current_features))
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
        except Exception:
            ai_high_quality_prob = None

    trend_strength = float(np.clip(r2_avg, 0, 1))
    direction_bias = float(np.clip((momentum_score - 5.0) / 5.0, -1, 1))

    adaptive_weights = build_adaptive_weights(
        static_weights=weights_baseline,
        scores=scores,
        trend_strength=trend_strength,
        direction_bias=direction_bias,
        ai_feature_weights=ai_feature_weights,
        ai_high_quality_prob=ai_high_quality_prob
    )

    total_score = float(sum(scores[k] * adaptive_weights[k] for k in scores))
    if np.isnan(total_score) or np.isinf(total_score):
        total_score = 0.0

    if total_score < thresh['NO_TRADE']:
        market_condition = "NO TRADE"
    elif total_score < thresh['WEAK']:
        market_condition = "WEAK CONDITIONS"
    elif total_score < thresh['NORMAL']:
        market_condition = "NORMAL MARKET"
    else:
        market_condition = "HIGH QUALITY MARKET"

    safe_idx = max(0, len(df) - 1)
    timestamp = str(df['time'].iloc[safe_idx]) if 'time' in df.columns else ""

    return {
        'timestamp': timestamp,
        'total_score': round(total_score, 4),
        'market_condition': market_condition,
        'scores': {k: round(v, 4) for k, v in scores.items()},
        'adaptive_weights': {k: round(v, 4) for k, v in adaptive_weights.items()},
        'metrics': {
            'current_atr': current_atr,
            'mean_atr': mean_atr,
            'atr_std': atr_std,
            'recent_volume': current_features['recent_volume'],
            'trend_strength': trend_strength,
            'direction_bias': direction_bias,
        },
        'ai_high_quality_prob': ai_high_quality_prob,
        'df_processed': df
    }
