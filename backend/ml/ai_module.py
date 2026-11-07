import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    RandomForestClassifier = None


def train_and_predict(history_csv, current_features):
    """Train a simple model on historical `market_scores.csv` and predict
    the probability that the current features correspond to a high-quality market.

    Returns a float probability in [0,1], or None if training is not possible.
    """
    if RandomForestClassifier is None:
        return None

    try:
        hist = pd.read_csv(history_csv)
    except Exception:
        return None

    # Required feature columns (fallback-safe)
    feature_cols = ['volatility_regime', 'true_liquidity', 'structure', 'momentum', 'volatility_level', 'current_atr', 'recent_volume']
    missing = [c for c in feature_cols if c not in hist.columns]
    if missing:
        return None

    # Create binary label: 1 if historical market_condition was HIGH QUALITY MARKET
    if 'market_condition' not in hist.columns:
        return None
    hist['label'] = hist['market_condition'].apply(lambda x: 1 if str(x).upper().strip().startswith('HIGH') else 0)

    # Need enough data to train
    if len(hist) < 50 or hist['label'].sum() < 10:
        return None

    X = hist[feature_cols].fillna(0).astype(float)
    y = hist['label'].astype(int)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    cur = pd.DataFrame([current_features])
    X_new = cur[feature_cols].fillna(0).astype(float)
    try:
        prob = clf.predict_proba(X_new)[0][1]
        return float(prob)
    except Exception:
        return None
