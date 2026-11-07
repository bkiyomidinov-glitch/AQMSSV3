"""
Unit tests for the core AMQSS mathematical scoring engine.
Verifies calculation of 5 factors, adaptive weights, boundary constraints, and resilience to anomalies.
"""

import pytest
import numpy as np
import pandas as pd

from backend.core.scoring import (
    calculate_amqss_scores,
    clean_price_dataframe,
    normalize_weights,
    build_adaptive_weights,
    DEFAULT_BASE_WEIGHTS
)


def generate_sample_candles(n_bars: int = 100, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV candle data for testing."""
    rng = np.random.RandomState(seed)
    base_price = 100.0
    prices = [base_price]
    for _ in range(n_bars):
        ret = rng.normal(trend, 0.005)
        prices.append(prices[-1] * (1.0 + ret))

    candles = []
    for i in range(n_bars):
        o = prices[i]
        c = prices[i + 1]
        h = max(o, c) * (1.0 + abs(rng.normal(0, 0.002)))
        l = min(o, c) * (1.0 - abs(rng.normal(0, 0.002)))
        v = float(rng.uniform(1000, 5000))
        candles.append({
            'time': f'2026-01-01 {i:02d}:00:00',
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': v
        })
    return pd.DataFrame(candles)


class TestScoringCore:
    def test_clean_price_dataframe_casts_numeric_and_drops_corrupt_data(self):
        """Verify cleaning casts strings to floats and removes invalid bars (high < low)."""
        corrupted_data = pd.DataFrame([
            {'time': '2026-01-01 00:00', 'open': '100.5', 'high': '102.0', 'low': '99.0', 'close': '101.0', 'volume': '5000'},
            {'time': '2026-01-01 01:00', 'open': '101.0', 'high': '95.0', 'low': '98.0', 'close': '96.0', 'volume': '4000'}, # high < low
            {'time': '2026-01-01 02:00', 'open': '96.0', 'high': '97.0', 'low': '95.0', 'close': '96.5', 'volume': '3000'}
        ])
        cleaned = clean_price_dataframe(corrupted_data)
        assert len(cleaned) == 2
        assert np.issubdtype(cleaned['close'].dtype, np.number)
        assert (cleaned['high'] >= cleaned['low']).all()

    def test_normalize_weights_sums_to_one(self):
        """Verify weight normalization guarantees sum == 1.0."""
        raw = {'structure': 10.0, 'momentum': 5.0, 'liquidity': 0.0}
        norm = normalize_weights(raw, min_weight=0.01)
        assert pytest.approx(sum(norm.values()), rel=1e-5) == 1.0
        assert norm['liquidity'] >= 0.01

    def test_calculate_amqss_scores_outputs_valid_ranges(self):
        """Verify all 5 factor scores and total score are strictly bounded in [0, 10]."""
        df = generate_sample_candles(n_bars=80)
        res = calculate_amqss_scores(df)

        assert 'total_score' in res
        assert 0.0 <= res['total_score'] <= 10.0

        scores = res['scores']
        for factor in ['volatility_regime', 'true_liquidity', 'structure', 'momentum', 'volatility_level']:
            assert factor in scores, f"Missing factor {factor}"
            assert 0.0 <= scores[factor] <= 10.0, f"Factor {factor} out of range: {scores[factor]}"

        assert res['market_condition'] in {'NO TRADE', 'WEAK CONDITIONS', 'NORMAL MARKET', 'HIGH QUALITY MARKET'}
        assert pytest.approx(sum(res['adaptive_weights'].values()), rel=1e-4) == 1.0

    def test_scoring_handles_strong_bullish_trend(self):
        """Strong trending market should produce elevated structure score."""
        df_trend = generate_sample_candles(n_bars=80, trend=0.015, seed=123)
        res = calculate_amqss_scores(df_trend)
        assert res['scores']['structure'] > 5.0
        assert res['metrics']['trend_strength'] > 0.3

    def test_insufficient_data_raises_error(self):
        """Should raise ValueError if dataframe has fewer candles than minimum required window."""
        df_short = generate_sample_candles(n_bars=15)
        with pytest.raises(ValueError):
            calculate_amqss_scores(df_short)
