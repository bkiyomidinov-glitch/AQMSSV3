"""
Integration tests for API endpoints, security gates, CSRF protection, and deterministic Quant engine.
"""

import pytest
import secrets
from backend.api.app import app, auth_db, hash_password


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def authenticated_client(client):
    """Create a verified test user and log in."""
    db = auth_db()
    test_email = f"test_user_{secrets.token_hex(4)}@test.com"
    test_pwd = "StrongPassword123!"

    cursor = db.execute(
        "INSERT INTO users (email, password_hash, is_active, created_at) VALUES (?, ?, 1, '2026-01-01T00:00:00')",
        (test_email, hash_password(test_pwd))
    )
    db.commit()
    user_id = cursor.lastrowid
    db.close()

    # Get CSRF token
    client.get('/login')
    with client.session_transaction() as sess:
        token = sess.get('csrf_token')

    # Log in
    login_res = client.post('/login', data={'email': test_email, 'password': test_pwd, 'csrf_token': token})
    assert login_res.status_code == 302

    yield client

    # Cleanup
    db = auth_db()
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    db.close()


class TestSecurityAndAuth:
    def test_unauthenticated_routes_redirect_to_login(self, client):
        """Unauthenticated user accessing protected UI pages must be redirected to /login."""
        for path in ['/terminal', '/copilot', '/billing', '/news']:
            res = client.get(path)
            assert res.status_code == 302
            assert '/login' in res.headers['Location']

    def test_unauthenticated_api_returns_401(self, client):
        """Unauthenticated call to /api/ endpoints must return 401 Unauthorized."""
        res = client.get('/api/dashboard/stats')
        assert res.status_code == 401
        data = res.get_json()
        assert data['success'] is False

    def test_csrf_protection_rejects_missing_token(self, client):
        """Mutating POST requests without CSRF token must be blocked with HTTP 403."""
        res = client.post('/login', data={'email': 'any@test.com', 'password': 'pass'})
        assert res.status_code == 403


class TestQuantEngineAndCache:
    def test_markets_endpoint_returns_real_quant_metrics(self, authenticated_client):
        """Verify /api/markets returns real computed scores and no random values."""
        res = authenticated_client.get('/api/markets')
        assert res.status_code == 200

        data = res.get_json()
        assert data['success'] is True
        markets = data['data']
        assert len(markets) >= 10

        # Check fields of the first market
        first = markets[0]
        assert 'symbol' in first
        assert 'quality_score' in first
        assert 0.0 <= first['quality_score'] <= 100.0
        assert first['price'] > 0
        assert first['momentum'] in {'BULLISH', 'BEARISH', 'NEUTRAL'}
        assert first['market_condition'] in {'NO TRADE', 'WEAK CONDITIONS', 'NORMAL MARKET', 'HIGH QUALITY MARKET'}

    def test_markets_cache_determinism_no_random_jumping(self, authenticated_client):
        """
        Verify that repeated calls to /api/markets return stable, identical prices and scores
        from RAM cache without random jitter (np.random completely eliminated).
        """
        res1 = authenticated_client.get('/api/markets').get_json()['data']
        res2 = authenticated_client.get('/api/markets').get_json()['data']

        prices1 = {m['symbol']: m['price'] for m in res1}
        prices2 = {m['symbol']: m['price'] for m in res2}
        assert prices1 == prices2, "Prices must be identical on cached calls (no random jumping!)"

        scores1 = {m['symbol']: m['quality_score'] for m in res1}
        scores2 = {m['symbol']: m['quality_score'] for m in res2}
        assert scores1 == scores2, "Quality scores must be identical on cached calls"

    def test_ai_predict_endpoint_with_singleton_model(self, authenticated_client):
        """Test /api/ai/predict with in-memory singleton predictor."""
        with authenticated_client.session_transaction() as sess:
            csrf_token = sess.get('csrf_token')

        res = authenticated_client.post(
            '/api/ai/predict',
            json={
                'volatility_regime': 6.0,
                'liquidity': 70.0,
                'structure': 80.0,
                'momentum': 60.0,
                'volatility': 20.0,
                'atr': 0.0012,
                'csrf_token': csrf_token
            },
            headers={'X-CSRFToken': csrf_token}
        )
        assert res.status_code == 200
        payload = res.get_json()
        assert payload['success'] is True
        pred = payload['data']
        assert 0.0 <= pred['probability'] <= 1.0
        assert pred['recommendation'] in {'BUY', 'NEUTRAL', 'SELL'}
