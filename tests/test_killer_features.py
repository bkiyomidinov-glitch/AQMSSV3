"""
Automated Test Suite for Step 4 Killer Features:
1. Copilot 2.0 Live Matrix & Deterministic Quant Engine
2. Regime Shift Alert Manager & Webhook Dispatcher
3. Cryptographic Billing HMAC Verification & 30-Day Subscription Activation
4. API Endpoints Integration (/api/ai/chat, /api/alerts, /api/billing/webhook)
"""

import hmac
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from backend.services.copilot import (
    find_symbol_in_text,
    find_symbols_for_comparison,
    format_ticker_analysis,
    format_comparison,
    format_top_opportunities,
    format_risk_warnings,
    query_local_quant_copilot,
    generate_copilot_response
)
from backend.services.alerts import (
    RegimeAlertManager,
    send_webhook_notification
)
from backend.services.billing import (
    verify_webhook_signature,
    extract_payment_info,
    activate_user_subscription,
    process_billing_webhook
)
import os
import secrets
from backend.api.app import app, init_db, auth_db, hash_password, AUTH_DB, DATA_DIR


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_markets():
    """Deterministic sample markets representing diverse regimes."""
    return [
        {
            'symbol': 'BTCUSD',
            'name': 'Bitcoin',
            'price': 65000.0,
            'change': 3.5,
            'quality_score': 82.4,
            'market_condition': 'HIGH_QUALITY_MARKET',
            'liquidity': 88.0,
            'volatility': 22.0,
            'structure': 85.0,
            'momentum': 'BULLISH',
            'momentum_score': 8.5,
            'ai_high_quality_prob': 0.88
        },
        {
            'symbol': 'EURUSD',
            'name': 'EUR/USD',
            'price': 1.0850,
            'change': -0.12,
            'quality_score': 63.5,
            'market_condition': 'NORMAL_MARKET',
            'liquidity': 72.0,
            'volatility': 18.0,
            'structure': 60.0,
            'momentum': 'NEUTRAL',
            'momentum_score': 5.0,
            'ai_high_quality_prob': 0.65
        },
        {
            'symbol': 'XAUUSD',
            'name': 'Gold Spot',
            'price': 2350.0,
            'change': 1.2,
            'quality_score': 77.0,
            'market_condition': 'HIGH_QUALITY_MARKET',
            'liquidity': 80.0,
            'volatility': 25.0,
            'structure': 75.0,
            'momentum': 'BULLISH',
            'momentum_score': 7.8,
            'ai_high_quality_prob': 0.79
        },
        {
            'symbol': 'TSLA',
            'name': 'Tesla Inc',
            'price': 175.0,
            'change': -4.5,
            'quality_score': 35.0,
            'market_condition': 'NO_TRADE_ZONE',
            'liquidity': 50.0,
            'volatility': 48.0,
            'structure': 30.0,
            'momentum': 'BEARISH',
            'momentum_score': 3.0,
            'ai_high_quality_prob': 0.22
        }
    ]


@pytest.fixture
def test_db():
    """Create a temporary SQLite database with users table."""
    with tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False) as f:
        db_path = f.name

    with sqlite3.connect(db_path) as conn:
        conn.execute('''CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 0,
            is_admin INTEGER NOT NULL DEFAULT 0,
            subscription_expires_at TEXT,
            created_at TEXT NOT NULL
        )''')
        # Insert test users: one inactive, one with future expiry
        now_utc = datetime.now(timezone.utc)
        future_iso = (now_utc + timedelta(days=10)).isoformat()
        conn.execute(
            "INSERT INTO users (email, password_hash, is_active, subscription_expires_at, created_at) VALUES (?, ?, ?, ?, ?)",
            ("trader@test.com", "hash123", 0, None, now_utc.isoformat())
        )
        conn.execute(
            "INSERT INTO users (email, password_hash, is_active, subscription_expires_at, created_at) VALUES (?, ?, ?, ?, ?)",
            ("active@test.com", "hash456", 1, future_iso, now_utc.isoformat())
        )
        conn.commit()

    yield db_path

    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def client():
    """Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def authenticated_client(client):
    """Create a verified test user and log in for API testing."""
    db = auth_db()
    test_email = f"test_copilot_{secrets.token_hex(4)}@test.com"
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


# ── 1. Copilot 2.0 Test Suite ─────────────────────────────────────────────────

class TestCopilotEngine:
    def test_symbol_extraction_aliases(self):
        assert find_symbol_in_text("Что думаешь про биткоин сегодня?") == "BTCUSD"
        assert find_symbol_in_text("Какой прогноз по золоту?") == "XAUUSD"
        assert find_symbol_in_text("Check TSLA momentum") == "TSLA"
        assert find_symbol_in_text("EUR looking weak") == "EURUSD"
        assert find_symbol_in_text("Random non-market text") is None

    def test_pairwise_comparison_extraction(self):
        syms = find_symbols_for_comparison("Сравни биткоин и золото")
        assert "BTCUSD" in syms
        assert "XAUUSD" in syms
        assert len(syms) == 2

    def test_ticker_analysis_formatting(self, mock_markets):
        btc = mock_markets[0]
        text = format_ticker_analysis(btc)
        assert "BTCUSD" in text
        assert "82.4/100" in text
        assert "Structure" in text
        assert "Momentum" in text
        assert "True Liquidity" in text
        assert "ВЫСОКИЙ ПРИОРИТЕТ ДЛЯ ВХОДА" in text

    def test_pairwise_comparison_formatting(self, mock_markets):
        btc = mock_markets[0]
        eur = mock_markets[1]
        text = format_comparison(btc, eur)
        assert "BTCUSD vs EURUSD" in text
        assert "Победитель по матрице качества: BTCUSD" in text
        assert "82.4" in text
        assert "63.5" in text

    def test_top_opportunities_formatting(self, mock_markets):
        text = format_top_opportunities(mock_markets)
        assert "Топ возможностей рынка" in text
        assert "BTCUSD" in text
        assert "XAUUSD" in text

    def test_risk_warnings_formatting(self, mock_markets):
        text = format_risk_warnings(mock_markets)
        assert "Внимание: Рынки с повышенным риском" in text
        assert "TSLA" in text
        assert "NO_TRADE_ZONE" in text

    def test_query_local_quant_copilot_deterministic(self, mock_markets):
        # 1. Comparison
        resp_comp = query_local_quant_copilot("Сравни BTC и EUR", mock_markets)
        assert "Сравнение инструментов" in resp_comp
        assert "BTCUSD" in resp_comp

        # 2. Single Ticker
        resp_ticker = query_local_quant_copilot("Проанализируй XAUUSD", mock_markets)
        assert "Анализ тикера XAUUSD" in resp_ticker
        assert "77.0/100" in resp_ticker

        # 3. Best opportunities
        resp_best = query_local_quant_copilot("Какой лучший рынок для покупки?", mock_markets)
        assert "Топ возможностей" in resp_best

        # 4. Risk / Stop
        resp_risk = query_local_quant_copilot("Где зона повышенного риска?", mock_markets)
        assert "повышенным риском" in resp_risk

        # 5. Default general
        resp_default = query_local_quant_copilot("Привет, что ты умеешь?", mock_markets)
        assert "AQMSS Quant Copilot 2.0" in resp_default

    def test_copilot_fallback_when_groq_absent(self, mock_markets):
        # Without groq client, must gracefully produce local quant response
        resp = generate_copilot_response(
            user_message="Что с битком?",
            session_id="test_sess",
            live_markets=mock_markets,
            groq_client=None
        )
        assert "BTCUSD" in resp
        assert "82.4/100" in resp


# ── 2. Regime Alert Manager Test Suite ────────────────────────────────────────

class TestRegimeAlertManager:
    def test_regime_shift_detection(self):
        manager = RegimeAlertManager(max_history=50)

        # Round 1: establish baseline
        round_1 = [
            {'symbol': 'BTCUSD', 'quality_score': 62.0, 'market_condition': 'NORMAL_MARKET', 'momentum': 'NEUTRAL', 'price': 60000.0}
        ]
        alerts_1 = manager.detect_shifts(round_1)
        assert len(alerts_1) == 0  # No baseline previous state yet

        # Round 2: Shift to HIGH_QUALITY_MARKET
        round_2 = [
            {'symbol': 'BTCUSD', 'quality_score': 81.0, 'market_condition': 'HIGH_QUALITY_MARKET', 'momentum': 'BULLISH', 'price': 64000.0}
        ]
        alerts_2 = manager.detect_shifts(round_2)
        assert len(alerts_2) == 1
        alert = alerts_2[0]
        assert alert['symbol'] == 'BTCUSD'
        assert alert['type'] == 'REGIME_SHIFT'
        assert alert['severity'] == 'HIGH'
        assert alert['old_condition'] == 'NORMAL_MARKET'
        assert alert['new_condition'] == 'HIGH_QUALITY_MARKET'
        assert alert['old_score'] == 62.0
        assert alert['new_score'] == 81.0

        # Round 3: Shift into NO_TRADE_ZONE
        round_3 = [
            {'symbol': 'BTCUSD', 'quality_score': 32.0, 'market_condition': 'NO_TRADE_ZONE', 'momentum': 'BEARISH', 'price': 58000.0}
        ]
        alerts_3 = manager.detect_shifts(round_3)
        assert len(alerts_3) == 1
        assert alerts_3[0]['new_condition'] == 'NO_TRADE_ZONE'
        assert alerts_3[0]['severity'] == 'HIGH'

    def test_score_surge_alert(self):
        manager = RegimeAlertManager(max_history=50)
        manager.detect_shifts([
            {'symbol': 'ETHUSD', 'quality_score': 55.0, 'market_condition': 'NORMAL_MARKET', 'momentum': 'NEUTRAL', 'price': 3000.0}
        ])

        # Sudden surge >= 15 pts within same condition
        surge_round = [
            {'symbol': 'ETHUSD', 'quality_score': 74.0, 'market_condition': 'NORMAL_MARKET', 'momentum': 'NEUTRAL', 'price': 3300.0}
        ]
        alerts = manager.detect_shifts(surge_round)
        assert len(alerts) == 1
        assert alerts[0]['type'] == 'SCORE_SURGE'
        assert alerts[0]['old_score'] == 55.0
        assert alerts[0]['new_score'] == 74.0

    def test_alert_manager_configuration_and_history(self):
        manager = RegimeAlertManager(max_history=10)
        manager.configure_telegram("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", "987654321")
        manager.configure_webhook("https://webhook.site/test-uuid", secret="sec123")

        cfg = manager.get_config()
        assert cfg['telegram_configured'] is True
        assert cfg['telegram_chat_id'] == "987654321"
        assert cfg['webhook_configured'] is True
        assert cfg['webhook_url'] == "https://webhook.site/test-uuid"


# ── 3. Billing & HMAC Webhook Test Suite ──────────────────────────────────────

class TestBillingAndSubscription:
    def test_verify_webhook_signature_sha256(self):
        secret = "super_secure_billing_secret_99"
        payload = b'{"status": "paid", "email": "user@test.com"}'
        sig = hmac.new(secret.encode('utf-8'), payload, hashlib.sha256).hexdigest()

        # Direct hex match
        assert verify_webhook_signature(payload, sig, secret) is True
        # sha256= prefix match
        assert verify_webhook_signature(payload, f"sha256={sig}", secret) is True
        # Tampered body must fail
        assert verify_webhook_signature(payload + b"x", sig, secret) is False
        # Wrong secret must fail
        assert verify_webhook_signature(payload, sig, "wrong_secret") is False
        # Empty inputs must fail
        assert verify_webhook_signature(payload, None, secret) is False

    def test_verify_webhook_signature_stripe_format(self):
        secret = "whsec_test_stripe_secret"
        body = b'{"type": "checkout.session.completed"}'
        timestamp = "1700000000"
        signed_payload = f"{timestamp}.".encode('utf-8') + body
        sig = hmac.new(secret.encode('utf-8'), signed_payload, hashlib.sha256).hexdigest()
        stripe_header = f"t={timestamp},v1={sig}"

        assert verify_webhook_signature(body, stripe_header, secret) is True
        # Invalid signature must fail
        assert verify_webhook_signature(body, f"t={timestamp},v1=bad_hash", secret) is False

    def test_extract_payment_info_gateways(self):
        # 1. NowPayments format
        np_payload = {
            'payment_status': 'finished',
            'payin_extra_id': 'trader@test.com',
            'pay_amount': 19.0,
            'pay_currency': 'USDT'
        }
        res_np = extract_payment_info(np_payload)
        assert res_np['is_confirmed'] is True
        assert res_np['email'] == 'trader@test.com'
        assert res_np['amount'] == 19.0

        # 2. Stripe format
        stripe_payload = {
            'type': 'checkout.session.completed',
            'data': {
                'object': {
                    'customer_details': {'email': 'stripe_user@test.com'},
                    'amount_total': 1900,
                    'currency': 'usd'
                }
            }
        }
        res_stripe = extract_payment_info(stripe_payload)
        assert res_stripe['is_confirmed'] is True
        assert res_stripe['email'] == 'stripe_user@test.com'
        assert res_stripe['amount'] == 19.0

        # 3. LemonSqueezy format
        ls_payload = {
            'meta': {'event_name': 'order_created'},
            'data': {
                'attributes': {
                    'user_email': 'ls_user@test.com',
                    'total_formatted': '$19.00'
                }
            }
        }
        res_ls = extract_payment_info(ls_payload)
        assert res_ls['is_confirmed'] is True
        assert res_ls['email'] == 'ls_user@test.com'

    def test_activate_user_subscription(self, test_db):
        # 1. Activate previously inactive user
        res = activate_user_subscription(test_db, "trader@test.com", days=30)
        assert res['success'] is True
        assert res['is_active'] == 1
        new_expiry_str = res['subscription_expires_at']
        new_expiry = datetime.fromisoformat(new_expiry_str)
        # Should be roughly 30 days from now
        now_utc = datetime.now(timezone.utc)
        assert 29 <= (new_expiry - now_utc).days <= 31

        # Verify DB persisted
        with sqlite3.connect(test_db) as conn:
            row = conn.execute("SELECT is_active, subscription_expires_at FROM users WHERE email = 'trader@test.com'").fetchone()
            assert row[0] == 1
            assert row[1] == new_expiry_str

        # 2. Extend already active user with future expiry (10 days remaining -> should become 40 days)
        res_extend = activate_user_subscription(test_db, "active@test.com", days=30)
        assert res_extend['success'] is True
        extended_expiry = datetime.fromisoformat(res_extend['subscription_expires_at'])
        assert 39 <= (extended_expiry - now_utc).days <= 41

    def test_process_billing_webhook_full(self, test_db):
        secret = "webhook_secret_key_abc"
        raw_payload = json.dumps({
            'payment_status': 'confirmed',
            'email': 'trader@test.com',
            'amount': 19.0
        }).encode('utf-8')
        sig = hmac.new(secret.encode('utf-8'), raw_payload, hashlib.sha256).hexdigest()

        # Success case
        headers = {'X-NowPayments-Sig': sig, 'Content-Type': 'application/json'}
        success, msg, code = process_billing_webhook(
            headers=headers,
            raw_body=raw_payload,
            payload=json.loads(raw_payload.decode('utf-8')),
            db_path=test_db,
            secret=secret
        )
        assert success is True
        assert code == 200

        # Invalid signature case
        bad_headers = {'X-NowPayments-Sig': 'fake_signature', 'Content-Type': 'application/json'}
        bad_ok, bad_msg, bad_code = process_billing_webhook(
            headers=bad_headers,
            raw_body=raw_payload,
            payload=json.loads(raw_payload.decode('utf-8')),
            db_path=test_db,
            secret=secret
        )
        assert bad_ok is False
        assert bad_code == 401


# ── 4. API Endpoints Integration Test Suite ───────────────────────────────────

class TestKillerFeaturesAPI:
    def test_ai_chat_copilot_endpoint(self, authenticated_client):
        with authenticated_client.session_transaction() as sess:
            token = sess.get('csrf_token')

        resp = authenticated_client.post(
            '/api/ai/chat',
            json={'message': 'Сравни EURUSD и BTCUSD', 'csrf_token': token}
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        reply = data['data']['ai_response']
        # Must contain both compared assets and scoring analysis (via Groq LLM or local quant engine)
        assert 'EURUSD' in reply and 'BTCUSD' in reply
        assert any(k in reply.lower() for k in ['score', 'качества', 'моментум', 'сравнение'])

    def test_alerts_endpoint(self, authenticated_client):
        resp = authenticated_client.get('/api/alerts')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert 'alerts' in data
        assert 'config' in data

    def test_billing_webhook_api_rejection_on_invalid_sig(self, client):
        # External webhooks must NOT require browser CSRF token, but MUST require valid HMAC
        resp = client.post(
            '/api/billing/webhook',
            data=b'{"payment_status": "confirmed"}',
            headers={'Content-Type': 'application/json', 'X-NowPayments-Sig': 'invalid_signature'}
        )
        assert resp.status_code == 401
        data = resp.get_json()
        assert data['success'] is False
        assert 'Invalid HMAC' in data['message']

    def test_billing_webhook_api_activation_success(self, client):
        # Create an inactive user in db
        db = auth_db()
        test_email = f"webhook_trader_{secrets.token_hex(4)}@test.com"
        db.execute(
            "INSERT INTO users (email, password_hash, is_active, created_at) VALUES (?, ?, 0, '2026-01-01T00:00:00')",
            (test_email, hash_password("secretpass"))
        )
        db.commit()
        db.close()

        secret = os.environ.get('AQMSS_BILLING_WEBHOOK_SECRET', 'test_billing_secret_key')
        raw_payload = json.dumps({
            'payment_status': 'confirmed',
            'email': test_email,
            'amount': 19.0
        }).encode('utf-8')
        sig = hmac.new(secret.encode('utf-8'), raw_payload, hashlib.sha256).hexdigest()

        resp = client.post(
            '/api/billing/webhook',
            data=raw_payload,
            headers={'Content-Type': 'application/json', 'X-NowPayments-Sig': sig}
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True

        # Verify DB is updated to is_active=1 and subscription date is populated
        db = auth_db()
        row = db.execute("SELECT is_active, subscription_expires_at FROM users WHERE email = ?", (test_email,)).fetchone()
        assert row['is_active'] == 1
        assert row['subscription_expires_at'] is not None
        db.execute("DELETE FROM users WHERE email = ?", (test_email,))
        db.commit()
        db.close()
