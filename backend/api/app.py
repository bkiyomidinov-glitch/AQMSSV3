"""
AQMSS TradingView-Style Web Dashboard API
Integrates with real AQMSS AI Market Quality Scoring System
Reads real data from market_scores.csv, dataset.csv, models/metrics.json
"""

import os
import sys
import json
import random
import uuid
import sqlite3
import hashlib
import secrets
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / 'frontend'
TEMPLATES_DIR = FRONTEND_DIR / 'templates'
STATIC_DIR = FRONTEND_DIR / 'static'
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = DATA_DIR
MODELS_DIR = PROJECT_ROOT / 'backend' / 'ml'
AUTH_DB = DATA_DIR / 'aqmss_auth.sqlite3'

load_dotenv(PROJECT_ROOT / '.env')
load_dotenv(Path(__file__).parent / '.env')

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.geolocation import lookup_ip_details, update_user_geo_async
from backend.services.market_feed import fetch_yahoo_candles
from backend.services.cache import market_cache
from backend.services.copilot import generate_copilot_response
from backend.services.alerts import alert_manager
from backend.services.billing import process_billing_webhook

# ── Groq LLM Configuration (Read once at startup) ─────────────────────────────
try:
    from groq import Groq as GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_MODEL = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-20b')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

def make_groq_client():
    """Create a fresh Groq client. Returns None if not available."""
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return None
    return GroqClient(api_key=GROQ_API_KEY)

# LRU Cache for chat histories (Memory leak prevention)
_chat_histories: OrderedDict = OrderedDict()
MAX_CHAT_SESSIONS = 200

def get_chat_history(session_id: str):
    if session_id in _chat_histories:
        _chat_histories.move_to_end(session_id)
        return _chat_histories[session_id]
    if len(_chat_histories) >= MAX_CHAT_SESSIONS:
        _chat_histories.popitem(last=False)
    _chat_histories[session_id] = []
    return _chat_histories[session_id]

# Initialize singleton ML predictor in memory
predictor = None
AI_AVAILABLE = False
try:
    from backend.ml.predictor import load_predictor
    if (MODELS_DIR / 'model.pkl').exists():
        predictor = load_predictor(str(MODELS_DIR))
        AI_AVAILABLE = True
        print(f"[OK] AMQSS Predictor loaded successfully from {MODELS_DIR}", flush=True)
    elif (PROJECT_ROOT / 'models' / 'model.pkl').exists():
        predictor = load_predictor(str(PROJECT_ROOT / 'models'))
        AI_AVAILABLE = True
        print(f"[OK] AMQSS Predictor loaded successfully from {PROJECT_ROOT / 'models'}", flush=True)
    else:
        print(f"[WARN] AMQSS model artifacts not found in {MODELS_DIR}", flush=True)
except Exception as e:
    print(f"[WARN] AMQSS Predictor unavailable: {e}", flush=True)

def get_or_create_secret_key() -> str:
    """Return persistent secret key so sessions survive across all Gunicorn workers."""
    env_key = os.environ.get('AQMSS_SECRET_KEY')
    if env_key:
        return env_key
    key_file = DATA_DIR / '.secret_key'
    if key_file.exists():
        try:
            key = key_file.read_text(encoding='utf-8').strip()
            if key:
                return key
        except Exception:
            pass
    new_key = secrets.token_hex(32)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        key_file.write_text(new_key, encoding='utf-8')
        try:
            os.chmod(key_file, 0o600)
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: could not write .secret_key: {e}")
    return new_key

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR)
)
# Protect against IP spoofing with 1 proxy hop
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app)

app.secret_key = get_or_create_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('SESSION_COOKIE_SECURE', 'false').lower() in {'true', '1'}
)

PAYMENT_ADDRESS = os.environ.get('USDT_TRC20_ADDRESS', 'PASTE_YOUR_TRC20_ADDRESS_HERE')
SUBSCRIPTION_PRICE = os.environ.get('SUBSCRIPTION_PRICE_USDT', '19')

# ── SQLite Database Setup (Init Schema ONCE at startup) ───────────────────────
def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUTH_DB, timeout=10) as db:
        db.execute('PRAGMA journal_mode = WAL;')
        db.execute('PRAGMA busy_timeout = 5000;')
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 0,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )''')
        columns = {row[1] for row in db.execute('PRAGMA table_info(users)').fetchall()}
        for field in ('last_ip', 'country', 'region', 'city', 'isp', 'asn', 'ip_flags', 'subscription_expires_at'):
            if field not in columns:
                db.execute(f'ALTER TABLE users ADD COLUMN {field} TEXT')
        if 'is_admin' not in columns:
            db.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0')

        # Alert configuration table
        db.execute('''CREATE TABLE IF NOT EXISTS alert_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            telegram_bot_token TEXT,
            telegram_chat_id TEXT,
            webhook_url TEXT,
            webhook_secret TEXT
        )''')
        db.commit()

init_db()

def load_alert_settings():
    """Load alert configuration from SQLite and environment variables."""
    try:
        with auth_db() as db:
            row = db.execute('SELECT telegram_bot_token, telegram_chat_id, webhook_url, webhook_secret FROM alert_settings WHERE id = 1').fetchone()
            if row:
                if row['telegram_bot_token'] and row['telegram_chat_id']:
                    alert_manager.configure_telegram(row['telegram_bot_token'], row['telegram_chat_id'])
                if row['webhook_url']:
                    alert_manager.configure_webhook(row['webhook_url'], row['webhook_secret'])
    except Exception:
        pass

    env_tg_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    env_tg_chat = os.environ.get('TELEGRAM_CHAT_ID')
    if env_tg_token and env_tg_chat:
        alert_manager.configure_telegram(env_tg_token, env_tg_chat)
    env_wh_url = os.environ.get('WEBHOOK_URL')
    if env_wh_url:
        alert_manager.configure_webhook(env_wh_url, os.environ.get('WEBHOOK_SECRET'))

load_alert_settings()

def auth_db():
    """Fast connection with WAL & timeout, without per-request schema migrations."""
    db = sqlite3.connect(AUTH_DB, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA busy_timeout = 5000;')
    return db

# ── Password Hashing with Transparent Legacy SHA-256 Migration ────────────────
def hash_password(password: str) -> str:
    """Hash password using Werkzeug's secure pbkdf2 algorithm with salt."""
    return generate_password_hash(password, method='pbkdf2:sha256')

def verify_and_migrate_password(db, user, password: str) -> bool:
    """Verify password and migrate legacy 64-char SHA-256 hashes to pbkdf2 on login."""
    stored_hash = user['password_hash']
    # Detect legacy unsalted SHA-256 (64 hex characters)
    if len(stored_hash) == 64 and all(c in '0123456789abcdefABCDEF' for c in stored_hash):
        legacy_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
        if secrets.compare_digest(legacy_hash, stored_hash):
            new_hash = hash_password(password)
            db.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user['id']))
            db.commit()
            return True
        return False
    return check_password_hash(stored_hash, password)

def request_ip():
    """Get real client IP validated through ProxyFix."""
    return request.remote_addr or 'unknown'

def ip_details(ip):
    return lookup_ip_details(ip)

# ── CSRF Protection ───────────────────────────────────────────────────────────
def get_csrf_token() -> str:
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(24)
    return session['csrf_token']

app.jinja_env.globals['csrf_token'] = get_csrf_token

@app.after_request
def set_csrf_cookie(response):
    if 'csrf_token' in session:
        response.set_cookie('csrf_token', session['csrf_token'], samesite='Lax', httponly=False)
    return response

@app.before_request
def verify_csrf():
    # Exclude machine-to-machine payment webhooks verified by cryptographic HMAC
    if request.path in {'/api/billing/webhook'}:
        return
    if request.method in {'POST', 'PUT', 'DELETE', 'PATCH'}:
        token = (
            request.form.get('csrf_token') or
            request.headers.get('X-CSRFToken') or
            request.headers.get('X-CSRF-Token')
        )
        if not token and request.is_json:
            token = (request.json or {}).get('csrf_token')

        expected = session.get('csrf_token')
        if not token or not expected or not secrets.compare_digest(token, expected):
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Invalid or missing CSRF token'}), 403
            return render_template(
                'auth.html',
                mode='login',
                error='Сессия формы устарела. Пожалуйста, обновите страницу и попробуйте снова.'
            ), 403


def current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    db = auth_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    db.close()
    return user


def require_login():
    if not current_user():
        return redirect(url_for('login_page', next=request.path))
    return None


@app.before_request
def protect_api():
    if not request.path.startswith('/api/'):
        return None
    # External machine-to-machine payment webhooks are authenticated cryptographically via HMAC
    if request.path in {'/api/billing/webhook'}:
        return None
    user = current_user()
    if not user:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    return None

# ==========================================
# REAL AQMSS DATA LOADING
# ==========================================

def load_aqmss_scores():
    """Load real AQMSS scores from market_scores.csv"""
    for candidate in [DATA_DIR / 'market_scores.csv', PROJECT_ROOT / 'results' / 'market_scores.csv', PROJECT_ROOT / 'market_scores.csv']:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                if not df.empty:
                    return df
            except Exception as e:
                print(f"Error loading AQMSS scores from {candidate}: {e}")
    return None


def load_model_metrics():
    """Load real AI model metrics from metrics.json"""
    for candidate in [MODELS_DIR / 'metrics.json', PROJECT_ROOT / 'models' / 'metrics.json']:
        if candidate.exists():
            try:
                with open(candidate, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading model metrics: {e}")
    return None


def load_model_metadata():
    """Load model metadata from metadata.json"""
    for candidate in [MODELS_DIR / 'metadata.json', PROJECT_ROOT / 'models' / 'metadata.json']:
        if candidate.exists():
            try:
                with open(candidate, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading model metadata: {e}")
    return None


def load_training_data():
    """Load training dataset for stats"""
    dataset_file = DATA_DIR / 'dataset.csv'
    if not dataset_file.exists():
        return None
    try:
        return pd.read_csv(dataset_file)
    except Exception:
        return None


def get_latest_aqmss():
    """Get the most recent AQMSS analysis result"""
    df = load_aqmss_scores()
    if df is None or df.empty:
        return None

    latest = df.iloc[-1]
    return {
        'timestamp': str(latest.get('timestamp', '')),
        'total_score': float(latest.get('total_score', 0)),
        'volatility_regime': float(latest.get('volatility_regime', 0)),
        'true_liquidity': float(latest.get('true_liquidity', 0)),
        'structure': float(latest.get('structure', 0)),
        'momentum': float(latest.get('momentum', 0)),
        'volatility_level': float(latest.get('volatility_level', 0)),
        'market_condition': str(latest.get('market_condition', 'UNKNOWN')),
        'current_atr': float(latest.get('current_atr', 0)),
        'mean_atr': float(latest.get('mean_atr', 0)),
        'atr_std': float(latest.get('atr_std', 0)),
        'recent_volume': float(latest.get('recent_volume', 0)),
        'ai_high_quality_prob': float(latest['ai_high_quality_prob']) if pd.notna(latest.get('ai_high_quality_prob')) else None
    }


# ==========================================
# REAL MULTI-MARKET QUANT ENGINE & CACHE
# ==========================================

from backend.services.market_feed import compute_all_markets_live

def _compute_and_alert():
    markets = compute_all_markets_live(predictor=predictor)
    try:
        alert_manager.process_market_updates(markets)
    except Exception as e:
        print(f"[AlertManager] Error processing updates: {e}", flush=True)
    return markets

def get_cached_markets():
    """
    Get real calculated AMQSS markets from RAM cache with Stale-While-Revalidate.
    Guarantees sub-5ms responses and non-blocking background refreshes.
    """
    return market_cache.get_or_compute_swr(
        key='all_markets_quant',
        compute_fn=_compute_and_alert,
        ttl=90
    )


@app.route('/api/chart/<symbol>', methods=['GET'])
def chart_data(symbol):
    """Return real OHLC candles for the selected market."""
    interval = request.args.get('interval', '15m')
    candles = fetch_yahoo_candles(symbol, interval=interval)
    if candles is None:
        return jsonify({'success': False, 'error': 'Live candle feed unavailable'}), 502
    return jsonify({
        'success': True,
        'symbol': symbol.upper(),
        'interval': interval,
        'source': 'Yahoo Finance',
        'data': candles
    })


# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/terminal')
def terminal_page():
    gate = require_login()
    if gate:
        return gate
    return render_template('terminal.html')


@app.route('/copilot')
def copilot_page():
    gate = require_login()
    if gate:
        return gate
    return render_template('copilot.html')


@app.route('/news')
def news_page():
    gate = require_login()
    if gate:
        return gate
    return render_template('news.html')


@app.route('/test')
def test_page():
    gate = require_login()
    if gate:
        return gate
    user = current_user()
    if not user or not user['is_active']:
        return redirect(url_for('billing_page'))
    return render_template('test.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not email or len(password) < 6:
            return render_template('auth.html', mode='register', error='Введите email и пароль минимум из 6 символов.')
        db = auth_db()
        try:
            ip = request_ip()
            details = ip_details(ip)
            cursor = db.execute(
                'INSERT INTO users(email, password_hash, created_at, last_ip, country, region, city, isp, asn, ip_flags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (email, hash_password(password), datetime.utcnow().isoformat(), ip, details['country'], details['region'], details['city'], details['isp'], details['asn'], details['ip_flags'])
            )
            db.commit()
            user_id = cursor.lastrowid
            session['user_id'] = user_id
            # Non-blocking async background enrichment
            update_user_geo_async(str(AUTH_DB), user_id, ip)
            return redirect(url_for('terminal_page'))
        except sqlite3.IntegrityError:
            return render_template('auth.html', mode='register', error='Этот email уже зарегистрирован.')
        finally:
            db.close()
    return render_template('auth.html', mode='register')


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        db = auth_db()
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user and verify_and_migrate_password(db, user, password):
            ip = request_ip()
            details = ip_details(ip)
            db.execute(
                'UPDATE users SET last_ip = ?, country = ?, region = ?, city = ?, isp = ?, asn = ?, ip_flags = ? WHERE id = ?',
                (ip, details['country'], details['region'], details['city'], details['isp'], details['asn'], details['ip_flags'], user['id'])
            )
            db.commit()
            db.close()
            session['user_id'] = user['id']
            # Non-blocking async background enrichment
            update_user_geo_async(str(AUTH_DB), user['id'], ip)
            return redirect(request.args.get('next') or url_for('terminal_page'))
        db.close()
        return render_template('auth.html', mode='login', error='Неверный email или пароль.')
    return render_template('auth.html', mode='login')


@app.route('/logout')
def logout_page():
    session.clear()
    return redirect(url_for('index'))


@app.route('/billing')
def billing_page():
    gate = require_login()
    if gate:
        return gate
    return render_template('billing.html', user=current_user(), address=PAYMENT_ADDRESS, price=SUBSCRIPTION_PRICE)


@app.route('/admin', methods=['GET', 'POST'])
def admin_page():
    admin_secret = os.environ.get('AQMSS_ADMIN_KEY')
    if not admin_secret:
        return 'Admin panel is not configured on this server (missing AQMSS_ADMIN_KEY)', 503

    if not session.get('is_admin'):
        if request.method == 'POST':
            entered_key = request.form.get('admin_key', '').strip()
            if entered_key and secrets.compare_digest(entered_key, admin_secret):
                session['is_admin'] = True
                return redirect(url_for('admin_page'))
            return render_template('admin_login.html', error='Неверный ключ администратора.')
        return render_template('admin_login.html')

    db = auth_db()
    users = db.execute('SELECT id, email, is_active, last_ip, country, region, city, isp, asn, ip_flags FROM users ORDER BY id DESC').fetchall()
    db.close()
    return render_template('admin.html', users=users)


@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('index'))


@app.route('/admin/activate/<int:user_id>', methods=['POST'])
def activate_user(user_id):
    if not session.get('is_admin'):
        return 'Forbidden', 403
    db = auth_db()
    db.execute('UPDATE users SET is_active = 1 WHERE id = ?', (user_id,))
    db.commit()
    db.close()
    return redirect(url_for('admin_page'))


@app.route('/api/markets', methods=['GET'])
def get_markets():
    """Get all market data with AQMSS-based scores"""
    try:
        markets = get_cached_markets()
        return jsonify({
            'success': True,
            'data': markets,
            'count': len(markets),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/market/<symbol>', methods=['GET'])
def get_market_detail(symbol):
    """Get detailed market data for a specific symbol"""
    try:
        markets = get_cached_markets()
        market = next((m for m in markets if m['symbol'] == symbol), None)
        if market:
            return jsonify({'success': True, 'data': market})
        return jsonify({'success': False, 'error': 'Symbol not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/aqmss', methods=['GET'])
def get_aqmss_data():
    """Get real AQMSS scoring data, model metrics, and history"""
    try:
        current = get_latest_aqmss()
        model_metrics = load_model_metrics()
        model_metadata = load_model_metadata()
        training_data = load_training_data()

        history_count = 0
        if training_data is not None:
            history_count = len(training_data)

        aqmss_history = load_aqmss_scores()
        readings_count = len(aqmss_history) if aqmss_history is not None else 0

        return jsonify({
            'success': True,
            'data': {
                'current': current,
                'model_metrics': model_metrics,
                'model_metadata': model_metadata,
                'history_count': history_count,
                'readings_count': readings_count,
                'ai_available': AI_AVAILABLE
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get overall dashboard statistics from cached markets"""
    try:
        markets = get_cached_markets()

        stats = {
            'total_markets': 0,
            'avg_quality': 50,
            'high_quality_count': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'volatile_count': 0,
        }

        if markets:
            stats['total_markets'] = len(markets)
            quality_scores = [m['quality_score'] for m in markets]
            stats['avg_quality'] = sum(quality_scores) / len(quality_scores) if quality_scores else 50
            stats['high_quality_count'] = sum(1 for m in markets if m['quality_score'] > 75)
            stats['bullish_count'] = sum(1 for m in markets if m['momentum'] == 'BULLISH')
            stats['bearish_count'] = sum(1 for m in markets if m['momentum'] == 'BEARISH')
            stats['volatile_count'] = sum(1 for m in markets if m['volatility'] > 40)

        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/insights', methods=['GET'])
def get_ai_insights():
    """Get AI-generated market insights based on real AQMSS data"""
    try:
        insights = []
        markets = get_cached_markets()
        aqmss = get_latest_aqmss()

        # AQMSS Core Insight (real data)
        if aqmss:
            score = aqmss['total_score']
            cond = aqmss['market_condition']
            ai_prob = aqmss.get('ai_high_quality_prob')

            if score >= 7.5:
                desc = f'AQMSS Total Score: {score:.2f}/10 - Excellent trading conditions. '
            elif score >= 6:
                desc = f'AQMSS Total Score: {score:.2f}/10 - Normal market conditions. '
            elif score >= 4:
                desc = f'AQMSS Total Score: {score:.2f}/10 - Weak conditions, trade with caution. '
            else:
                desc = f'AQMSS Total Score: {score:.2f}/10 - NO TRADE zone. Avoid entries. '

            desc += f'Condition: {cond}. '
            if ai_prob is not None:
                desc += f'AI High Quality Probability: {ai_prob*100:.1f}%. '

            # Factor breakdown
            desc += f'Structure: {aqmss["structure"]:.1f}/10, Momentum: {aqmss["momentum"]:.1f}/10, Liquidity: {aqmss["true_liquidity"]:.1f}/10.'

            insights.append({
                'type': 'aqmss',
                'title': f'🎯 AQMSS Core Analysis: {cond}',
                'description': desc,
                'priority': 'HIGH'
            })

        if markets:
            # High quality opportunities
            high_quality = [m for m in markets if m['quality_score'] > 75]
            if high_quality:
                symbols = ', '.join([m['symbol'] for m in high_quality[:3]])
                insights.append({
                    'type': 'opportunity',
                    'title': f'🎯 {len(high_quality)} High Quality Markets',
                    'description': f'Excellent conditions in {len(high_quality)} markets. Top: {symbols}',
                    'priority': 'HIGH'
                })

            # Bullish momentum
            bullish = [m for m in markets if m['momentum'] == 'BULLISH']
            if bullish:
                symbols = ', '.join([m['symbol'] for m in bullish[:3]])
                extra = f' and {len(bullish)-3} more' if len(bullish) > 3 else ''
                insights.append({
                    'type': 'bullish',
                    'title': f'📈 {len(bullish)} Bullish Markets',
                    'description': f'Strong upward momentum in {symbols}{extra}.',
                    'priority': 'HIGH'
                })

            # Bearish warnings
            bearish = [m for m in markets if m['momentum'] == 'BEARISH']
            if bearish:
                symbols = ', '.join([m['symbol'] for m in bearish[:3]])
                insights.append({
                    'type': 'bearish',
                    'title': f'📉 {len(bearish)} Bearish Markets',
                    'description': f'Downward pressure in {symbols}. Consider defensive strategies.',
                    'priority': 'MEDIUM'
                })

            # Volatility alerts
            volatile = [m for m in markets if m['volatility'] > 40]
            if volatile:
                symbols = ', '.join([m['symbol'] for m in volatile[:3]])
                insights.append({
                    'type': 'warning',
                    'title': f'⚡ High Volatility Alert',
                    'description': f'{len(volatile)} markets with elevated volatility: {symbols}. Widen stops.',
                    'priority': 'MEDIUM'
                })

            # Poor quality warnings
            poor = [m for m in markets if m['quality_score'] < 40]
            if poor:
                symbols = ', '.join([m['symbol'] for m in poor[:3]])
                insights.append({
                    'type': 'warning',
                    'title': f'⚠️ {len(poor)} Low Quality Markets',
                    'description': f'Poor conditions in {symbols}. Avoid or use minimal size.',
                    'priority': 'LOW'
                })

            # Best liquidity
            top_liquid = sorted(markets, key=lambda x: x['liquidity'], reverse=True)[:3]
            liquid_info = ', '.join([f"{m['symbol']} ({m['liquidity']:.0f})" for m in top_liquid])
            insights.append({
                'type': 'info',
                'title': '💧 Best Liquidity Markets',
                'description': f'Top liquid markets for smooth execution: {liquid_info}',
                'priority': 'LOW'
            })

        return jsonify({
            'success': True,
            'data': insights,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/predict', methods=['POST'])
def ai_predict():
    """Use pre-loaded singleton AMQSS AI model to predict market quality."""
    try:
        data = request.json or {}

        features = {
            'volatility_regime': float(data.get('volatility_regime', 5)),
            'true_liquidity': float(data.get('liquidity', 50)) / 10.0,
            'structure': float(data.get('structure', 50)) / 10.0,
            'momentum': float(data.get('momentum', 50)) / 10.0,
            'volatility_level': float(data.get('volatility', 20)) / 10.0,
            'current_atr': float(data.get('atr', 0.001)),
        }

        if predictor is not None:
            prob = predictor.predict(features)
            prediction = {
                'probability': float(prob),
                'confidence': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.5 else 'LOW',
                'recommendation': 'BUY' if prob > 0.6 else 'NEUTRAL' if prob > 0.4 else 'SELL',
                'timestamp': datetime.now().isoformat(),
                'model': 'AQMSS RandomForest (Singleton)'
            }
            return jsonify({'success': True, 'data': prediction})

        # Fallback mock prediction if model is not loaded
        mock_prob = 0.3 + (features['true_liquidity'] / 20.0) + (features['structure'] / 15.0)
        mock_prob = min(max(mock_prob, 0.05), 0.95)
        prediction = {
            'probability': float(mock_prob),
            'confidence': 'MEDIUM',
            'recommendation': 'BUY' if mock_prob > 0.6 else 'NEUTRAL' if mock_prob > 0.4 else 'SELL',
            'timestamp': datetime.now().isoformat(),
            'model': 'Fallback (Model not loaded)'
        }
        return jsonify({'success': True, 'data': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """AI Copilot 2.0 chat with live quant context and function tools."""
    try:
        data = request.json or {}
        user_message = str(data.get('message', '')).strip()
        session_id = str(data.get('session_id', 'default'))

        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400

        client = make_groq_client()
        live_markets = get_cached_markets()

        response = generate_copilot_response(
            user_message=user_message,
            session_id=session_id,
            live_markets=live_markets,
            groq_client=client,
            groq_model=GROQ_MODEL
        )

        return jsonify({
            'success': True,
            'data': {
                'user_message': user_message,
                'ai_response': response,
                'session_id': session_id,
                'powered_by': 'groq' if client else 'quant-copilot-2.0',
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



def _build_system_prompt(aqmss) -> str:
    """Build a system prompt that gives the LLM live AQMSS context."""
    base = (
        "You are an intelligent AI trading assistant embedded in the AQMSS dashboard.\n"
        "AQMSS (AI Market Quality Scoring System) analyses EURUSD and scores market quality 0-10.\n"
        "Score interpretation: <4 = NO TRADE | 4-6 = WEAK | 6-7.5 = NORMAL | >7.5 = HIGH QUALITY.\n"
        "Five factors (weights): Structure 30%, Volatility Regime 25%, Momentum 20%, Liquidity 15%, Volatility Level 10%.\n\n"
    )
    if aqmss:
        prob_str = f"{aqmss['ai_high_quality_prob']*100:.1f}%" if aqmss.get('ai_high_quality_prob') is not None else 'N/A'
        base += (
            f"=== LIVE AQMSS DATA (EURUSD) ===\n"
            f"Total Score    : {aqmss['total_score']:.2f}/10\n"
            f"Condition      : {aqmss['market_condition']}\n"
            f"Structure      : {aqmss['structure']:.2f}/10\n"
            f"Momentum       : {aqmss['momentum']:.2f}/10\n"
            f"Liquidity      : {aqmss['true_liquidity']:.2f}/10\n"
            f"Vol Regime     : {aqmss['volatility_regime']:.2f}/10\n"
            f"Vol Level      : {aqmss['volatility_level']:.2f}/10\n"
            f"AI HQ Prob     : {prob_str}\n"
            f"================================\n"
        )
    base += (
        "Always answer concisely and clearly. "
        "If the user supplies CSV-derived metrics for another asset, treat those metrics as the authoritative context for that asset and analyze it directly; do not say that only EURUSD is supported. "
        "If the user writes in Russian, answer in Russian. "
        "Do not use markdown headers or bullet asterisks — plain text only."
    )
    return base


def _groq_chat(client, session_id: str, user_message: str, aqmss) -> str:
    """Send message to Groq with conversation history, return assistant reply."""
    history = get_chat_history(session_id)
    history.append({'role': 'user', 'content': user_message})

    if len(history) > 20:
        history[:] = history[-20:]

    messages = [
        {'role': 'system', 'content': _build_system_prompt(aqmss)},
        *history
    ]

    print(f'[GROQ] Sending to {GROQ_MODEL}, history_len={len(history)}', flush=True)
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            timeout=15,
        )
        reply = completion.choices[0].message.content.strip()
        print(f'[GROQ] Reply received: {reply[:60]}...', flush=True)
        history.append({'role': 'assistant', 'content': reply})
        return reply
    except Exception as e:
        print(f'[GROQ] ERROR: {type(e).__name__}: {e}', flush=True)
        if history:
            history.pop()
        # Show real error so user knows what's wrong
        return f'[AI Error: {type(e).__name__}: {str(e)[:200]}]'



def generate_ai_response(user_message: str, aqmss=None) -> str:
    """Generate AI response with real AQMSS data context"""
    msg = user_message.lower()

    # Build AQMSS context string
    aqmss_ctx = ""
    if aqmss:
        aqmss_ctx = (
            f"\n\n📊 Current AQMSS Data (EURUSD):\n"
            f"• Total Score: {aqmss['total_score']:.2f}/10\n"
            f"• Condition: {aqmss['market_condition']}\n"
            f"• Structure: {aqmss['structure']:.2f}/10\n"
            f"• Momentum: {aqmss['momentum']:.2f}/10\n"
            f"• Liquidity: {aqmss['true_liquidity']:.2f}/10\n"
            f"• Vol Regime: {aqmss['volatility_regime']:.2f}/10\n"
            f"• Vol Level: {aqmss['volatility_level']:.2f}/10\n"
        )
        if aqmss.get('ai_high_quality_prob') is not None:
            aqmss_ctx += f"• AI High Quality Prob: {aqmss['ai_high_quality_prob']*100:.1f}%\n"

    # Response templates
    if 'quality' in msg or 'score' in msg or 'aqmss' in msg:
        return (
            f"📊 AQMSS Quality Scoring System\n\n"
            f"The AQMSS computes a total score (0-10) from 5 weighted factors:\n"
            f"• Volatility Regime (25%): ATR vs historical mean\n"
            f"• True Liquidity (15%): Volume-based metric\n"
            f"• Structure (30%): Trend strength via regression\n"
            f"• Momentum (20%): Candle range analysis\n"
            f"• Volatility Level (10%): ATR stability\n\n"
            f"Thresholds: <4 NO TRADE | 4-6 WEAK | 6-7.5 NORMAL | >7.5 HIGH QUALITY"
            f"{aqmss_ctx}"
        )

    if 'best' in msg or 'recommend' in msg or 'top' in msg:
        return (
            f"✨ Current Recommendations\n\n"
            f"Based on AQMSS analysis:\n"
            f"1. Focus on markets with Quality Scores > 75 (>7.5/10 AQMSS)\n"
            f"2. Look for BULLISH momentum + high liquidity\n"
            f"3. Avoid markets in NO TRADE or WEAK CONDITIONS\n"
            f"4. Use ATR for position sizing (current ATR from AQMSS)\n"
            f"5. Check AI probability before entry (>60% favorable)"
            f"{aqmss_ctx}"
        )

    if 'volatile' in msg or 'volatility' in msg:
        return (
            f"⚡ Volatility Analysis\n\n"
            f"AQMSS measures two volatility aspects:\n"
            f"• Volatility Regime: Current ATR vs historical mean\n"
            f"• Volatility Level: ATR standard deviation (stability)\n\n"
            f"High volatility = wider stops, smaller positions\n"
            f"Low volatility = tighter stops, larger positions"
            f"{aqmss_ctx}"
        )

    if 'liquid' in msg:
        return (
            f"💧 Liquidity Analysis\n\n"
            f"AQMSS True Liquidity measures volume-based market depth.\n\n"
            f"• Score > 7/10: Excellent execution expected\n"
            f"• Score 4-7/10: Normal conditions\n"
            f"• Score < 4/10: Thin market, watch for slippage"
            f"{aqmss_ctx}"
        )

    if 'bullish' in msg:
        return (
            f"📈 Bullish Market Analysis\n\n"
            f"AQMSS Momentum factor measures candle range vs ATR.\n\n"
            f"Bullish signals:\n"
            f"• Rising momentum score\n"
            f"• Strong structure (consistent higher highs/lows)\n"
            f"• High quality score in bullish markets\n\n"
            f"Best strategy: Follow momentum with AQMSS confirmation"
            f"{aqmss_ctx}"
        )

    if 'bearish' in msg:
        return (
            f"📉 Bearish Market Analysis\n\n"
            f"AQMSS can detect bearish conditions through:\n"
            f"• Weakening structure score\n"
            f"• Low momentum\n"
            f"• Overall weak quality score\n\n"
            f"Consider: Short positions or staying flat"
            f"{aqmss_ctx}"
        )

    if 'risk' in msg:
        return (
            f"🛡️ Risk Management with AQMSS\n\n"
            f"1. Never risk more than 2% per trade\n"
            f"2. Use AQMSS ATR for stop-loss placement\n"
            f"3. Only trade in NORMAL or HIGH QUALITY conditions\n"
            f"4. Scale position size with quality score\n"
            f"5. Check AI probability before entry"
            f"{aqmss_ctx}"
        )

    if 'predict' in msg or 'ai' in msg or 'model' in msg:
        metrics = load_model_metrics()
        meta = load_model_metadata()
        model_info = ""
        if metrics:
            model_info = (
                f"\n\n🤖 Your AI Model Stats:\n"
                f"• ROC AUC: {metrics.get('roc_auc', 'N/A')}\n"
                f"• Precision: {metrics.get('precision', 'N/A')}\n"
                f"• Recall: {metrics.get('recall', 'N/A')}\n"
                f"• F1 Score: {metrics.get('f1_score', 'N/A')}\n"
            )
        if meta:
            model_info += (
                f"• Model: {meta.get('model_type', 'N/A')}\n"
                f"• Train Size: {meta.get('train_size', 'N/A')}\n"
                f"• Features: {meta.get('n_features', 'N/A')}\n"
            )
        return (
            f"🔮 AQMSS AI Prediction System\n\n"
            f"Your trained RandomForest model classifies market quality.\n"
            f"It predicts probability of HIGH QUALITY conditions.\n\n"
            f"Use predictions as ONE input alongside the AQMSS score."
            f"{model_info}{aqmss_ctx}"
        )

    if 'help' in msg:
        return (
            f"🤖 AQMSS AI Assistant\n\n"
            f"I can help with:\n"
            f"• 'quality' - AQMSS scoring explained\n"
            f"• 'best' - Trading recommendations\n"
            f"• 'volatility' - Volatility analysis\n"
            f"• 'liquidity' - Liquidity insights\n"
            f"• 'risk' - Risk management\n"
            f"• 'predict' / 'model' - AI model info\n"
            f"• 'bullish' / 'bearish' - Direction analysis"
            f"{aqmss_ctx}"
        )

    if 'strategy' in msg:
        return (
            f"🎲 AQMSS Trading Strategy\n\n"
            f"1. Screen: Filter by AQMSS score > 6/10\n"
            f"2. Analyze: Check all 5 factors\n"
            f"3. Confirm: AI probability > 60%\n"
            f"4. Enter: Position size based on ATR\n"
            f"5. Manage: Stop at 1.5x ATR\n"
            f"6. Exit: Follow your target or trailing stop"
            f"{aqmss_ctx}"
        )

    # Default
    return (
        f"I specialize in AQMSS market analysis. Try asking about:\n\n"
        f"📊 Quality scores & AQMSS factors\n"
        f"💡 Trading recommendations\n"
        f"🤖 AI model predictions\n"
        f"📈 Market direction analysis\n"
        f"🛡️ Risk management\n\n"
        f"Type 'help' for all available topics."
        f"{aqmss_ctx}"
    )


# ==========================================
# REGIME SHIFT ALERTS & NOTIFICATION ROUTES
# ==========================================

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent regime shifts and market quality alerts."""
    try:
        recent = alert_manager.get_recent_alerts(limit=50)
        config = alert_manager.get_config()
        return jsonify({
            'success': True,
            'alerts': recent,
            'count': len(recent),
            'config': config,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/alerts/config', methods=['POST'])
def update_alerts_config():
    """Update Telegram and Webhook notification settings (Admin only)."""
    user = current_user()
    if not user or not user.get('is_admin'):
        return jsonify({'success': False, 'error': 'Admin privileges required'}), 403

    try:
        data = request.json or {}
        tg_token = str(data.get('telegram_bot_token', '')).strip()
        tg_chat = str(data.get('telegram_chat_id', '')).strip()
        wh_url = str(data.get('webhook_url', '')).strip()
        wh_secret = str(data.get('webhook_secret', '')).strip()

        if tg_token and tg_chat:
            alert_manager.configure_telegram(tg_token, tg_chat)
        if wh_url:
            alert_manager.configure_webhook(wh_url, wh_secret)

        with auth_db() as db:
            db.execute('''
                INSERT INTO alert_settings (id, telegram_bot_token, telegram_chat_id, webhook_url, webhook_secret)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    telegram_bot_token = excluded.telegram_bot_token,
                    telegram_chat_id = excluded.telegram_chat_id,
                    webhook_url = excluded.webhook_url,
                    webhook_secret = excluded.webhook_secret
            ''', (tg_token, tg_chat, wh_url, wh_secret))
            db.commit()

        return jsonify({
            'success': True,
            'message': 'Alert settings updated successfully',
            'config': alert_manager.get_config()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/alerts/test', methods=['POST'])
def test_alert():
    """Send a test alert through configured channels (Admin only)."""
    user = current_user()
    if not user or not user.get('is_admin'):
        return jsonify({'success': False, 'error': 'Admin privileges required'}), 403

    try:
        test_payload = {
            'id': str(uuid.uuid4()),
            'symbol': 'BTCUSD',
            'timestamp': datetime.now().isoformat(),
            'type': 'TEST_ALERT',
            'severity': 'INFO',
            'old_condition': 'TESTING',
            'new_condition': 'HIGH_QUALITY_MARKET',
            'old_score': 70.0,
            'new_score': 85.5,
            'message': '🔔 <b>[AQMSS TEST ALERT]</b> Тестовое уведомление из панели управления AQMSS Quant.'
        }
        alert_manager.dispatch_alert(test_payload)
        return jsonify({'success': True, 'message': 'Test alert dispatched to configured channels'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# AUTOMATED CRYPTO/STRIPE BILLING ROUTES
# ==========================================

@app.route('/api/billing/webhook', methods=['POST'])
def billing_webhook():
    """
    Automated webhook endpoint for payment gateways (NowPayments, Stripe, LemonSqueezy).
    Protected cryptographically by HMAC signature check.
    """
    secret = os.environ.get('AQMSS_BILLING_WEBHOOK_SECRET', 'test_billing_secret_key')
    raw_body = request.get_data()
    payload = request.json or {}
    headers = dict(request.headers)

    success, message, code = process_billing_webhook(
        headers=headers,
        raw_body=raw_body,
        payload=payload,
        db_path=str(AUTH_DB),
        secret=secret
    )
    return jsonify({'success': success, 'message': message}), code


@app.route('/api/billing/status', methods=['GET'])
def billing_status():
    """Get active subscription status for current user."""
    user = current_user()
    if not user:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    with auth_db() as db:
        row = db.execute(
            'SELECT id, email, is_active, subscription_expires_at FROM users WHERE id = ?',
            (user['id'],)
        ).fetchone()

        if not row:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        expires_at = row['subscription_expires_at']
        is_active = bool(row['is_active'])

        if expires_at:
            try:
                from datetime import timezone
                dt = datetime.fromisoformat(expires_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < datetime.now(timezone.utc):
                    is_active = False
            except Exception:
                pass

        return jsonify({
            'success': True,
            'data': {
                'email': row['email'],
                'is_active': is_active,
                'subscription_expires_at': expires_at,
                'subscription_price_usdt': SUBSCRIPTION_PRICE,
                'payment_address': PAYMENT_ADDRESS
            }
        })


if __name__ == '__main__':
    print("=" * 50)
    print("AQMSS Dashboard Server")
    print("=" * 50)

    # Check real data availability
    aqmss = get_latest_aqmss()
    if aqmss:
        print(f"[OK] AQMSS data found: Score {aqmss['total_score']:.2f}/10 ({aqmss['market_condition']})")
    else:
        print("[WARN] No AQMSS data found - run main.py first")

    metrics = load_model_metrics()
    if metrics:
        print(f"[OK] AI Model loaded: ROC AUC = {metrics.get('roc_auc', 'N/A')}")
    else:
        print("[WARN] No model metrics found - run run_pipeline.py first")

    training = load_training_data()
    if training is not None:
        print(f"[OK] Training data: {len(training)} rows")
    else:
        print("[WARN] No training data found")

    print(f"[INFO] AI Module: {'Available' if AI_AVAILABLE else 'Not available'}")
    print(f"\n[INFO] Starting server at http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)
