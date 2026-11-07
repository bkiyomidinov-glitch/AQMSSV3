"""
Regime Shift Alerts & Notification Engine for AMQSS.
Detects market regime transitions, score surges, and momentum flips,
and dispatches real-time alerts via Telegram Bot API and universal Webhooks.
"""

import json
import logging
import threading
import time
import uuid
import hmac
import hashlib
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class RegimeAlertManager:
    """
    Stateful manager for market regime shift detection and notifications.
    Maintains historical state per asset and dispatches instant alerts.
    """

    def __init__(self, max_history: int = 200):
        self._lock = threading.Lock()
        self._states: Dict[str, Dict[str, Any]] = {}
        self._alerts: deque = deque(maxlen=max_history)
        self._telegram_config: Dict[str, str] = {}
        self._webhook_config: Dict[str, str] = {}

    def configure_telegram(self, bot_token: str, chat_id: str) -> None:
        """Configure Telegram bot credentials."""
        with self._lock:
            self._telegram_config = {
                'bot_token': bot_token.strip(),
                'chat_id': chat_id.strip()
            }

    def configure_webhook(self, webhook_url: str, secret: Optional[str] = None) -> None:
        """Configure Webhook destination and optional HMAC secret."""
        with self._lock:
            self._webhook_config = {
                'url': webhook_url.strip(),
                'secret': secret.strip() if secret else ''
            }

    def get_config(self) -> Dict[str, Any]:
        """Return non-sensitive view of alert configuration."""
        with self._lock:
            has_tg = bool(self._telegram_config.get('bot_token') and self._telegram_config.get('chat_id'))
            masked_tg = ""
            if has_tg:
                token = self._telegram_config['bot_token']
                masked_tg = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "***"

            return {
                'telegram_configured': has_tg,
                'telegram_token_masked': masked_tg,
                'telegram_chat_id': self._telegram_config.get('chat_id', ''),
                'webhook_configured': bool(self._webhook_config.get('url')),
                'webhook_url': self._webhook_config.get('url', ''),
            }

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent alert history."""
        with self._lock:
            return list(self._alerts)[:limit]

    def clear_alerts(self) -> None:
        """Clear recorded alert history."""
        with self._lock:
            self._alerts.clear()

    def detect_shifts(self, current_markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compare current market states against previous states.
        Emits alerts upon regime shifts, crossing score thresholds, or momentum flips.
        """
        detected: List[Dict[str, Any]] = []

        with self._lock:
            for m in current_markets:
                sym = m.get('symbol')
                if not sym:
                    continue

                curr_score = float(m.get('quality_score', 0))
                curr_cond = str(m.get('market_condition', 'UNKNOWN'))
                curr_mom = str(m.get('momentum', 'NEUTRAL'))
                curr_price = m.get('price', 0)

                prev = self._states.get(sym)

                if prev is not None:
                    old_score = prev['score']
                    old_cond = prev['condition']
                    old_mom = prev['momentum']

                    alert: Optional[Dict[str, Any]] = None

                    # 1. Condition Shift (e.g. into HIGH_QUALITY_MARKET or into NO_TRADE)
                    if curr_cond != old_cond:
                        severity = 'HIGH' if 'HIGH_QUALITY' in curr_cond or 'NO_TRADE' in curr_cond else 'MEDIUM'
                        emoji = '🚀' if 'HIGH_QUALITY' in curr_cond else '⚠️' if 'NO_TRADE' in curr_cond else '🔄'
                        msg = (
                            f"{emoji} <b>[REGIME SHIFT] {sym}</b>\n"
                            f"Режим изменен: <code>{old_cond}</code> ➔ <code>{curr_cond}</code>\n"
                            f"Score: {old_score:.1f} ➔ {curr_score:.1f} | Цена: {curr_price}"
                        )
                        alert = {
                            'id': str(uuid.uuid4()),
                            'symbol': sym,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'type': 'REGIME_SHIFT',
                            'severity': severity,
                            'old_condition': old_cond,
                            'new_condition': curr_cond,
                            'old_score': old_score,
                            'new_score': curr_score,
                            'message': msg
                        }

                    # 2. Significant Score Surge or Plunge (>= 15 pts or crossing 75.0)
                    elif (abs(curr_score - old_score) >= 15.0) or (old_score < 75.0 <= curr_score) or (old_score >= 75.0 > curr_score):
                        is_surge = curr_score > old_score
                        emoji = '📈' if is_surge else '📉'
                        severity = 'HIGH' if curr_score >= 75.0 else 'MEDIUM'
                        action = 'Резкий рост качества' if is_surge else 'Резкое падение качества'
                        msg = (
                            f"{emoji} <b>[SCORE DIVERGENCE] {sym}</b>\n"
                            f"{action}: {old_score:.1f} ➔ {curr_score:.1f}\n"
                            f"Состояние: {curr_cond} | Моментум: {curr_mom}"
                        )
                        alert = {
                            'id': str(uuid.uuid4()),
                            'symbol': sym,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'type': 'SCORE_SURGE',
                            'severity': severity,
                            'old_condition': old_cond,
                            'new_condition': curr_cond,
                            'old_score': old_score,
                            'new_score': curr_score,
                            'message': msg
                        }

                    # 3. Momentum Flip in High Quality Environment
                    elif curr_mom != old_mom and curr_score >= 60.0:
                        emoji = '🐂' if curr_mom == 'BULLISH' else '🐻' if curr_mom == 'BEARISH' else '⚖️'
                        msg = (
                            f"{emoji} <b>[MOMENTUM FLIP] {sym}</b>\n"
                            f"Моментум сменился: {old_mom} ➔ {curr_mom}\n"
                            f"AQMSS Score: {curr_score:.1f}/100 ({curr_cond})"
                        )
                        alert = {
                            'id': str(uuid.uuid4()),
                            'symbol': sym,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'type': 'MOMENTUM_FLIP',
                            'severity': 'MEDIUM',
                            'old_condition': old_cond,
                            'new_condition': curr_cond,
                            'old_score': old_score,
                            'new_score': curr_score,
                            'message': msg
                        }

                    if alert:
                        detected.append(alert)
                        self._alerts.appendleft(alert)

                # Update recorded state
                self._states[sym] = {
                    'score': curr_score,
                    'condition': curr_cond,
                    'momentum': curr_mom,
                    'price': curr_price,
                    'updated_at': time.time()
                }

            # Return list of newly triggered alerts
            return detected

    def dispatch_alert(self, alert: Dict[str, Any]) -> None:
        """Dispatch alert to active channels in a detached background thread."""
        tg_conf = dict(self._telegram_config)
        wb_conf = dict(self._webhook_config)

        def _worker():
            # Send Telegram
            if tg_conf.get('bot_token') and tg_conf.get('chat_id'):
                try:
                    send_telegram_notification(
                        bot_token=tg_conf['bot_token'],
                        chat_id=tg_conf['chat_id'],
                        message=alert.get('message', '')
                    )
                except Exception as e:
                    logger.warning(f"[AlertManager] Telegram dispatch error: {e}")

            # Send Webhook
            if wb_conf.get('url'):
                try:
                    send_webhook_notification(
                        webhook_url=wb_conf['url'],
                        alert=alert,
                        secret=wb_conf.get('secret')
                    )
                except Exception as e:
                    logger.warning(f"[AlertManager] Webhook dispatch error: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def process_market_updates(self, current_markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Full pipeline: detect shifts and dispatch notifications."""
        new_alerts = self.detect_shifts(current_markets)
        for alert in new_alerts:
            self.dispatch_alert(alert)
        return new_alerts


def send_telegram_notification(bot_token: str, chat_id: str, message: str, timeout: int = 5) -> bool:
    """Send HTML message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True
    }).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning(f"Telegram API request failed: {e}")
        return False


def send_webhook_notification(
    webhook_url: str,
    alert: Dict[str, Any],
    secret: Optional[str] = None,
    timeout: int = 5
) -> bool:
    """Send JSON alert payload to external webhook with optional HMAC-SHA256 signature."""
    payload = json.dumps(alert).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'AQMSS-Alert-Dispatcher/2.0'
    }

    if secret:
        sig = hmac.new(secret.encode('utf-8'), payload, hashlib.sha256).hexdigest()
        headers['X-AQMSS-Signature'] = f"sha256={sig}"

    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers=headers,
        method='POST'
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        logger.warning(f"Webhook request failed to {webhook_url}: {e}")
        return False


# Singleton Alert Manager instance
alert_manager = RegimeAlertManager()
