"""
Automated Cryptographic Billing & Subscription Engine.
Supports HMAC-SHA256/SHA512 verification for Webhooks (NowPayments, Stripe, LemonSqueezy)
and executes atomic 30-day subscription renewals in SQLite.
"""

import hmac
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def verify_webhook_signature(
    raw_body: bytes,
    signature_header: Optional[str],
    secret: str,
    algorithm: str = 'sha256'
) -> bool:
    """
    Cryptographically verify HMAC signature from payment gateway webhook.
    Guarantees timing-attack safety via constant-time hmac.compare_digest.
    """
    if not signature_header or not secret:
        return False

    secret_bytes = secret.encode('utf-8')
    sig_str = signature_header.strip()

    # Extract signature if format is "t=...,v1=..." (Stripe standard)
    if 'v1=' in sig_str:
        parts = dict(item.split('=', 1) for item in sig_str.split(',') if '=' in item)
        timestamp = parts.get('t', '')
        v1_sig = parts.get('v1', '')
        # Stripe signs: timestamp + "." + raw_body
        signed_payload = f"{timestamp}.".encode('utf-8') + raw_body
        expected = hmac.new(secret_bytes, signed_payload, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected.lower(), v1_sig.lower()):
            return True
        # Also check direct payload verification in case timestamp is omitted
        direct_expected = hmac.new(secret_bytes, raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(direct_expected.lower(), v1_sig.lower())

    # Strip prefixes like "sha256=" or "sha512="
    if sig_str.startswith('sha256='):
        sig_str = sig_str.split('sha256=', 1)[1]
    elif sig_str.startswith('sha512='):
        sig_str = sig_str.split('sha512=', 1)[1]
        algorithm = 'sha512'

    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    expected_hex = hmac.new(secret_bytes, raw_body, hash_func).hexdigest()

    return hmac.compare_digest(expected_hex.lower(), sig_str.lower())


def extract_payment_info(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract normalized email and payment confirmation status across major gateways:
    NowPayments, Stripe, LemonSqueezy, and generic JSON webhooks.
    """
    email = None
    is_confirmed = False
    amount = 0.0
    currency = "USD"

    # 1. NowPayments format
    if 'payment_status' in payload:
        status = str(payload.get('payment_status', '')).lower()
        if status in {'finished', 'confirmed', 'sending'}:
            is_confirmed = True
        email = payload.get('payin_extra_id') or payload.get('order_id') or payload.get('email')
        amount = float(payload.get('pay_amount') or payload.get('price_amount') or 0.0)
        currency = payload.get('pay_currency') or payload.get('price_currency') or 'USDT'

    # 2. Stripe Checkout / Charge format
    elif payload.get('object') == 'event' or 'type' in payload:
        evt_type = str(payload.get('type', '')).lower()
        data_obj = payload.get('data', {}).get('object', {})
        if evt_type in {'checkout.session.completed', 'invoice.payment_succeeded', 'charge.succeeded'}:
            is_confirmed = True
        email = (
            data_obj.get('customer_email')
            or data_obj.get('customer_details', {}).get('email')
            or data_obj.get('billing_details', {}).get('email')
        )
        amount = float(data_obj.get('amount_total', 0)) / 100.0
        currency = data_obj.get('currency', 'usd').upper()

    # 3. LemonSqueezy format
    elif 'meta' in payload and 'event_name' in payload.get('meta', {}):
        event_name = str(payload['meta']['event_name']).lower()
        if event_name in {'order_created', 'subscription_created', 'subscription_payment_success'}:
            is_confirmed = True
        attrs = payload.get('data', {}).get('attributes', {})
        email = attrs.get('user_email')
        amount = float(attrs.get('total_formatted', '0').replace('$', '').strip() or 0.0)

    # 4. Generic Webhook format
    else:
        status = str(payload.get('status', payload.get('state', ''))).lower()
        if status in {'paid', 'completed', 'finished', 'confirmed', 'success'}:
            is_confirmed = True
        email = payload.get('email') or payload.get('user_email') or payload.get('customer_email')
        amount = float(payload.get('amount', 0.0))
        currency = str(payload.get('currency', 'USDT')).upper()

    return {
        'email': email.strip().lower() if email else None,
        'is_confirmed': is_confirmed,
        'amount': amount,
        'currency': currency
    }


def activate_user_subscription(db_path: str, email: str, days: int = 30) -> Dict[str, Any]:
    """
    Atomically activate or extend user subscription in SQLite database by specified days.
    """
    if not email:
        return {'success': False, 'error': 'No email provided'}

    clean_email = email.strip().lower()
    now_utc = datetime.now(timezone.utc)

    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Find user
        cur.execute('SELECT id, email, is_active, subscription_expires_at FROM users WHERE lower(email) = ?', (clean_email,))
        user = cur.fetchone()

        if not user:
            return {'success': False, 'error': f"User with email '{clean_email}' not found in database"}

        # Calculate new expiry date
        current_expiry = user['subscription_expires_at']
        base_time = now_utc

        if current_expiry:
            try:
                # Handle ISO formatted string
                parsed_expiry = datetime.fromisoformat(current_expiry)
                if parsed_expiry.tzinfo is None:
                    parsed_expiry = parsed_expiry.replace(tzinfo=timezone.utc)
                # If currently active in future, extend from that point
                if parsed_expiry > now_utc:
                    base_time = parsed_expiry
            except Exception:
                base_time = now_utc

        new_expiry = base_time + timedelta(days=days)
        new_expiry_iso = new_expiry.isoformat()

        # Update user record
        cur.execute(
            'UPDATE users SET is_active = 1, subscription_expires_at = ? WHERE id = ?',
            (new_expiry_iso, user['id'])
        )
        conn.commit()

        logger.info(f"[Billing] Subscription activated for {clean_email}: expires at {new_expiry_iso}")
        return {
            'success': True,
            'user_id': user['id'],
            'email': clean_email,
            'is_active': 1,
            'subscription_expires_at': new_expiry_iso,
            'days_added': days
        }


def process_billing_webhook(
    headers: Dict[str, str],
    raw_body: bytes,
    payload: Dict[str, Any],
    db_path: str,
    secret: str
) -> Tuple[bool, str, int]:
    """
    Full pipeline to handle external payment webhooks:
    Signature verification -> Payload extraction -> DB subscription update.
    Returns: (success: bool, message: str, http_status_code: int)
    """
    # 1. Header lookup (case-insensitive)
    lower_headers = {k.lower(): v for k, v in headers.items()}
    sig_header = (
        lower_headers.get('x-nowpayments-sig')
        or lower_headers.get('stripe-signature')
        or lower_headers.get('x-signature')
        or lower_headers.get('x-webhook-signature')
    )

    # 2. Cryptographic signature verification
    if not verify_webhook_signature(raw_body, sig_header, secret):
        logger.warning("[Billing] Webhook signature verification failed")
        return False, "Invalid HMAC webhook signature", 401

    # 3. Extract payment info
    info = extract_payment_info(payload)
    if not info['is_confirmed']:
        return True, f"Webhook received, status is not confirmed ({payload.get('payment_status', 'pending')})", 200

    target_email = info['email']
    if not target_email:
        return False, "Webhook payload missing user email reference", 400

    # 4. Activate subscription
    res = activate_user_subscription(db_path, target_email, days=30)
    if not res.get('success'):
        return False, res.get('error', 'Activation failed'), 404

    return True, f"Subscription successfully activated for {target_email}", 200
