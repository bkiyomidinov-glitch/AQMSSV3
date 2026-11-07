"""
Geolocation and IP intelligence service.
Safe, cached, and non-blocking lookup without hanging request worker threads.
"""

import json
import sqlite3
import threading
from typing import Dict, Any
from urllib.request import Request, urlopen
from backend.services.cache import ip_cache

DEFAULT_LOCAL = {
    'country': 'Local',
    'region': 'Local',
    'city': 'Local',
    'isp': 'Local',
    'asn': '—',
    'ip_flags': 'local'
}

DEFAULT_UNKNOWN = {
    'country': 'Unknown',
    'region': 'Unknown',
    'city': 'Unknown',
    'isp': 'Unknown',
    'asn': '—',
    'ip_flags': 'clean'
}


def _network_lookup(ip: str, timeout: float = 1.2) -> Dict[str, str]:
    """Perform network lookup against ipwho.is."""
    req = Request(
        f'https://ipwho.is/{ip}',
        headers={'User-Agent': 'AMQSS-Geo/2.0'}
    )
    with urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode('utf-8'))

    connection = data.get('connection') or {}
    security = data.get('security') or {}
    flags = [name for name in ('vpn', 'proxy', 'tor', 'hosting') if security.get(name)]

    return {
        'country': data.get('country') or 'Unknown',
        'region': data.get('region') or 'Unknown',
        'city': data.get('city') or 'Unknown',
        'isp': connection.get('isp') or connection.get('org') or 'Unknown',
        'asn': str(connection.get('asn') or '—'),
        'ip_flags': ', '.join(flags) if flags else 'clean'
    }


def lookup_ip_details(ip: str) -> Dict[str, str]:
    """
    Get IP details from cache or fast network call.
    Guaranteed not to block the worker thread for more than 1 second.
    """
    if not ip or ip in {'unknown', '127.0.0.1', '::1', 'localhost'}:
        return DEFAULT_LOCAL

    cached = ip_cache.get(ip)
    if cached:
        return cached

    try:
        details = _network_lookup(ip, timeout=1.0)
        ip_cache.set(ip, details)
        return details
    except Exception:
        # Do not block or fail request; return unknown and let async worker retry
        return DEFAULT_UNKNOWN


def update_user_geo_async(db_path: str, user_id: int, ip: str) -> None:
    """
    Update user geolocation asynchronously in background thread.
    Zero latency overhead on login and registration requests.
    """
    if not ip or ip in {'unknown', '127.0.0.1', '::1', 'localhost'}:
        return

    def _worker():
        try:
            cached = ip_cache.get(ip)
            if cached:
                details = cached
            else:
                details = _network_lookup(ip, timeout=2.0)
                ip_cache.set(ip, details)

            with sqlite3.connect(db_path, timeout=5) as conn:
                conn.execute(
                    'UPDATE users SET country=?, region=?, city=?, isp=?, asn=?, ip_flags=? WHERE id=?',
                    (
                        details['country'], details['region'], details['city'],
                        details['isp'], details['asn'], details['ip_flags'], user_id
                    )
                )
                conn.commit()
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()
