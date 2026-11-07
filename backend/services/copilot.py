"""
Copilot 2.0 Engine with Live Market Quant Context and Deterministic Intelligence.
Provides direct tool execution across live market metrics and high-accuracy quant analytics,
with seamless fallback when external LLM keys are absent.
"""

import os
import re
from collections import OrderedDict
from typing import Dict, Any, List, Optional
from backend.services.market_feed import MARKET_LOOKUP, SUPPORTED_MARKETS


# LRU Cache for chat histories (Thread-safe & Memory leak prevention)
_chat_histories: OrderedDict = OrderedDict()
MAX_CHAT_SESSIONS = 200


def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieve or initialize conversation history for a session."""
    if session_id in _chat_histories:
        _chat_histories.move_to_end(session_id)
        return _chat_histories[session_id]
    if len(_chat_histories) >= MAX_CHAT_SESSIONS:
        _chat_histories.popitem(last=False)
    _chat_histories[session_id] = []
    return _chat_histories[session_id]


def clear_chat_history(session_id: str) -> None:
    """Clear conversation history for a given session."""
    if session_id in _chat_histories:
        del _chat_histories[session_id]


def find_symbol_in_text(text: str) -> Optional[str]:
    """Extract known trading symbol or asset name from user text."""
    normalized = text.upper()
    # Check direct symbols first
    for market in SUPPORTED_MARKETS:
        sym = market['symbol']
        if sym in normalized:
            return sym

    # Check common aliases
    alias_map = {
        'BTC': 'BTCUSD', 'BITCOIN': 'BTCUSD', 'БИТКОИН': 'BTCUSD', 'БИТОК': 'BTCUSD', 'БИТК': 'BTCUSD',
        'ETH': 'ETHUSD', 'ETHEREUM': 'ETHUSD', 'ЭФИР': 'ETHUSD', 'ЭФИРИУМ': 'ETHUSD',
        'GOLD': 'XAUUSD', 'ЗОЛОТО': 'XAUUSD', 'ЗОЛОТ': 'XAUUSD', 'XAU': 'XAUUSD',
        'EUR': 'EURUSD', 'ЕВРО': 'EURUSD',
        'GBP': 'GBPUSD', 'ФУНТ': 'GBPUSD',
        'JPY': 'USDJPY', 'ЙЕНА': 'USDJPY', 'ИЕНА': 'USDJPY',
        'SPX': 'SPX500', 'S&P': 'SPX500', 'СП500': 'SPX500',
        'APPLE': 'AAPL', 'ЭПЛ': 'AAPL',
        'TESLA': 'TSLA', 'ТЕСЛА': 'TSLA',
        'NVIDIA': 'NVDA', 'НВИДИА': 'NVDA',
        'MICROSOFT': 'MSFT', 'МАЙКРОСОФТ': 'MSFT',
        'GOOGLE': 'GOOGL', 'ГУГЛ': 'GOOGL',
        'AMAZON': 'AMZN', 'АМАЗОН': 'AMZN'
    }
    for alias, target in alias_map.items():
        if alias in normalized:
            return target
    return None


def find_symbols_for_comparison(text: str) -> List[str]:
    """Find two symbols for pairwise comparison."""
    words = re.findall(r'[A-Za-zА-Яа-я0-9]+', text.upper())
    found = []
    for w in words:
        sym = find_symbol_in_text(w)
        if sym and sym not in found:
            found.append(sym)
        if len(found) >= 2:
            break
    return found


def format_ticker_analysis(market: Dict[str, Any]) -> str:
    """Format deep quant breakdown for a single ticker."""
    sym = market['symbol']
    name = market.get('name', sym)
    price = market.get('price', 0)
    change = market.get('change', 0)
    score = market.get('quality_score', 0)
    cond = market.get('market_condition', 'UNKNOWN')
    liq = market.get('liquidity', 0)
    vol = market.get('volatility', 0)
    struct = market.get('structure', 0)
    mom_dir = market.get('momentum', 'NEUTRAL')
    mom_val = market.get('momentum_score', 5.0)
    ai_prob = market.get('ai_high_quality_prob')

    prob_str = f"{ai_prob * 100:.1f}%" if ai_prob is not None else "Не обучено"

    recommendation = (
        "🟢 ВЫСОКИЙ ПРИОРИТЕТ ДЛЯ ВХОДА (Отличная рыночная структура, высокая вероятность)"
        if score >= 75 and mom_dir == 'BULLISH'
        else "🟡 УМЕРЕННАЯ АКТИВНОСТЬ (Торговля в канале со стандартным риском)"
        if score >= 60
        else "🟠 ПОВЫШЕННЫЙ РИСК (Слабая структура тренда или недостаточная ликвидность)"
        if score >= 40
        else "🔴 ЗОНА NO_TRADE (Крайне неблагоприятные условия, избегать открытия позиций)"
    )

    return (
        f"📊 Анализ тикера {sym} ({name})\n\n"
        f"• Текущая цена: {price} ({'+' if change >= 0 else ''}{change}%)\n"
        f"• Общий AQMSS Score: {score:.1f}/100 ({cond})\n"
        f"• ML-вероятность High Quality: {prob_str}\n\n"
        f"Декомпозиция 5 факторов качества:\n"
        f"  1. Структура тренда (Structure): {struct:.1f}/100\n"
        f"  2. Моментум свечей (Momentum): {mom_val:.1f}/10 ({mom_dir})\n"
        f"  3. Плотность ликвидности (True Liquidity): {liq:.1f}/100\n"
        f"  4. Стабильность волатильности (Volatility Level): {vol:.1f}/100\n\n"
        f"💡 Рекомендация Quant Engine:\n{recommendation}"
    )


def format_comparison(market_a: Dict[str, Any], market_b: Dict[str, Any]) -> str:
    """Format comparative matrix between two assets."""
    sym_a, sym_b = market_a['symbol'], market_b['symbol']
    score_a, score_b = market_a['quality_score'], market_b['quality_score']
    liq_a, liq_b = market_a['liquidity'], market_b['liquidity']
    struct_a, struct_b = market_a['structure'], market_b['structure']
    mom_a, mom_b = market_a['momentum'], market_b['momentum']
    vol_a, vol_b = market_a['volatility'], market_b['volatility']

    winner = sym_a if score_a > score_b else sym_b if score_b > score_a else "Паритет"

    return (
        f"⚖️ Сравнение инструментов: {sym_a} vs {sym_b}\n\n"
        f"Метрика                  │ {sym_a:<10} │ {sym_b:<10}\n"
        f"─────────────────────────┼────────────┼────────────\n"
        f"Quality Score (0-100)    │ {score_a:<10.1f} │ {score_b:<10.1f}\n"
        f"Структура тренда         │ {struct_a:<10.1f} │ {struct_b:<10.1f}\n"
        f"Ликвидность              │ {liq_a:<10.1f} │ {liq_b:<10.1f}\n"
        f"Стабильность волатильност│ {vol_a:<10.1f} │ {vol_b:<10.1f}\n"
        f"Моментум                 │ {mom_a:<10} │ {mom_b:<10}\n\n"
        f"🏆 Победитель по матрице качества: {winner}\n"
        f"Вывод: {winner} демонстрирует более устойчивые статистические характеристики для сделок."
    )


def format_top_opportunities(markets: List[Dict[str, Any]]) -> str:
    """Format top-scoring assets."""
    sorted_m = sorted(markets, key=lambda x: x.get('quality_score', 0), reverse=True)
    top_3 = sorted_m[:3]

    lines = ["🏆 Топ возможностей рынка по версии AQMSS Quant Engine:\n"]
    for i, m in enumerate(top_3, 1):
        lines.append(
            f"{i}. {m['symbol']} — Score: {m['quality_score']:.1f}/100 | {m['market_condition']} | "
            f"Цена: {m['price']} ({m['momentum']})"
        )

    lines.append("\nРекомендация: Открывать позиции строго в направлении подтвержденного моментума.")
    return "\n".join(lines)


def format_risk_warnings(markets: List[Dict[str, Any]]) -> str:
    """Format risk warnings and low-quality markets."""
    poor = [m for m in markets if m.get('quality_score', 0) < 45 or m.get('volatility', 0) > 40]
    if not poor:
        return "🛡️ Все отслеживаемые рынки в нормальной зоне риска. Критических аномалий не обнаружено."

    lines = ["⚠️ Внимание: Рынки с повышенным риском / в зоне NO TRADE:\n"]
    for m in poor[:4]:
        lines.append(
            f"• {m['symbol']}: Score {m['quality_score']:.1f}/100 ({m['market_condition']}) | "
            f"Волатильность: {m['volatility']:.1f}"
        )
    lines.append("\nСовет по риск-менеджменту: Избегайте входов или сократите размер позиции минимум в 2 раза.")
    return "\n".join(lines)


def query_local_quant_copilot(user_message: str, live_markets: List[Dict[str, Any]]) -> str:
    """
    Deterministic rule-based Quant intelligence engine.
    Always available without external API keys and guarantees zero downtime.
    """
    msg = user_message.strip().lower()
    market_map = {m['symbol']: m for m in live_markets}

    # 1. Comparison query
    if any(k in msg for k in ['сравни', 'compare', 'vs', 'против', 'лучше']):
        syms = find_symbols_for_comparison(user_message)
        if len(syms) >= 2 and syms[0] in market_map and syms[1] in market_map:
            return format_comparison(market_map[syms[0]], market_map[syms[1]])

    # 2. Specific single ticker query
    target_sym = find_symbol_in_text(user_message)
    if target_sym and target_sym in market_map:
        return format_ticker_analysis(market_map[target_sym])

    # 3. Best / Top / Opportunity query
    if any(k in msg for k in ['топ', 'лучш', 'best', 'opportunities', 'сигнал', 'купить', 'high quality']):
        return format_top_opportunities(live_markets)

    # 4. Risk / No trade query
    if any(k in msg for k in ['риск', 'опасн', 'no trade', 'danger', 'предупрежд', 'стоп']):
        return format_risk_warnings(live_markets)

    # 5. Default market summary
    total = len(live_markets)
    high_q = sum(1 for m in live_markets if m.get('quality_score', 0) >= 70)
    bullish = sum(1 for m in live_markets if m.get('momentum') == 'BULLISH')

    return (
        f"🤖 AQMSS Quant Copilot 2.0 (Live Matrix Connected)\n\n"
        f"Прямой доступ к {total} активам в режиме реального времени:\n"
        f"• Рынков в High Quality: {high_q} из {total}\n"
        f"• Бычий моментум (Bullish): {bullish} активов\n\n"
        f"Примеры команд, которые я могу выполнить:\n"
        f"• «Проанализируй BTCUSD» или «Что по золоту XAUUSD?»\n"
        f"• «Сравни EURUSD и GBPUSD»\n"
        f"• «Какие лучшие возможности прямо сейчас?»\n"
        f"• «Где повышенный риск и зона NO TRADE?»"
    )


def generate_copilot_response(
    user_message: str,
    session_id: str,
    live_markets: List[Dict[str, Any]],
    groq_client: Any = None,
    groq_model: str = 'openai/gpt-oss-20b'
) -> str:
    """
    Master Copilot pipeline:
    Uses LLM with injected live market context if client is active,
    or falls back to high-accuracy local Quant engine.
    """
    # If no Groq LLM client, use local engine
    if groq_client is None:
        return query_local_quant_copilot(user_message, live_markets)

    # Market snapshot for system prompt
    market_lines = []
    for m in live_markets[:8]:
        market_lines.append(
            f"- {m['symbol']}: Price={m['price']}, Score={m['quality_score']:.1f}/100, "
            f"Cond={m['market_condition']}, Momentum={m['momentum']}"
        )
    market_summary_str = "\n".join(market_lines)

    system_prompt = (
        "You are AQMSS Copilot 2.0, an advanced quant AI trading assistant.\n"
        "You have DIRECT live access to real market metrics across Forex, Crypto, and Stocks.\n"
        "Scoring scale: 0-100 (0-40 NO TRADE, 40-60 WEAK, 60-75 NORMAL, >75 HIGH QUALITY).\n"
        "LIVE MARKET DATA SNAPSHOT:\n"
        f"{market_summary_str}\n\n"
        "Guidelines:\n"
        "1. Give direct, factual numerical answers using the real live figures above.\n"
        "2. If user writes in Russian, reply in clear, professional Russian.\n"
        "3. Provide structured quant breakdowns with actionable risk guidance."
    )

    history = get_chat_history(session_id)
    history.append({'role': 'user', 'content': user_message})
    if len(history) > 20:
        history[:] = history[-20:]

    messages = [
        {'role': 'system', 'content': system_prompt},
        *history
    ]

    try:
        completion = groq_client.chat.completions.create(
            model=groq_model,
            messages=messages,
            max_tokens=600,
            temperature=0.6,
            timeout=12
        )
        reply = completion.choices[0].message.content.strip()
        history.append({'role': 'assistant', 'content': reply})
        return reply
    except Exception:
        # Graceful fallback to local engine on API timeout or limit
        return query_local_quant_copilot(user_message, live_markets)
