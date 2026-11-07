# AMQSS Master Plan

Источник стандартов: `AGENTS.md` и активированные файлы в `.agents/rules/`.
Дата аудита: 2026-11-07.

Цель: превратить текущий прототип (скрипты + Flask-монолит + синтетический ML) в production SaaS рыночного quality-scoring без фейковых данных и с безопасным auth.

---

## P0 — Критические исправления (стандарты AGENTS.md / rules)

С этого уровня нельзя пропускать релизы.

### P0.1 Auth и секреты
- Заменить unsalted SHA-256 на bcrypt/Argon2 (`python.mdc`, `security-devsecops-ssdls-appsec.mdc`).
- Обязательный `AQMSS_SECRET_KEY`; убрать `secrets.token_hex(32)` fallback.
- Flask-Login (или эквивалент), CSRF на login/register/admin/billing.
- Закрыть open redirect: `next` только на внутренние path.
- Admin: не передавать ключ в query string; session + RBAC, не `?key=`.
- Не доверять `X-Forwarded-For` без trusted proxy config.
- Rate limit: login, register, `/api/ai/chat`, `/api/ai/predict`.
- Не писать traceback Groq в ответ пользователю.

### P0.2 Перестать выдавать симуляцию за продукт
- Убрать `generate_markets_from_aqmss()` как «рынки»: 14 из 15 пар — random noise вокруг EURUSD.
- Убрать synthetic candles fallback как «live» (`web/static/app.js`).
- News: не генерировать «новости» из quality_score; либо реальный feed, либо честный empty state.
- `/api/ai/predict` не должен возвращать mock BUY/SELL при отсутствии модели.
- `is_active` должен реально гейтить terminal/copilot/API, не только `/test`.

### P0.3 Production Flask runtime
- `app.run(debug=True, host='0.0.0.0')` запрещён как default.
- CORS allowlist вместо `CORS(app)`.
- Structured logging вместо `print`.
- Healthcheck endpoint без auth (`/healthz`), не корень с `requests`.

### P0.4 ML integrity
- Удалить/изолировать `ai_module.train_and_predict`: retraining на каждый request + label из `market_condition` = circular leakage.
- Evaluator split должен совпадать с trainer (сейчас trainer — per-class chronological, evaluator — простой 80/20 и fallback на full dataset → ROC-AUC 1.0).
- Не коммитить/не показывать в UI метрики 1.0 с синтетики как «модель загружена».
- `pickle.load` → joblib + schema/version + hash; не грузить произвольный blob.

### P0.5 Конфиг и дубли
- Вычистить hardcoded `C:\Users\bkiyo\...` из `dashboard.py`.
- Убрать nested копию `Project.X/Project.X/`.
- Один `requirements.txt` с pinned versions (сейчас корневой без pin pandas/numpy).

---

## P1 — Инженерный фундамент

### P1.1 Структура пакета
Целевой layout:

```
src/amqss/
  scoring/          # main.py engine → library
  ml/               # dataset, trainer, evaluator, predictor
  web/              # Flask factory, blueprints, services
  config.py
tests/
```

- Application factory + blueprints: `auth`, `markets`, `ai`, `admin`, `billing`.
- SQLAlchemy + Alembic вместо ad-hoc `CREATE TABLE` / `ALTER TABLE` в request path.
- Общий OHLCV parser (сейчас дублируется в `main.py`, `dataset_builder.py`, `dashboard.py`).

### P1.2 Scoring engine
- Вынести `main.py` из script-with-side-effects в чистые функции: `load_ohlcv` → `compute_factors` → `score` → `persist`.
- Named constants вместо magic numbers (окна, пороги, веса).
- Векторизовать `compute_forward_performance` (сейчас Python loop по каждой свече).
- Убрать legacy path `~/Downloads/EURUSD_60_...`.

### P1.3 Тесты и CI
- pytest: scoring invariants, weight normalization, auth, API 401, no-leakage split.
- Coverage цель ≥80% (`python-github-setup`).
- GitHub Actions: ruff, pytest, pip-audit/osv, secret scan, docker build.
- Conventional Commits.

### P1.4 Docker / Railway
- Build context = repo root (текущий `COPY ../data` в `web/Dockerfile` ломается).
- Non-root USER, pinned `python:3.11.9-slim`, `.dockerignore`.
- HEALTHCHECK через stdlib `urllib` на `/healthz`.
- Named volume для sqlite; не bind-mount production code.
- gunicorn workers согласовать с sqlite (WAL + single writer) или уйти на Postgres.

### P1.5 Observability
- JSON logs: request_id, user_id (не PII сверх необходимого), latency.
- Метрики: prediction latency, Yahoo fetch fail rate, chat token usage.
- Не логировать API keys, пароли, session cookies.

---

## P2 — Киллер-фичи и продуктовый потолок

Ниша: **market environment quality** (не предсказание цены). Конкуренты: TradingView Supercharts, QuantConnect research, Bookmap liquidity, Bloomberg/Refinitiv terminals, retail «AI copilot» брокеров.

### P2.1 Честный multi-asset scoring
- Реальный OHLCV per symbol (broker/exchange API), не Yahoo-only scrape.
- Считать AMQSS для каждого инструмента, не клонировать EURUSD.
- Session calendar, spread/cost, holiday calendars.

### P2.2 Research-grade ML
- Walk-forward / purged CV; IC/ICIR по факторам; calibration plots.
- Model registry (версионность, rollback).
- Explainability: SHAP на факторы, не keyword-bot.

### P2.3 Product surface
- Live WebSocket quotes + score stream.
- Alerts (Telegram/email) при смене режима NO TRADE ↔ HIGH QUALITY.
- Paper journal: «я вошёл при score X» → forward outcome (закрывает петлю продукта).
- Billing: on-chain verify TRC-20 или Stripe; ручная активация только как fallback.
- RBAC: viewer / trader / admin.
- i18n ru/en без смешения copy.

### P2.4 Trust UX
- Явно маркировать: live vs delayed vs synthetic.
- Disclaimer + не выдавать BUY/SELL из quality model (сейчас `/api/ai/predict` делает recommendation).
- Onboarding: загрузить свой CSV → score за 30 секунд (core loop продукта).

### P2.5 Performance at 100x
- Precompute scores на worker (Celery/RQ), API только читает.
- Postgres + Redis cache вместо CSV rewrite на каждый run.
- Не train RandomForest в request path.
- Connection pool; CDN для static; gzip.

---

## Порядок внедрения (рекомендуемый)

1. P0.1 Auth/secrets  
2. P0.2 Честность данных в UI  
3. P0.3 Flask production flags  
4. P0.4 ML integrity  
5. P0.5 Дубли и pin deps  
6. P1.1 Package + blueprints  
7. P1.2 Engine library  
8. P1.3 Tests + CI  
9. P1.4 Docker  
10. P1.5 Logging/metrics  
11. P2.x продуктовые фичи по приоритету рынка

Не переписывать весь репозиторий одним PR. Каждый пункт — отдельный вертикальный срез со тестами.
