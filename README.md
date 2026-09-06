# SPY Weekly Option Screener

A screener + backtester for **long SPY weekly call/put contracts**. Pick a
directional view, and it ranks the weekly chain by how favorable each contract
is for a buyer; backtest the same logic on reconstructed history before you
trust it.

> Educational tool. Option prices in free-data mode are **reconstructed and
> approximate**. Nothing here is investment advice. Paper trade first.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# dashboard: Today's Pick + Option Chain + Backtest
streamlit run spy_option_screener/app/streamlit_app.py

# the single best contract to buy for the current/next session
python daily_pick.py --entry 10:30

# pullback -> continuation trade plan, formatted for Slack
python slack_plan.py                 # mrkdwn to stdout
python slack_plan.py --json          # Slack Block Kit payload
python slack_plan.py --for 2026-08-21    # back-check a past session
python research/backtest_trade_plan.py   # test the plan's entry/stop/target

# scheduled Slack alert (posts only when a setup qualifies)
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
python run_alert.py --mode auto          # picks morning/close by the ET clock
python run_alert.py --mode morning --dry-run   # see what it would post
python research/backtest_overnight.py    # why overnight option holds don't work

# CLI backtests
python run_backtest.py --strategy mean_reversion --start 2015-01-01
python run_backtest.py --intraday --entry-time 10:30 --start 2018-01-01
python run_backtest.py --strategy combo --entry-time 10:00 --start 2018-01-01
```

First run downloads ~15 years of SPY + VIX daily bars from yfinance and caches
them to `cache/`. After that it works offline.

## What it does

**Backtest tab** — two engines:
- *Daily indicators*: `pullback` / `trend_continuation` / `mean_reversion` /
  `momentum_breakout` / `vol_regime` / `combo` on close-to-close data,
  optionally with a timed intraday entry.
- *Intraday*: the gap + opening-range-breakout signals with same-day entry at
  10:00 / 10:30 / 11:00 ET, on ~2 years of hourly bars.

Signals:

| Signal | Idea |
|---|---|
| `pullback` | buy the dip inside an established trend once it turns back up |
| `trend_continuation` | trend intact (MA stack), price makes a fresh leg |
| `mean_reversion` | RSI-2 snapback from an extreme |
| `momentum_breakout` | fresh N-day range break in the trend direction |
| `vol_regime` | long premium when VIX is cheap vs its range and realized is rising |
| `gap_continuation` | large opening gap up → continuation (intraday) |
| `opening_range_breakout` | first hold beyond the 9:30–10:00 range (intraday) |

Either way it buys the nearest weekly at a target delta, manages with a
profit-target / stop / time-stop, and reports CAGR, drawdown, Sharpe, win rate,
profit factor and a full trade log.

**What the entry-time + signal work showed** (≈2-year samples, small):
- `pullback` entered ~10:00 ET was the best-behaved signal — a shallow loss
  (≈ −8%, PF 0.8) over Oct-2023→now vs ≈ −47% for the same signal held from the
  close over 2012→now, because it buys weakness (cheaper) and the morning entry
  catches the bounce. `pullback + mean_reversion` at 10:00 came out **flat**
  (≈ 0%, PF 1.0) on that ~2-year, ~45-trade bull-market window — "promising,"
  not "proven."
- `gap_continuation` roughly **halved** the bleed vs daily-only (−15% vs −30%).
- Entering 10:00–10:30 beat 11:00 and the close for dip-buying signals; it made
  breakout/continuation signals *worse* (they chase).
- `opening_range_breakout` alone was a *drag* (−48%) and needs sub-30-minute
  data to validate; it carries a low weight.

Most configurations are still net-negative — the volatility risk premium is
hard to beat on free reconstructed prices.

**Scheduled alerts** — [`run_alert.py`](run_alert.py) posts the trade plan to a
Slack channel via an incoming webhook (`SLACK_WEBHOOK_URL` or `--webhook`).
Default is the **simple card** — five lines, one action each:

```
📈 SPY $764 CALL · exp Aug 27 (3 days left)
GO IF   → SPY is green and above $758 around 10am ET
BUY     → about $3.70  ($370 per contract)
STOP    → $1.85  · sell here if it drops, no exceptions
TARGET  → $6.95 (+88%)  · sell most of it here
DONE BY → Wed Aug 26  · don't hold to expiry
```

Pass `--full` for the detailed card (all Greeks, runner, overnight line, POP).
`--mode close` (~3:50 pm ET) posts "forms at tomorrow's open"; `--mode morning`
(~10:15 am ET) posts it with the live 10:00 price. `--mode auto` picks by the
clock. It posts **only when a setup qualifies** and de-dupes per day. [`.github/workflows/spy-alert.yml`](.github/workflows/spy-alert.yml)
runs it on GitHub's cron (add `SLACK_WEBHOOK_URL` as a repo secret); or use a
local cron — see the crontab snippet at the bottom of this file.

**Overnight holds** — asked for, and tested: buying an option purely to hold
one night **loses** (≈ −4% to −10% per night, `research/backtest_overnight.py`)
— the bid/ask + a night of theta swamp the ~3–5 bps of overnight index drift,
and longer-DTE multi-day holds don't beat the weekly plan either. So there's no
standalone overnight signal. Instead every plan carries an **overnight-carry
line** rating whether holding *that* position through the close is favourable
(uptrend, calm VIX, one weeknight) or not (Friday/holiday weekend, VIX spike).

**Trade Plan tab / `slack_plan.py`** — the pullback→continuation setup written
as a complete, Slack-ready plan: contract + expiration, a limit **entry** price
and the SPY level/time that triggers it, a **stop** (−50% of premium, with the
matching SPY level), **Target 1** (take ⅔ at ≈ +1 ATR) and a **runner** (the
prior swing high), a **time stop**, plus reward:risk, breakeven and POP. Copy
the mrkdwn straight into Slack, or use the Block Kit payload / `--webhook`.
**Backtested** (`research/backtest_trade_plan.py`, 60 setups, 2024–2026):
win rate ~53%, avg trade ≈ +7% of premium, profit factor ≈ 1.2, expectancy
≈ +0.13 R — the first positive result here, though a short one-regime (bull)
sample on reconstructed prices.

**Today's Pick tab / `daily_pick.py`** — the headline feature. Combines the
opening **gap-continuation** signal (big gap up → continues, per the intraday
research), the **opening-range-breakout** signal (~70% of breaks hold), and the
daily trend/mean-reversion bias, then ranks the nearest weekly chain and names
one contract to buy — call or put, **regardless of price**. Runs off intraday
bars as of a chosen decision time (default 10:30 ET); outside market hours it
shows the setup from the last completed session.

**Option Chain tab** — a broker-style options screen: toggle Call/Put and
Buy/Sell, pick an expiration from the weekly pills, and scroll the strike
ladder. Each row shows premium, cost to open, breakeven, % to breakeven,
chance of profit, delta, theta/day, IV — plus a **Score** column (0–100) that
brokers don't give you. Tap a row for the full card: max loss, leverage, all
Greeks, a P/L-at-expiration chart, and a breakdown of why it scored what it
did. "Directional-buy zone only" hides deep-ITM and far-OTM strikes.

The score ranks each contract 0–100 on:

| Component | Weight | Rewards |
|---|---|---|
| Breakeven vs expected move | 30% | breakeven inside a 1σ weekly move |
| Delta | 15% | ~0.45 delta — participation without lottery odds |
| Theta bleed | 20% | low daily decay as a share of premium |
| Vol value | 20% | IV cheap vs SPY's recent realized vol |
| Liquidity | 15% | tight spreads / open interest |

## What SPY history says (read this before trading)

Implied vol has exceeded subsequent realized vol in ~80–85% of weeks. A naive
long-option buyer pays that premium every time — every bundled strategy here
backtests **negative** on 2012–present. The "least bad" is short-term mean
reversion (RSI-2 snapbacks), which buys *after* a drop when the bounce tends to
be sharp. Treat the screener as a way to find the *least overpriced* contract
for a view you already hold from your own edge — not as a buy signal by itself.

## Architecture

```
data/loader.py             yfinance SPY + ^VIX -> cached parquet  (swap for ThetaData/Polygon)
data/intraday.py           intraday bars + per-day features (gap, opening range, timed price)
data/market_calendar.py    NYSE trading days / next valid expiry
pricing/black_scholes.py   price, greeks, IV solver
pricing/vol_model.py       VIX -> weekly ATM IV + skew  (the big approximation)
signals/indicators.py      SMA/EMA/RSI/MACD/ATR/Bollinger/zscore
signals/strategies.py      daily directional signals -> {direction, confidence}
signals/intraday_signals.py  gap_continuation, opening_range_breakout, combine
screener/chain.py          synthetic or live option chain
screener/score.py          rank long calls/puts (breakeven, POP, greeks, leverage...)
screener/daily_pick.py     combine signals -> the one contract to buy today
screener/trade_plan.py     pullback->continuation -> full plan + Slack mrkdwn / Block Kit
backtest/engine.py         event-driven loop (entry_prices + same_day_signal aware)
backtest/intraday.py       glue: intraday signals + timed entry -> engine
backtest/rules.py          entry/exit config (TradeRules, incl. entry_time)
backtest/metrics.py        performance stats
research/intraday_pattern.py   the 9:30->close study (charts + report in outputs/)
app/streamlit_app.py       dashboard
```

## Upgrading to real data

Only `data/loader.py` touches yfinance. Implement `load_history` /
`load_live_chain` against a paid options feed (real historical chains with
per-strike IV) and everything downstream — pricing, signals, scoring,
backtest — works unchanged and far more accurately.

## Tests

```bash
pytest -q
```

Covers put-call parity, IV round-trip, Greek signs, equity skew, no
look-ahead in signals, and an end-to-end backtest run.

## Local cron (alternative to the GitHub Action)

```cron
# m h  dom mon dow   command  (server clock in UTC; adjust for your box)
15 14 * * 1-5  cd /path/to/Spy\ Option\ Screener && .venv/bin/python run_alert.py --mode auto
15 15 * * 1-5  cd /path/to/Spy\ Option\ Screener && .venv/bin/python run_alert.py --mode auto
50 19 * * 1-5  cd /path/to/Spy\ Option\ Screener && .venv/bin/python run_alert.py --mode auto
50 20 * * 1-5  cd /path/to/Spy\ Option\ Screener && .venv/bin/python run_alert.py --mode auto
```

Set `SLACK_WEBHOOK_URL` in the crontab environment or a sourced `.env`.
Runs at both EDT and EST offsets; the per-day marker in `outputs/` stops the second from double-posting.
