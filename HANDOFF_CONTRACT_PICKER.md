# Handoff: make the contract picker real

**Goal.** When the screener shows a gap alert, a user sets what they'd spend on one
contract and how long they'd hold it. The page suggests ONE SPY contract and shows
the profit if SPY reaches the target and the loss if it hits the stop.

**Where it stands.** The "Pick a contract" card on the Signal tab already runs
`suggest_contract` and shows its result, but on a **fake** option chain, and it is
labeled "Preview - fake prices". The selection logic is tested. No live option prices
are connected, so the dollar amounts show the *shape* of the risk and reward, not
real quotes.

## What exists

| Piece | File |
|---|---|
| Alert (direction, target, stop, time stop) | `screener/gap_screener.py` -> `build_view()` returns a `GapSignal` |
| The card (already calls the picker) | `app/streamlit_app.py` -> `render_contract_picker`; `get_chain()` right above it is the one line to swap for a live feed |
| Picker logic (tested) | `screener/contract_picker.py` -> `suggest_contract(...)` returns a `PickResult` |
| The chain schema + a fake chain | `data/fake_chain.py` (schema is in its docstring) |
| Vendor adapters from the earlier version | `data/polygon.py` (Massive), `data/schwab.py`, `screener/chain.py` -> `prep_live_chain` |
| Option pricing | `pricing/black_scholes.py` |
| Tests / offline-vendor test pattern | `tests/test_contract_picker.py`, `tests/test_polygon.py` |

Gap up -> buy a PUT. Gap down -> buy a CALL. Cost is the ask x 100. P&L re-prices the
option at the target/stop N days later (scenarios: 1, 5, 10 days) with the contract's
own IV, and gives up half the spread on exit.

## Do these in order

1. **Read what the model says first.** On the fake chain, a ~2-week put costs about
   $680-$980. Reaching the target (a ~0.7% move) earns roughly +$200-$280 if it
   happens next day, +$100-$180 after a week, and about break-even-or-worse after 10
   days. Hitting the stop (a ~1.9% move against) loses $450-$700. The card shows
   this honestly (headline = the 5-day case, plus a timing caption and table), so
   decide what the finished page should emphasize before building more.
2. **Real chain -> schema.** Write one function that returns a DataFrame in the
   schema in `data/fake_chain.py` from a live vendor (start with Massive/Polygon:
   key only, works on the hosted app). Reuse `prep_live_chain` to fill IV, Greeks,
   mid and spread. The adapters predate this project's rebuild, so expect to fix
   things (Polygon is now Massive; confirm the API base URL).
3. **Swap in the live chain.** The card already shows the suggestion, the `reason`
   when nothing fits, the timing scenarios and the `warnings`. Replace `get_chain()`
   with your vendor-backed function. Keep it cached (the page refreshes every minute
   per viewer, so mind the vendor's rate limit), and handle market closed, stale
   quotes and API failures (show a clear message, never a stale or fake price).
4. **Backtest the option P&L.** The 85% hit rate is about SPY's *price*. Whether an
   *option* makes money after spread, time decay and IV changes has never been
   tested. Do this before the page presents profit as anything but an estimate.
5. **Before sharing the link beyond one person:** see the last section.

## Open decisions (all in `PickerRules` or marked in the code)

- Objective: it currently picks the best modeled % gain at the target. That favors
  cheaper, lower-delta contracts. Is that what we want?
- Stop: the loss uses the *underlying* stop. Should the option have its own exit
  (e.g. a fixed % of premium)?
- IV: held constant (`iv_shift=0`). Test a drop.
- Thresholds: delta 0.35-0.60, spread <= 8%, open interest >= 100, expiry >= hold + 3
  days. All are starting points, not validated.
- Default budget slider ($400-$1,000, range $100-$2,000): the earlier Robinhood
  screenshots showed a ~2-week near-the-money SPY option at ~$520 and a ~4-week one at
  ~$910. Re-check against real prices.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py
.venv/bin/python -m pytest
```
```python
from spy_option_screener.data.fake_chain import fake_chain
from spy_option_screener.screener.contract_picker import suggest_contract
r = suggest_contract(fake_chain(769.40), gap_type="up", spot=769.40,
                     target=763.99, stop=783.83, budget=(500, 1000), hold_days=14)
print(r.suggestion or r.reason)
```

## Access and secrets

- Repo is public: `BrittBanks88/spy-option-screener`. Fork it, or the owner adds you
  as a collaborator. **Pushes to `main` redeploy the live page** - use a branch + PR.
- Never commit keys. Use your own in a local `.streamlit/secrets.toml` or `.env`
  (both gitignored; e.g. `POLYGON_API_KEY`). Only the owner can set keys on the hosted app.

## Before sharing the link with anyone else

- **Data license.** Personal market-data plans generally forbid showing the data to
  third parties (Massive's terms say so; a business license is separate). Check with
  the vendor first.
- **Advice.** Suggesting specific contracts to other people edges toward investment
  advice. Keep the "informational only" framing and get qualified advice before
  making it public or commercial.
