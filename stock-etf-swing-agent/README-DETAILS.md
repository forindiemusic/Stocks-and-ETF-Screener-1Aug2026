# ETF Swing Trading Agent

A Python-based agent that evaluates ETFs **and individual stocks** based on technical, fundamental, and sentiment factors to identify top candidates for near-term outperformance. Supports both **swing** (3–20 day) and **day-trade** (1–5 day) horizons.

## Features

- **Multi-factor Analysis**: Combines technical, fundamental, and sentiment analysis
- **Market Regime Detection**: Adapts scoring based on current market conditions (bull/bear/sideways)
- **Configurable Universe**: Easily customize the ETF universe to screen
- **Dual Asset Support**: Screen ETFs (`--mode etf`), stocks (`--mode stock`), or both (`--mode all`)
- **Dual Horizon Support**: Choose between swing (`--horizon swing`, default) and day-trade (`--horizon day`) scoring
- **Swing Stock Scoring**: Stocks ranked by days-scale score with relative strength vs SPY, risk-adjusted momentum, short MA structure, RSI(5), MACD histogram, and volume confirmation
- **Day-Trade Stock Scoring**: Stocks ranked by ultra-short indicators — RSI(2), 1/2/3-day ROC, Bollinger squeeze, overnight gaps, momentum acceleration, and volume spikes
- **Dual Action Model**: Stocks use a score-based action (Strong Buy / Buy / Hold / Sell); ETFs use growth outlook as the primary recommendation with dividend-yield evaluation as supplementary context (e.g., "Buy | Yield: 3.74% (low) | Currently owned")
- **Rotation Signals**: Tracks rankings across runs and flags symbols that have dropped significantly since the previous evaluation
- **Risk Management**: Built-in position sizing and correlation limits; separate stop-loss/take-profit multipliers for day-trade mode
- **CLI Output**: Clear, formatted output with rankings and scores
- **Extensible Design**: Modular components for easy extension
- **Modern Tooling**: Type hints (mypy strict), pre-commit hooks, GitHub Actions CI, pinned dependencies

## Installation

** For Windows 11 please see Platform-Specific Installation section **

```bash
# Clone the repository
git clone <repository-url>
cd stock-etf-swing-agent

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (may require system dependencies)
# On Ubuntu/Debian:
sudo apt-get install libta-lib0 libta-lib-dev
pip install ta-lib

# Or use the pre-built wheels:
pip install TA-Lib

# Install dev dependencies (optional, for testing/linting)
pip install -r requirements-dev.txt
```

## Configuration

Edit `config.yaml` to customize:
- ETF universe to screen
- Technical/fundamental factor weights
- Market regime detection parameters
- Risk management settings
- Day-trade horizon settings (stop-loss, take-profit, thresholds)
- Output preferences

A documented template is available at `config.example.yaml`.

## Usage

```bash
# Run the agent (defaults to --mode stock, --horizon swing)
python etf_and_stock_agent.py

# Run with custom config
python etf_and_stock_agent.py --config my_config.yaml

# Screen only ETFs from config.yaml's etf_universe
python etf_and_stock_agent.py --mode etf

# Screen only stocks listed in corrently_own_stocks.dat
python etf_and_stock_agent.py --mode stock

# Screen both ETFs and stocks (union of the two universes)
python etf_and_stock_agent.py --mode all

# Screen ONLY the ETFs listed in currently_own_etf.dat (your holdings)
python etf_and_stock_agent.py --mode owned-etf

# Day-trade horizon (1-5 day hold) — uses ultra-short indicators
python etf_and_stock_agent.py --mode stock --horizon day

# Run tests
python test_agent.py
```

### Modes

| Mode | Universe source | Ranking |
|------|----------------|---------|
| `etf` (default in config) | `etf_universe` in `config.yaml` | Composite score |
| `stock` | `corrently_own_stocks.dat` (one symbol per line) | **Short-term score** (swing) or **day-trade score** (day) |
| `all` | Union of the above two | Composite score for ETFs, short-term/day-trade score for stocks |
| `owned-etf` | `currently_own_etf.dat` (one symbol per line) | Composite score; never filtered by threshold |

> Note: the stock file is named `corrently_own_stocks.dat` (intentional spelling). If the file is empty or missing, stock mode falls back to the config's `etf_universe`.

### Horizons

| Horizon | Flag | Hold Period | Indicators | Score Threshold |
|---------|------|-------------|------------|-----------------|
| Swing (default) | `--horizon swing` | 3–20 days | RSI(5), ROC(5/10), SMA5/10/20, MACD, ATR(5) | 0.35 |
| Day-trade | `--horizon day` | 1–5 days | RSI(2), ROC(1/2/3), Bollinger squeeze, gaps, ATR(2) | 0.30 |

## Output

The agent prints the **top 3 recommendations** with detailed metrics:

- **Action** — score-based for stocks (Strong Buy / Buy / Hold / Sell); for ETFs: growth outlook primary | dividend-yield evaluation supplementary (e.g., "Buy | Yield: 3.74% (low) | Currently owned")
- **Currently owned** — shown on the action line when the ETF is listed in `currently_own_etf.dat`
- **4-Week Growth Outlook** (ETFs only) — price-appreciation potential from momentum, sentiment/demand, price trend, market regime, and volume
- **Day-Trade Score** (stocks, `--horizon day`) — 1–5 day ranking from ultra-short indicators
- **Short-term Score** (stocks, `--horizon swing`) — days-scale, risk-adjusted, relative to SPY
- Dividend yield (%)
- Current price with 1-day and 1-week % change
- **Stop-loss and take-profit** dollar amounts derived from ATR (tighter multipliers in day-trade mode)
- Composite score with technical / fundamental / sentiment components
- **Sentiment source** — shows article count for real news, or "price-momentum proxy" when fallback is used
- Market regime (bull/bear/sideways + volatility)
- **Rotation signals**: flags symbols that dropped from the previous run's top 5

Example (`--mode stock --horizon swing`):

```
==============================================================================
TOP 3 STOCKS RECOMMENDATIONS (SWING HORIZON)
==============================================================================
AAPL: Strong Buy
   Dividend Yield : 0.31%
   Price          : $333.74 (1D: +0.14%, 1W: +5.18%)
   Stop-loss      : $317.48 | Take-profit: $358.13 (ATR: $8.13)
   Short-term Score: 0.745 (days-scale ranking)
   Composite Score: 0.543 (Tech: 0.573, Fund: 0.350, Sent: 0.492)
   Sentiment Source: 9 news articles
   Regime         : bull (moderate vol)

CVX: Strong Buy
   Dividend Yield : 3.69%
   Price          : $187.38 (1D: +1.91%, 1W: +2.84%)
   Stop-loss      : $179.57 | Take-profit: $199.10 (ATR: $3.91)
   Short-term Score: 0.725 (days-scale ranking)
   Composite Score: 0.542 (Tech: 0.572, Fund: 0.320, Sent: 0.520)
   Sentiment Source: 10 news articles
   Regime         : bull (moderate vol)

DOW: Buy
   Dividend Yield : 5.85%
   Price          : $29.92 (1D: +2.12%, 1W: -1.48%)
   Stop-loss      : $27.59 | Take-profit: $33.42 (ATR: $1.17)
   Short-term Score: 0.691 (days-scale ranking)
   Composite Score: 0.560 (Tech: 0.596, Fund: 0.290, Sent: 0.541)
   Sentiment Source: 8 news articles
   Regime         : bull (moderate vol)
```

Example (`--mode stock --horizon day`):

```
==============================================================================
TOP 3 STOCKS RECOMMENDATIONS (DAY-TRADE HORIZON)
==============================================================================
AAPL: Strong Buy
   Dividend Yield : 0.31%
   Price          : $333.74 (1D: +0.14%, 1W: +5.18%)
   Stop-loss      : $321.54 | Take-profit: $346.00 (ATR: $8.13)
   Day-Trade Score: 0.812 (1-5 day ranking)
   Composite Score: 0.543 (Tech: 0.573, Fund: 0.350, Sent: 0.492)
   Sentiment Source: 9 news articles
   Regime         : bull (moderate vol)
```

Example (`--mode etf`):

```
==============================================================================
TOP 3 ETFS RECOMMENDATIONS (SWING HORIZON)
==============================================================================
XLE: Strong Buy | Yield: 3.74% (low) | Currently owned
   Dividend Yield : 3.74%
   Price          : $92.15 (1D: +0.87%, 1W: +2.33%)
   Stop-loss      : $88.42 | Take-profit: $97.73 (ATR: $1.87)
   4-Week Growth   : 0.786 (Mom: 0.823, Sent: 0.514, Trend: 0.833, Regime: 1.000, Vol: 0.712)
   Composite Score: 0.563 (Tech: 0.591, Fund: 0.480, Sent: 0.541)
   Sentiment Source: 8 news articles
   Regime         : bull (moderate vol)

XLF: Buy | Yield: 1.03% (low) | Currently owned
   Dividend Yield : 1.03%
   Price          : $58.22 (1D: +0.52%, 1W: +1.89%)
   Stop-loss      : $56.14 | Take-profit: $61.38 (ATR: $1.04)
   4-Week Growth   : 0.677 (Mom: 0.745, Sent: 0.471, Trend: 0.722, Regime: 1.000, Vol: 0.598)
   Composite Score: 0.568 (Tech: 0.583, Fund: 0.510, Sent: 0.520)
   Sentiment Source: 7 news articles
   Regime         : bull (moderate vol)

SPY: Hold | Yield: 0.76% (low)
   Dividend Yield : 0.76%
   Price          : $743.29 (1D: -0.99%, 1W: -0.78%)
   Stop-loss      : $726.22 | Take-profit: $768.89 (ATR: $8.53)
   4-Week Growth   : 0.345 (Mom: 0.000, Sent: 0.337, Trend: 0.304, Regime: 1.000, Vol: 0.648)
   Composite Score: 0.584 (Tech: 0.477, Fund: 0.796, Sent: 0.536)
   Sentiment Source: 9 news articles
   Regime         : bull (moderate vol)
```

> Note: `main()` prints the top 3 detailed recommendations with stop-loss,
> take-profit, and sentiment source. The richer `display_results()` view
> (position sizing, full score legend) is used by `debug_agent.py` and
> `test_agent.py`. CSV export (`output/etf_signals.csv`) is produced by the
> backtester, not by `main()`.

## Methodology

### Composite Score (ETFs)

The composite score is a weighted blend of three factors (weights defined in
`composite_weights` in `config.yaml` and shared with the backtester):

```
composite = technical * 0.50 + fundamental * 0.30 + sentiment * 0.20
```

### Composite Score (Stocks)

For stocks, fundamental and sentiment weights are reduced because ETF-specific
fields (expense ratio, tracking error, AUM) are meaningless and news scraping
is unreliable. The stock composite uses:

```
composite = technical * 0.80 + fundamental * 0.10 + sentiment * 0.10
```

Stock fundamental scoring uses only liquidity (50%) and Sharpe ratio (50%).
Stock technical scoring reduces mean-reversion weight (0.20 → 0.05) and
redistributes to trend (0.35) and momentum (0.35), since mean-reversion
penalizes trending stocks near upper Bollinger Bands.

### Technical Analysis (50% weight)
- Trend: Price vs moving averages, MACD
- Momentum: RSI, Rate of Change
- Mean Reversion: Bollinger Bands position
- Volume: Volume trends and OBV
- Volatility: ATR and historical volatility

### Fundamental Analysis (30% weight)
- Expense ratio (lower is better)
- Liquidity (average volume)
- Assets Under Management (AUM)
- Tracking error vs benchmark
- Dividend yield
- Risk-adjusted returns (Sharpe ratio)

### Sentiment Analysis

**Source**: `yfinance`'s built-in `ticker.news` API — fast, free, no web
scraping needed. Fetches the 20 most recent news articles per symbol and
scores each title + summary with TextBlob polarity. Falls back to a
price-momentum proxy if no articles are returned.

**Weight**: 20% for ETFs, 10% for stocks (reduced because the signal is
noisier for individual names).

The output shows the article count (e.g., "9 news articles") or "price-momentum
proxy (no news found)" so you can assess signal quality at a glance.

### Market Regime Context
- Bull/Bear/Sideways market detection using SPY vs SMA200
- Volatility regime (VIX levels)
- Reported alongside results (currently informational; not a score weight)

### Short-Term Stock Scoring (stocks only, swing horizon)

When running with `--mode stock --horizon swing` (or `all`), each stock is scored on a
**days-scale** basis so recommendations target gains over days rather than
weeks. The score (0.0–1.0) is computed from a 1-month daily window in
`indicators.py` (`calculate_short_term_indicators` + `calculate_short_term_score`):

- **Risk-adjusted relative momentum (35%)**: 5-day and 10-day rate-of-change
  minus SPY's ROC, divided by ATR(5)/price. Rewards stocks that are
  outperforming the market on a risk-adjusted basis.
- **Short MA structure (25%)**: price > SMA5 > SMA10 > SMA20 alignment
- **RSI(5) (15%)**: prefers a healthy 40–75 zone (momentum without being
  extremely overbought)
- **MACD histogram (15%)**: positive = near-term upward acceleration
- **Volume confirmation (10%)**: average of last 3 days vs 20-day average

In `stock` mode the screening is **ranked by `short_term_score`**; ETFs (and
`all` mode's ETF portion) continue to rank by the composite score above.

### Day-Trade Stock Scoring (stocks only, day-trade horizon)

When running with `--mode stock --horizon day`, each stock is scored on an
**ultra-short (1–5 day)** basis using a separate set of indicators optimized
for catching moves over 1–3 days. The score (0.0–1.0) is computed from a
~2-week daily window in `indicators.py`
(`calculate_day_trade_indicators` + `calculate_day_trade_score`):

- **Risk-adjusted ultra-short momentum (30%)**: 1-day, 2-day, and 3-day
  rate-of-change minus SPY's ROC, divided by ATR(2)/price. Captures the
  most recent price action with maximum sensitivity.
- **Momentum acceleration (15%)**: ROC(1) minus ROC(3). Positive values mean
  the stock is getting stronger, not fading — a key day-trade signal.
- **Gap analysis (15%)**: Overnight gap percentage and whether the gap
  direction held through the trading day. Bullish gaps that hold score
  highest; gap-down reversals also score well.
- **RSI(2) (15%)**: Ultra-sensitive 2-period RSI. Prefers the 30–70 zone
  (not exhausted). Scores above 80 or below 20 are penalized.
- **Bollinger squeeze (10%)**: Volatility contraction (narrow Bollinger
  Bands) often precedes explosive breakouts. Lower squeeze values = higher
  score.
- **Proximity to 5-day high (10%)**: Price near the 5-day high suggests a
  breakout is in progress; near the 5-day low suggests fading.
- **Volume spike (5%)**: Today's volume vs 5-day average. Spikes confirm
  institutional interest.

In `day` mode the screening is **ranked by `day_trade_score`** with a lower
threshold (0.30) since day-trade scores cluster lower than swing scores.

### 4-Week Growth Outlook (ETFs only)

ETFs can appreciate through demand, sentiment, and momentum — not just
dividends. The 4-week growth outlook estimates near-term price appreciation
potential from non-dividend factors. It is computed in `indicators.py`
(`calculate_4week_growth_outlook`) from a 1-month daily window:

- **Risk-adjusted relative momentum (35%)**: 5-day and 10-day ROC vs SPY,
  scaled by ATR(5)/price. Rewards ETFs outperforming the market on a
  risk-adjusted basis.
- **Sentiment / demand proxy (25%)**: News sentiment score (0.0–1.0) from
  TextBlob polarity analysis. Reflects market demand and narrative.
- **1-week price trend (15%)**: Recent price direction; +2% or more = 1.0,
  flat = 0.5, -2% or worse = 0.0.
- **Market regime (15%)**: Bull = 1.0, sideways = 0.5, bear = 0.15.
- **Volume confirmation (10%)**: Average of last 3 days vs 20-day average.

The growth outlook is the **primary** ETF recommendation and ranking criteria,
displayed first in the output. The dividend-yield evaluation follows as
supplementary context, e.g. `Strong Buy | Yield: 3.74% (low) | Currently owned`.

### Action Models

**Stocks (swing horizon)** use a score-based action model driven by short-term momentum:

| Short-term Score | Action      |
|------------------|-------------|
| ≥ 0.70           | Strong Buy  |
| 0.50 – 0.69      | Buy         |
| 0.30 – 0.49      | Hold        |
| < 0.30           | Sell        |

**Stocks (day-trade horizon)** use the same thresholds but ranked by `day_trade_score`:

| Day-Trade Score | Action      |
|-----------------|-------------|
| ≥ 0.70          | Strong Buy  |
| 0.50 – 0.69     | Buy         |
| 0.30 – 0.49     | Hold        |
| < 0.30          | Sell        |

Day-trade mode uses a lower minimum threshold (0.30 vs 0.35 for swing) and
tighter stop-loss (1.5× ATR) and take-profit (2.0× ATR) multipliers,
configurable in the `day_trade` section of `config.yaml`.

**ETFs** use a **growth-primary** recommendation model:

1. **4-week growth outlook (primary)** — price-appreciation-focused; combines
   risk-adjusted relative momentum (35%), news sentiment / demand proxy (25%),
   1-week price trend (15%), market regime (15%), and volume confirmation (10%).
   **ETFs are ranked by growth score** — Strong Buy > Buy > Hold > Sell.

2. **Dividend-yield evaluation (supplementary)** — income-focused context;
   yield is labeled good (≥5.0%), standard (4.0–4.9%), or low (≤3.9%).
   Yield does **not** affect ranking.

3. **Currently owned** — appended to the action line when the ETF appears in
   `currently_own_etf.dat`.

Output format: `Growth Action | Yield: X.XX% (good/standard/low) | Currently owned`

**Growth outlook thresholds (primary):**

| Growth Score | Action      |
|--------------|-------------|
| ≥ 0.70       | Strong Buy  |
| 0.50 – 0.69  | Buy         |
| 0.30 – 0.49  | Hold        |
| < 0.30       | Sell        |

**Dividend-yield evaluation (supplementary, does not affect ranking):**

| Dividend Yield | Yield Label |
|----------------|-------------|
| ≥ 5.0%         | good        |
| 4.0% – 4.9%    | standard    |
| ≤ 3.9%         | low         |

Dividend yield is computed from `trailingAnnualDividendRate / price` (robust
to Yahoo's inconsistent `dividendYield` field).

### Rotation / Exit Signals

Each run saves rankings to `output/last_rankings.csv`. On the next run, any
symbol that was previously in the top 5 but has dropped below the top 10 (or
below a 0.30 score threshold) is flagged as a rotation exit candidate.

## Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

## License

MIT

## Backtesting Framework

The agent includes a comprehensive backtesting framework to validate strategy performance:

### Features
- **Historical Simulation**: Test the strategy on past data
- **Multiple Rebalancing Frequencies**: Daily, weekly, or monthly rebalancing
- **Performance Metrics**: Returns, volatility, Sharpe ratio, max drawdown
- **Benchmark Comparison**: Compare against SPY (S&P 500 ETF)
- **Holdings Analysis**: See what ETFs were selected at each rebalancing point
- **Stock Support**: Mirrors the live agent's modes (`etf`, `stock`, `all`,
  `owned-etf`). Stocks are ranked by `short_term_score` (days-scale), use a
  stock-specific technical-weight profile and an 80/10/10 composite blend
  (ETF-centric fundamentals are down-weighted). ETFs are ranked by the
  4-week growth outlook (primary) with a composite-score fallback.

### Usage
```bash
# Run a sample backtest
python3 run_backtest_example.py

# Or run the backtester directly
python3 backtest.py
```

The `ETFBacktester` constructor accepts a `mode` argument (`'etf'`, `'stock'`,
`'all'`, `'owned-etf'`) and an optional `stock_symbols` list. When `mode` is
omitted but `etf_universe` is provided, it defaults to ETF behavior. In
`'stock'` mode the universe is read from `corrently_own_stocks.dat` (falling
back to `config.yaml`'s `etf_universe`), and every symbol is treated as a
stock. In `'all'` mode, stocks and ETFs are mixed and ranked on a common 0–1
scale via `_rank_score_for()`.

### Backtest Results Example
```
BACKTEST RESULTS
==================================================
Period: 2026-01-06 to 2026-07-05
Rebalancing: M
Number of periods: 5
--------------------------------------------------
RETURNS:
  Strategy Cumulative Return: 155.81%
  Benchmark (SPY) Cumulative Return: 36.89%
  Strategy Annual Return: 379.46%
  Benchmark Annual Return: 68.88%
  Excess Return: 118.92%
--------------------------------------------------
RISK METRICS:
  Strategy Sharpe Ratio: 25.44
  Benchmark Sharpe Ratio: 4.89
  Strategy Max Drawdown: 0.00%
  Benchmark Max Drawdown: -2.27%
  Strategy Volatility (Annual): 14.91%
  Benchmark Volatility (Annual): 14.09%
==================================================

PORTFOLIO HOLDINGS HISTORY (last 3 rebalances):
  2026-03-31: XLE, VGK, XLF, VEA, EWJ
  2026-04-30: XLE, EWJ, VLUE, XLV, MTUM
  2026-05-29: VLUE, XLK, MTUM, VWO, QQQ
```

### Customization
Modify the backtest parameters in `run_backtest_example.py`:
- ETF universe: Add/remove ETFs to test
- Time period: Adjust start/end dates

## Project Structure

```
stock-etf-swing-agent/
├── pyproject.toml              # Modern packaging with tool configs
├── config.yaml                 # User configuration
├── config.example.yaml         # Documented configuration template
├── .env.example                # Environment variables template
├── requirements.txt            # Pinned production dependencies
├── requirements-dev.txt        # Pinned development dependencies
├── .pre-commit-config.yaml     # Pre-commit hooks (ruff, mypy, bandit)
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── etf_and_stock_agent.py      # Main agent (CLI entry point)
├── indicators.py               # Technical indicator calculations
├── scoring.py                  # Scoring functions (technical, fundamental)
├── retry.py                    # Retry utility with exponential backoff
├── backtest.py                 # Backtesting framework
├── debug_agent.py              # Debug script for single-ETF testing
├── quick_start.py              # Quick-start installation script
├── test_agent.py               # Integration test for full pipeline
├── tests/
│   ├── conftest.py             # Shared test fixtures
│   ├── test_indicators.py      # Unit tests for indicators
│   ├── test_scoring.py         # Unit tests for scoring
│   └── test_backtest.py        # Unit tests for backtester
└── output/                     # Generated rankings and signals
```

## Development

### Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=src --cov-report=term-missing
```

### Type Checking
```bash
python3 -m mypy etf_and_stock_agent.py indicators.py scoring.py retry.py
```

### Linting
```bash
# Check
ruff check src tests

# Format
ruff format --check src tests
```

## Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

## License

MIT
- Rebalancing frequency: Change to 'D' (daily), 'W' (weekly), or 'M' (monthly)
- Lookback period: Modify how much historical data is used for evaluation

## Platform-Specific Installation

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies for TA-Lib
sudo apt-get update
sudo apt-get install libta-lib0 libta-lib-dev

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install TA-Lib

# Verify installation
python3 -c "import talib; print('TA-Lib version:', talib.__version__)"
```

### Windows 11

Open powershell 

type: python  and hit enter

# Above made Microsoft Store come up-  Use it to Install Python Install Manager

open a regular command prompt

cd to where python was install

** for me it was C:\Users\<my-user-name>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python\Python 3.14

curl -O  https://bootstrap.pypa.io/get-pip.py

# go back to the powershell 

> cd to C:\Users\susim\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python\Python 3.14

# type and follow prompts

>  python get-pip.py

# Make a note of where pip was install for me it was.  
#   C:\Users\<my-user-name>\AppData\Local\Python\pythoncore-3.14-64\Scripts

#  cd to above directory

# Install TA-Lib using pre-built wheels (recommended)

>  ./pip install TA-LIB

# If wheels fail, download from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Then install: pip install path\to\TA_Lib‑X.X.X‑cpXX‑cpXX‑win_amd64.whl

## Go back to the command promp and cd where above was installed

# For me  C:\Users\<my-user-name>\AppData\Local\Python\pythoncore-3.14-64\Scripts

# cd to above directory type and hit enter

> python -m venv venv

> .\venv\Scripts\activate

# Install Python dependencies

> pip install --upgrade pip  # did not do this since I had just installed the latest version

> pip install  -r  C:\Users\<my-user-name>\Documents\ETF-Sreener-for-the-rest-of-us-main\stock-etf-swing-agent\requirements.txt

# cd to where ETF-Screener code was saved on your system

> cd C:\Users\<my-user-name>\Documents\ETF-Sreener-for-the-rest-of-us-main\stock-etf-swing-agent

> python etf_and_stock_agent.py



### Common Issues

**Linux: TA-Lib installation fails**
```bash
# Try installing build dependencies first
sudo apt-get install build-essential
pip install --no-binary TA-Lib TA-Lib
```

**Windows: TA-Lib DLL not found**
- Ensure you installed the correct wheel for your Python version
- Check that `TA_Lib.dll` is in your PATH or Python directory

**Rate limiting errors**
- The agent includes built-in rate limiting for Yahoo Finance API calls
- If you see rate limit warnings, reduce `max_workers` in `config.yaml`
