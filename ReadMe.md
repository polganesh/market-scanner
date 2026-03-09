# Market Scanner

A Python-based daily market scanner that screens a watchlist of 120+ instruments across four technical filters and generates a self-contained HTML report. All data is fetched live from Yahoo Finance via `yfinance`.

---

## Features

- **4 independent technical filters** — Crossover, Supertrend, Trend Alignment, and Pullback
- **Bullish / Bearish split** — every section shows two side-by-side tables so signals are instantly categorised
- **Fibonacci Golden Ratio filter** on Golden Crossovers to reduce false positives
- **Multi-timeframe RSI** for pullback detection (daily, weekly, monthly)
- **Auto-generated HTML report** with colour-coded badges, hover highlights, and timestamped filename
- Runs as a single script — no database, no server required

---

## Instrument Universe

120+ instruments across five categories:

| Category | Examples |
|---|---|
| Energy & Commodities | USO, UNG, XLE, GLD, SLV, IBIT |
| Index ETFs | QQQ, SPY, IWM, TLT, MAGS, URTH |
| Country ETFs | MCHI, EWY, EWJ, EWZ, VWO, TUR, EIS |
| Tech & Large Caps | NVDA, AAPL, MSFT, AMZN, META, TSLA, NVDA… |
| Diversified Large Caps | JPM, JNJ, XOM, WMT, BRK-B, V, MA… |

To modify the watchlist, edit the `USER_INSTRUMENTS` list near the top of `market_scanner.py`.

---

## Filters Explained

### Section 1 — Crossover Signals

Detects when the **SMA 50 crosses the SMA 200** within the last 15 trading days.

- **Bullish (Golden Cross)** — SMA 50 crosses above SMA 200, AND the last closing price is within ±2% of the **Fibonacci 61.8% retracement level** between the 52-week low and 52-week high. This additional filter removes low-quality crossovers that occur far from a meaningful support zone.
- **Bearish (Death Cross)** — SMA 50 crosses below SMA 200. No additional filter applied.

| Parameter | Value |
|---|---|
| Short SMA | 50 |
| Long SMA | 200 |
| Lookback window | 15 trading days |
| Fibonacci level | 0.618 |
| Fibonacci tolerance | ±2% |

**Report columns:** Instrument · Last Price · 52W High · 52W Low · Crossover Type · Crossover Date

---

### Section 2 — Supertrend Signals

Detects a **fresh flip in Supertrend direction** on the daily timeframe within the last 15 trading days.

- **Bullish** — Supertrend flips from bearish (red) to bullish (green)
- **Bearish** — Supertrend flips from bullish (green) to bearish (red)

Supertrend is calculated using ATR-based upper and lower bands, and direction is determined by whether the close is above the upper band or below the lower band.

| Parameter | Value |
|---|---|
| ATR Period | 10 |
| Multiplier | 3 |
| Lookback window | 15 trading days |

**Report columns:** Instrument · Last Price · 52W High · 52W Low · ST Type · ST Date

---

### Section 3 — Up / Down Trend

Checks for **full EMA stack alignment** on the daily timeframe, confirming a clean trend.

- **Bullish** — `Close > EMA 20 > EMA 50 > EMA 200`
- **Bearish** — `Close < EMA 20 < EMA 50 < EMA 200`

The scanner also walks backwards through the price history to identify the exact date the current alignment started (i.e. the trend start date).

| EMA | Span |
|---|---|
| EMA 20 | 20 days |
| EMA 50 | 50 days |
| EMA 200 | 200 days |

**Report columns:** Instrument · Last Price · EMA 20 · EMA 50 · EMA 200 · 52W High · 52W Low · Trend Start Date · Trend Type

---

### Section 4 — Possible Pullback

Identifies instruments showing a **short-term pullback within a longer-term uptrend** using multi-timeframe RSI(14).

**Condition (all three must be true):**

| Timeframe | RSI Condition |
|---|---|
| Monthly | RSI > 60 |
| Weekly | RSI > 60 |
| Daily | RSI < 45 |

The logic is: the monthly and weekly RSI confirm the instrument is in a strong uptrend on higher timeframes, while the daily RSI dipping below 45 signals a short-term pullback — a potential re-entry opportunity.

**Report columns:** Instrument · Last Price · Daily RSI · Weekly RSI · Monthly RSI · 52W High · 52W Low

---

## Requirements

```
python >= 3.8
pandas
numpy
yfinance
pytz
```

Install all dependencies with:

```bash
pip install pandas numpy yfinance pytz
```

---

## Usage

```bash
python market_scanner.py
```

The script will:
1. Loop through all instruments in `USER_INSTRUMENTS`
2. Fetch 2 years of daily OHLCV data per instrument (plus weekly/monthly for pullback RSI)
3. Apply all four filters
4. Save the HTML report in the current directory

**Output filename format:**
```
Market_Report_YYYYMMDD_HHMM.html
```

Example: `Market_Report_20250309_0830.html`

Open the file in any browser — no internet connection required to view it.

---

## Configuration

All key parameters are defined at the top of `market_scanner.py`:

```python
SMA_SHORT       = 50      # Short SMA for crossover
SMA_LONG        = 200     # Long SMA for crossover
LOOKBACK_WINDOW = 15      # Days to look back for crossover/supertrend flip
FIB_LEVEL       = 0.618   # Fibonacci retracement level for Golden Cross filter
FIB_TOLERANCE   = 0.02    # ±2% tolerance around the Fib level
```

---

## Report Structure

The HTML report is divided into four sections. Each section has two colour-coded sub-tables side by side:

| Colour | Meaning |
|---|---|
| 🟢 Green header + green rows | Bullish signals |
| 🔴 Red header + red rows | Bearish signals |

Signal type badges (Golden ✦, Death ✖, Bullish ▲, Bearish ▼) are shown inline in the table cells. Each section is sorted by the most recent signal date.

---

## Data Source

All price data is fetched from **Yahoo Finance** via the `yfinance` library with `auto_adjust=True` (prices adjusted for dividends and splits).

| Data fetch | Period | Interval |
|---|---|---|
| Daily OHLCV | 2 years | Daily |
| Weekly (RSI only) | 5 years | Weekly |
| Monthly (RSI only) | 10 years | Monthly |

Instruments with fewer than 200 daily bars are automatically skipped (insufficient history for SMA 200).

---

## Limitations

- Data accuracy depends entirely on Yahoo Finance availability. Some tickers (especially ETFs or foreign listings) may occasionally return empty data.
- The scanner is designed for **end-of-day** analysis. Running it intraday will use the latest incomplete candle for the current day.
- The Fibonacci filter on Golden Crossovers is strict by design — on most days zero or very few instruments will pass it.
- Multi-timeframe RSI fetches (weekly + monthly) add extra network calls per instrument and increase total scan time.

---

## Disclaimer

This tool is for **informational and research purposes only**. It does not constitute financial advice. Always do your own research before making any investment decisions.
