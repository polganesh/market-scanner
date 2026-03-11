import logging
import os
import glob
import yaml
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from datetime import datetime
import pytz
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- Configuration ---
SMA_SHORT       = 50
SMA_LONG        = 200
LOOKBACK_WINDOW = 15
ADX_PERIOD      = 14
RSI_PERIOD      = 14
EMA_FAST        = 20
EMA_MID         = 50
EMA_SLOW        = 200
# ADX interpretation tiers
# 0-18:   Weak/No Trend
# 18-25:  Neutral/Developing
# 25-45:  Strong Trend
# 45-100: Trend Reversal / Exhaustion

# Weekly 200 SMA Zone Filter Configuration
# Qualifies if: weekly_sma200 <= last_close <= weekly_sma200 * (1 + W200_UPPER_BUFFER)
# i.e. price is AT or ABOVE weekly SMA200, but NOT more than W200_UPPER_BUFFER% above it
# Stock A: SMA200=100, close=105 -> qualifies  (above SMA200, within +10% ceiling)
# Stock B: SMA200=100, close=95  -> excluded   (below SMA200, bearish)
# Stock C: SMA200=100, close=115 -> excluded   (+15%, above +10% ceiling, overheated)
W200_SMA_PERIOD   = 200
W200_UPPER_BUFFER = 0.10   # 10% ceiling above weekly SMA200

# ── Default Instruments ────────────────────────────────────────────────────────
# Used ONLY when .yaml-tickers/ folder is absent or contains no valid YAML files.

_DEFAULT_INSTRUMENTS = [
    # --- ENERGY & COMMODITIES ---
    'USO', 'UNG', 'XLE', 'XOP', 'GLD', 'SLV', 'IBIT',
    # --- INDICES ETF ---
    'QQQ', 'SPY', 'IWM', 'TLT', 'MAGS', 'URTH',
    # --- Country ETFs ---
    'MCHI', 'EWY', 'EWJ', 'EWZ', 'VWO', 'TUR', 'EIS',
    # --- TECH & LARGE CAPS ---
    'NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSM', 'META', 'AVGO', 'TSLA', 'BRK-B',
    'WMT', 'LLY', 'JPM', 'XOM', 'V', 'JNJ', 'ASML', 'MA', 'COST', 'ORCL',
    'NFLX', 'MU', 'ABBV', 'CVX', 'PLTR', 'PG', 'HD', 'BAC', 'GE', 'KO',
    'ADBE', 'CRM', 'AMD', 'PEP', 'TMO', 'LIN', 'WFC', 'ACN', 'MCD', 'DIS',
    'PM', 'CSCO', 'ABT', 'INTU', 'TXN', 'GEHC', 'DHR', 'AXP', 'IBM', 'AMAT',
    'CAT', 'PFE', 'VZ', 'QCOM', 'MS', 'NEE', 'HON', 'RTX', 'LOW', 'UNP',
    'AMGN', 'GS', 'SPGI', 'INTC', 'COP', 'SYK', 'BLK', 'ETN', 'TJX', 'VRTX',
    'REGN', 'BKNG', 'LMT', 'BSX', 'BA', 'ADP', 'MDLZ', 'C', 'ADI', 'CB',
    'PANW', 'GILD', 'AMT', 'NOW', 'SNPS', 'CDNS', 'CI', 'ZTS', 'SCHW',
    'PLD', 'DE', 'ISRG', 'LRCX', 'T', 'MMC', 'EL', 'MO', 'HCA', 'UBER',
]


# ── YAML Ticker Loader ─────────────────────────────────────────────────────────

def load_instruments() -> list:
    """
    Loads ticker symbols from all *.yaml / *.yml files inside the
    '.yaml-tickers' folder located in the same directory as this script.

    Supported YAML structures (all handled automatically):

      # 1. Flat list
      - AAPL
      - MSFT

      # 2. List under any top-level key
      tickers: [AAPL, MSFT]

      # 3. Nested sectors with {ticker: X, name: Y} dicts  (NDX format)
      nasdaq_100_constituents:
        sectors:
          tech:
            - { ticker: AAPL, name: "Apple Inc." }
          semis:
            - NVDA

      # 4. Any arbitrary nesting — the recursive walker handles it all.

    Returns:
      Deduplicated, uppercased, sorted list of ticker strings.
      Falls back to _DEFAULT_INSTRUMENTS if the folder is absent,
      empty, or yields no valid tickers after parsing.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_dir   = os.path.join(script_dir, ".yaml-tickers")
    yaml_files = []

    if os.path.isdir(yaml_dir):
        yaml_files = (glob.glob(os.path.join(yaml_dir, "*.yaml")) +
                      glob.glob(os.path.join(yaml_dir, "*.yml")))

    if not yaml_files:
        log.info("No YAML files found in .yaml-tickers/ — using default instrument list "
                 f"({len(_DEFAULT_INSTRUMENTS)} tickers).")
        return list(_DEFAULT_INSTRUMENTS)

    tickers = set()

    def _extract(node):
        """Recursively pull ticker strings out of any YAML node."""
        if isinstance(node, str):
            stripped = node.strip().upper()
            if stripped:
                tickers.add(stripped)
        elif isinstance(node, dict):
            # {ticker: AAPL, name: ...} dict → use the 'ticker' value only
            if "ticker" in node:
                _extract(node["ticker"])
            else:
                for v in node.values():
                    _extract(v)
        elif isinstance(node, list):
            for item in node:
                _extract(item)

    for fpath in sorted(yaml_files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is not None:
                _extract(data)
            log.info(f"Loaded tickers from: {os.path.basename(fpath)}")
        except Exception as e:
            log.warning(f"Could not parse {os.path.basename(fpath)}: {e}")

    if not tickers:
        log.warning("YAML files found but no tickers extracted — using default instrument list.")
        return list(_DEFAULT_INSTRUMENTS)

    loaded = sorted(tickers)
    log.info(f"Total unique tickers loaded from YAML: {len(loaded)}")
    return loaded


# ── Data Helpers ───────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period: str = "2y") -> tuple:
    """
    Returns (df, full_name, market_cap_raw, ipo_year).
    - market_cap_raw : raw USD integer from yfinance; None for ETFs
    - ipo_year       : int year the instrument first traded (from firstTradeDateEpochUtc
                       or firstTradeDateMilliseconds); "N/A" if unavailable
    """
    try:
        t_obj  = yf.Ticker(ticker)
        info   = t_obj.info
        name   = info.get('longName', ticker)
        mktcap = info.get('marketCap', None)   # raw USD integer; None for ETFs

        # IPO / first trade date — yfinance exposes this via firstTradeDateEpochUtc (seconds)
        # or firstTradeDateMilliseconds. We try both and fall back to "N/A" gracefully.
        ipo_year = "N/A"
        epoch_s  = info.get('firstTradeDateEpochUtc', None)
        if epoch_s is None:
            epoch_ms = info.get('firstTradeDateMilliseconds', None)
            if epoch_ms is not None:
                epoch_s = epoch_ms / 1000
        if epoch_s is not None:
            try:
                ipo_year = datetime.utcfromtimestamp(float(epoch_s)).year
            except Exception:
                ipo_year = "N/A"

        df = t_obj.history(period=period, auto_adjust=True)
        if df.empty:
            return None, name, mktcap, ipo_year
        df.index = pd.DatetimeIndex(df.index)
        return df, name, mktcap, ipo_year
    except Exception:
        return None, ticker, None, "N/A"


def fetch_weekly_data(ticker: str) -> pd.DataFrame:
    """
    Fetch weekly OHLCV data with maximum available history.
    A 200-period weekly SMA requires at least 200 weekly bars (~4 years).
    Using "5y" (~260 bars) leaves only ~60 bars of valid SMA200 history which
    produces unreliable SMA200 values near the start of the series.
    We request "max" so the SMA200 is anchored on a full 200-week base.
    Falls back through "10y" -> "5y" if "max" is unavailable for a ticker.
    """
    for period in ("max", "10y", "5y"):
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1wk", auto_adjust=True)
            if df.empty:
                continue
            df.index = pd.DatetimeIndex(df.index)
            # Prefer a dataset with enough runway beyond the 200-bar SMA warmup
            if len(df) >= W200_SMA_PERIOD + 50:
                return df
        except Exception:
            continue
    # Last-resort: return whatever we can get
    try:
        df = yf.Ticker(ticker).history(period="5y", interval="1wk", auto_adjust=True)
        if not df.empty:
            df.index = pd.DatetimeIndex(df.index)
            return df
    except Exception:
        pass
    return None


def fetch_monthly_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period="10y", interval="1mo", auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.DatetimeIndex(df.index)
        return df
    except Exception:
        return None


def format_market_cap(mktcap) -> str:
    """Format raw market cap integer into human-readable string (e.g. $2.45T, $345.00B)."""
    if mktcap is None:
        return "N/A"
    try:
        v = float(mktcap)
        if v >= 1e12:
            return f"${v/1e12:.2f}T"
        elif v >= 1e9:
            return f"${v/1e9:.2f}B"
        elif v >= 1e6:
            return f"${v/1e6:.2f}M"
        else:
            return f"${v:,.0f}"
    except Exception:
        return "N/A"


# ── Indicators via pandas-ta ───────────────────────────────────────────────────

def get_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> float:
    """
    RSI using pandas-ta (Wilder smoothing RMA).
    Matches TradingView RSI exactly.
    """
    try:
        result = ta.rsi(df['Close'], length=period)
        if result is None:
            return None
        val = result.dropna()
        return round(float(val.iloc[-1]), 2) if not val.empty else None
    except Exception:
        return None


def get_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    """
    ADX using pandas-ta (Wilder smoothing RMA).
    Matches TradingView 'ADX and DI' indicator exactly.
    """
    try:
        result = ta.adx(df['High'], df['Low'], df['Close'], length=period)
        if result is None:
            return None
        col = f"ADX_{period}"
        if col not in result.columns:
            return None
        val = result[col].dropna()
        return round(float(val.iloc[-1]), 2) if not val.empty else None
    except Exception:
        return None


def get_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """SMA using pandas-ta."""
    try:
        return ta.sma(df['Close'], length=period)
    except Exception:
        return None


def get_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """EMA using pandas-ta."""
    try:
        return ta.ema(df['Close'], length=period)
    except Exception:
        return None


def get_supertrend_direction(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Supertrend direction using pandas-ta.
    Returns a Series where 1 = bullish, -1 = bearish.
    Matches TradingView Supertrend (10, 3) exactly.
    """
    try:
        result = ta.supertrend(
            df['High'], df['Low'], df['Close'],
            length=period, multiplier=multiplier
        )
        if result is None:
            return None
        # Column name format: SUPERTd_<period>_<multiplier>
        dir_cols = [c for c in result.columns if c.startswith('SUPERTd')]
        if not dir_cols:
            return None
        return result[dir_cols[0]]
    except Exception:
        return None


# ── Signal Detection ───────────────────────────────────────────────────────────

def detect_crossover(df: pd.DataFrame):
    """
    Detects SMA50/SMA200 Golden or Death cross within LOOKBACK_WINDOW bars.
    Uses pandas-ta SMA.
    """
    sma_s = get_sma(df, SMA_SHORT)
    sma_l = get_sma(df, SMA_LONG)
    if sma_s is None or sma_l is None:
        return None, None

    combined = pd.DataFrame({'s': sma_s, 'l': sma_l}).dropna()
    if len(combined) < 2:
        return None, None

    lookback = min(LOOKBACK_WINDOW, len(combined) - 1)
    for i in range(1, lookback + 1):
        prev = float(combined['s'].iloc[-i-1]) - float(combined['l'].iloc[-i-1])
        curr = float(combined['s'].iloc[-i])   - float(combined['l'].iloc[-i])
        if prev < 0 and curr >= 0:
            return "Golden", combined.index[-i].date()
        if prev > 0 and curr <= 0:
            return "Death",  combined.index[-i].date()
    return None, None


def detect_supertrend_change(df: pd.DataFrame):
    """
    Detects Supertrend (10, 3) direction flip within LOOKBACK_WINDOW bars.
    Uses pandas-ta Supertrend.
    """
    direction = get_supertrend_direction(df)
    if direction is None:
        return None, None

    direction = direction.dropna()
    if len(direction) < 2:
        return None, None

    lookback = min(LOOKBACK_WINDOW, len(direction) - 1)
    for i in range(1, lookback + 1):
        prev = int(direction.iloc[-i-1])
        curr = int(direction.iloc[-i])
        if prev == -1 and curr == 1:
            return "Bullish", direction.index[-i].date()
        if prev == 1  and curr == -1:
            return "Bearish", direction.index[-i].date()
    return None, None


def detect_trend(df: pd.DataFrame):
    """
    EMA trend alignment.
    Bullish : Close > EMA20 > EMA50 > EMA200
    Bearish : Close < EMA20 < EMA50 < EMA200
    Uses pandas-ta EMA.
    Returns (trend_type, start_date, ema20, ema50, ema200)
    """
    e20  = get_ema(df, EMA_FAST)
    e50  = get_ema(df, EMA_MID)
    e200 = get_ema(df, EMA_SLOW)

    if e20 is None or e50 is None or e200 is None:
        return None, None, None, None, None

    frame = pd.DataFrame({
        'Close': df['Close'],
        'e20':   e20,
        'e50':   e50,
        'e200':  e200,
    }).dropna()

    if len(frame) < 2:
        return None, None, None, None, None

    last = frame.iloc[-1]
    v20  = round(float(last['e20']),  2)
    v50  = round(float(last['e50']),  2)
    v200 = round(float(last['e200']), 2)

    def is_bull(r): return r['Close'] > r['e20'] > r['e50'] > r['e200']
    def is_bear(r): return r['Close'] < r['e20'] < r['e50'] < r['e200']

    if is_bull(last):
        trend_type = "Bullish"
    elif is_bear(last):
        trend_type = "Bearish"
    else:
        return None, None, v20, v50, v200

    check    = is_bull if trend_type == "Bullish" else is_bear
    start_dt = frame.index[-1].date()
    for i in range(2, len(frame) + 1):
        if check(frame.iloc[-i]):
            start_dt = frame.index[-i].date()
        else:
            break

    return trend_type, start_dt, v20, v50, v200


def detect_pullback(ticker: str, df_daily: pd.DataFrame):
    """
    RSI Pullback: Monthly RSI > 60 AND Weekly RSI > 60 AND Daily RSI < 45.
    All RSI values computed via pandas-ta (Wilder smoothing).
    """
    if df_daily is None or len(df_daily) < RSI_PERIOD + 1:
        return None

    d_rsi = get_rsi(df_daily)
    if d_rsi is None or d_rsi >= 45:
        return None

    df_w  = fetch_weekly_data(ticker)
    if df_w is None or len(df_w) < RSI_PERIOD + 1:
        return None
    w_rsi = get_rsi(df_w)
    if w_rsi is None or w_rsi <= 60:
        return None

    df_m  = fetch_monthly_data(ticker)
    if df_m is None or len(df_m) < RSI_PERIOD + 1:
        return None
    m_rsi = get_rsi(df_m)
    if m_rsi is None or m_rsi <= 60:
        return None

    return round(d_rsi, 2), round(w_rsi, 2), round(m_rsi, 2)


def detect_weekly_sma200_zone(ticker: str, last_price: float, df_weekly: pd.DataFrame = None):
    """
    Weekly SMA200 Zone Filter.

    Qualifies if: weekly_sma200 <= last_price <= weekly_sma200 * (1 + W200_UPPER_BUFFER)
    i.e. price is AT or ABOVE the weekly 200 SMA, but NOT more than 10% above it.

    Real-world examples (SMA200 = $100):
      Stock A: close=$105 -> qualifies  (above SMA200, within +10% ceiling, bullish zone)
      Stock B: close=$95  -> excluded   (below SMA200, bearish)
      Stock C: close=$115 -> excluded   (+15%, above +10% ceiling, overheated / extended)

    "First Entry Date":
      - Scans the ENTIRE available weekly history (after SMA200 is computable)
        and finds the EARLIEST EVER weekly bar where the zone condition was true.
      - This is a historical first-entry date, not a streak start.

    Returns:
      (w_sma200_val, pct_above_sma200, first_ever_date) if qualifies, else None
    """
    try:
        df_w = df_weekly
        if df_w is None:
            df_w = fetch_weekly_data(ticker)
        if df_w is None or len(df_w) < W200_SMA_PERIOD:
            return None

        w_sma200_series = get_sma(df_w, W200_SMA_PERIOD)
        if w_sma200_series is None:
            return None

        # Align close and SMA200 into a clean frame (only rows where SMA200 is available)
        frame = pd.DataFrame({
            'Close':  df_w['Close'],
            'sma200': w_sma200_series,
        }).dropna()

        if frame.empty:
            return None

        w_sma200_val = round(float(frame['sma200'].iloc[-1]), 2)
        upper_bound  = w_sma200_val * (1 + W200_UPPER_BUFFER)

        # Check CURRENT bar qualifies: price in [SMA200, SMA200 * 1.10]
        # Log key values so you can verify the filter is working correctly at runtime
        log.debug(f"  W200 check: last={last_price:.2f}  sma200={w_sma200_val:.2f}  "
                  f"upper={upper_bound:.2f}  weekly_bars={len(frame)}")
        if not (last_price >= w_sma200_val and last_price <= upper_bound):
            return None

        pct_above = round(((last_price - w_sma200_val) / w_sma200_val) * 100, 2)
        log.info(f"  W200 QUALIFIES: last={last_price:.2f}  sma200={w_sma200_val:.2f}  "
                 f"+{pct_above:.2f}%  weekly_bars={len(frame)}")

        # Find the EARLIEST EVER date in the full history where the zone condition was true.
        # Iterates through all rows oldest-first; stops at the first qualifying bar found.
        # Zone condition per bar: row_close in [row_sma200, row_sma200 * 1.10]
        first_ever_date = frame.index[-1].date()   # fallback: today's bar
        for i in range(len(frame)):
            row_close  = float(frame['Close'].iloc[i])
            row_sma200 = float(frame['sma200'].iloc[i])
            row_upper  = row_sma200 * (1 + W200_UPPER_BUFFER)
            if row_close >= row_sma200 and row_close <= row_upper:
                first_ever_date = frame.index[i].date()
                break   # found the earliest qualifying bar — stop

        return w_sma200_val, pct_above, first_ever_date

    except Exception:
        return None


# ── Fibonacci ──────────────────────────────────────────────────────────────────

FIB_ZONES = [
    ("0.382 - 0.500", 0.382, 0.500, "#1a6b3c"),
    ("0.500 - 0.618", 0.500, 0.618, "#2471a3"),
    ("0.618 - 0.786", 0.618, 0.786, "#7d3c98"),
]

def detect_fib_zone(last_price: float, hi_52: float, lo_52: float):
    rng = hi_52 - lo_52
    if rng <= 0:
        return None
    retrace = round((hi_52 - last_price) / rng, 4)
    for label, lo_b, hi_b, color in FIB_ZONES:
        if lo_b <= retrace < hi_b:
            return label, retrace, color
    return None


# ── Main Scanner ────────────────────────────────────────────────────────────────

def run_scanner():
    results = {
        "w200_zone":     [],   # Weekly SMA200 zone: price in [SMA200, SMA200*1.10]
        "combined":      [],
        "bullish_cross": [],
        "bearish_cross": [],
        "bullish_st":    [],
        "bearish_st":    [],
        "bullish_trend": [],
        "bearish_trend": [],
        "pullback":      [],
        "fib_382_500":   [],
        "fib_500_618":   [],
        "fib_618_786":   [],
    }

    # ── Load instruments from .yaml-tickers/ or fall back to defaults ─────────
    instruments = load_instruments()
    log.info(f"Scanner starting — {len(instruments)} instruments to process.")

    for ticker in instruments:
        log.info(f"Scanning {ticker}...")
        df, full_name, mktcap, ipo_year = fetch_data(ticker)
        if df is None or len(df) < SMA_LONG:
            log.warning(f"  Skipping {ticker} - insufficient data")
            continue

        last_price = float(df['Close'].iloc[-1])
        hi_52      = float(df['High'].iloc[-252:].max())
        lo_52      = float(df['Low'].iloc[-252:].min())
        label      = f"{full_name} ({ticker})"
        adx_val    = get_adx(df)
        mktcap_fmt = format_market_cap(mktcap)
        mktcap_raw = mktcap if mktcap else 0   # raw int for JS numeric sort; 0 = ETFs/N/A (sort last)
        ipo_raw    = ipo_year if isinstance(ipo_year, int) else 9999  # 9999 = N/A, sorts last

        # Fetch weekly data once per ticker and reuse across filters
        # (avoids duplicate API calls for pullback + w200 zone filters)
        df_weekly_cached = fetch_weekly_data(ticker)

        # Base row shared across most sections.
        # Market Cap and IPO Year added to every section to enable universal sort.
        # Hidden _*_raw fields carry numeric sort keys; not rendered as columns.
        row_base = {
            "Instrument":  label,
            "Last Price":  round(last_price, 2),
            "52W High":    round(hi_52, 2),
            "52W Low":     round(lo_52, 2),
            "ADX (14)":    adx_val if adx_val is not None else "N/A",
            "Market Cap":  mktcap_fmt,
            "IPO Year":    ipo_year,
            "_mktcap_raw": mktcap_raw,
            "_ipo_raw":    ipo_raw,
        }

        # ── 0. Weekly SMA200 Zone Filter ──────────────────────────────────────
        # Qualifies if: weekly SMA200 <= last_close <= weekly SMA200 * 1.10
        # Stock A (SMA200=100, close=105): qualifies  (bullish, within +10% ceiling)
        # Stock B (SMA200=100, close=95 ): excluded   (below SMA200, bearish)
        # Stock C (SMA200=100, close=115): excluded   (above +10% ceiling, overheated)
        w200_result = detect_weekly_sma200_zone(ticker, last_price, df_weekly=df_weekly_cached)
        if w200_result:
            w_sma200_val, pct_above, first_entry_date = w200_result
            row = {
                "Instrument":       label,
                "Last Price":       round(last_price, 2),
                "Weekly SMA200":    w_sma200_val,
                "% Above SMA200":   pct_above,
                "First Entry Date": first_entry_date,  # earliest ever date criteria was met
                "Market Cap":       mktcap_fmt,
                "IPO Year":         ipo_year,
                "ADX (14)":         adx_val if adx_val is not None else "N/A",
                "52W High":         round(hi_52, 2),
                "52W Low":          round(lo_52, 2),
                "_mktcap_raw":      mktcap_raw,
                "_ipo_raw":         ipo_raw,
            }
            results["w200_zone"].append(row)

        # ── 1. Crossover ──────────────────────────────────────────────────────
        c_type, c_date = detect_crossover(df)
        if c_type:
            row = {**row_base, "Crossover Type": c_type, "Crossover Date": c_date}
            results["bullish_cross" if c_type == "Golden" else "bearish_cross"].append(row)

        # ── 2. Supertrend ─────────────────────────────────────────────────────
        st_type, st_date = detect_supertrend_change(df)
        if st_type:
            row = {**row_base, "ST Type": st_type, "ST Date": st_date}
            results["bullish_st" if st_type == "Bullish" else "bearish_st"].append(row)

        # ── 3. EMA Trend ──────────────────────────────────────────────────────
        trend_type, trend_start, ema20, ema50, ema200 = detect_trend(df)
        if trend_type:
            row = {
                "Instrument":       label,
                "Last Price":       round(last_price, 2),
                "EMA 20":           ema20,
                "EMA 50":           ema50,
                "EMA 200":          ema200,
                "52W High":         round(hi_52, 2),
                "52W Low":          round(lo_52, 2),
                "ADX (14)":         adx_val if adx_val is not None else "N/A",
                "Trend Start Date": trend_start,
                "Trend Type":       trend_type,
                "Market Cap":       mktcap_fmt,
                "IPO Year":         ipo_year,
                "_mktcap_raw":      mktcap_raw,
                "_ipo_raw":         ipo_raw,
            }
            results["bullish_trend" if trend_type == "Bullish" else "bearish_trend"].append(row)

        # ── 4. Pullback ───────────────────────────────────────────────────────
        pb = detect_pullback(ticker, df)
        if pb:
            d_rsi, w_rsi, m_rsi = pb
            row = {
                "Instrument":  label,
                "Last Price":  round(last_price, 2),
                "Daily RSI":   d_rsi,
                "Weekly RSI":  w_rsi,
                "Monthly RSI": m_rsi,
                "ADX (14)":    adx_val if adx_val is not None else "N/A",
                "52W High":    round(hi_52, 2),
                "52W Low":     round(lo_52, 2),
                "Market Cap":  mktcap_fmt,
                "IPO Year":    ipo_year,
                "_mktcap_raw": mktcap_raw,
                "_ipo_raw":    ipo_raw,
            }
            results["pullback"].append(row)

        # ── 5. Fibonacci ──────────────────────────────────────────────────────
        fib = detect_fib_zone(last_price, hi_52, lo_52)
        if fib:
            fib_label, fib_level, _ = fib
            row = {
                "Instrument":       label,
                "Last Price":       round(last_price, 2),
                "52W High":         round(hi_52, 2),
                "52W Low":          round(lo_52, 2),
                "Fib Level":        fib_level,
                "Retracement Zone": fib_label,
                "ADX (14)":         adx_val if adx_val is not None else "N/A",
                "Market Cap":       mktcap_fmt,
                "IPO Year":         ipo_year,
                "_mktcap_raw":      mktcap_raw,
                "_ipo_raw":         ipo_raw,
            }
            if   fib_label == "0.382 - 0.500": results["fib_382_500"].append(row)
            elif fib_label == "0.500 - 0.618": results["fib_500_618"].append(row)
            elif fib_label == "0.618 - 0.786": results["fib_618_786"].append(row)

        # ── 6. Combined ───────────────────────────────────────────────────────
        if pb and fib:
            d_rsi, w_rsi, m_rsi = pb
            fib_label, fib_level, _ = fib
            row = {
                "Instrument":       label,
                "Last Price":       round(last_price, 2),
                "Fib Zone":         fib_label,
                "Fib Level":        fib_level,
                "Daily RSI":        d_rsi,
                "Weekly RSI":       w_rsi,
                "Monthly RSI":      m_rsi,
                "ADX (14)":         adx_val if adx_val is not None else "N/A",
                "52W High":         round(hi_52, 2),
                "52W Low":          round(lo_52, 2),
                "Market Cap":       mktcap_fmt,
                "IPO Year":         ipo_year,
                "_mktcap_raw":      mktcap_raw,
                "_ipo_raw":         ipo_raw,
            }
            results["combined"].append(row)

    save_html_report(results)


# ── HTML Helpers ───────────────────────────────────────────────────────────────

CSS = """<style>
  :root {
    --bull:#27ae60; --bull-lt:#eafaf1;
    --bear:#c0392b; --bear-lt:#fdedec;
    --comb:#b7770d; --comb-lt:#fef9e7;
    --fib1:#1a6b3c; --fib1-lt:#e9f7ef;
    --fib2:#2471a3; --fib2-lt:#eaf3fb;
    --fib3:#7d3c98; --fib3-lt:#f5eef8;
    --w200:#0e6655; --w200-lt:#e8f8f5;  /* Weekly SMA200 zone section color */
    --head:#2c3e50;
    --adx-weak:#95a5a6;
    --adx-neutral:#2471a3;    --adx-neutral-bg:#eaf3fb;
    --adx-strong:#27ae60;     --adx-strong-bg:#eafaf1;
    --adx-exhaust:#c0392b;    --adx-exhaust-bg:#fdedec;
  }
  * { box-sizing:border-box; }
  body { font-family:'Segoe UI',Arial,sans-serif; margin:0; padding:28px;
         background:#ecf0f1; color:#333; }
  h2   { color:var(--head); border-bottom:3px solid var(--head);
         padding-bottom:10px; margin-bottom:28px; }
  .adx-legend {
    display:inline-flex; align-items:center; gap:14px; flex-wrap:wrap;
    background:#fff; border:1px solid #ddd; border-radius:8px;
    padding:6px 14px; font-size:12px; margin-bottom:18px;
  }
  .adx-legend-dot { display:inline-block; width:11px; height:11px;
                    border-radius:50%; margin-right:5px; }
  .section { background:white; border-radius:10px; padding:22px;
             margin-bottom:38px; box-shadow:0 2px 8px rgba(0,0,0,0.09); }
  .sec-title { font-size:16px; font-weight:700; color:var(--head);
               border-bottom:2px solid var(--head);
               padding-bottom:8px; margin-bottom:18px; }
  .pair { display:flex; gap:20px; flex-wrap:wrap; }
  .sub  { flex:1; min-width:300px; }
  .sub-hdr { font-size:13px; font-weight:700; padding:7px 10px;
             border-radius:5px 5px 0 0; color:white; margin-bottom:0; }
  .hdr-bull { background:var(--bull); }
  .hdr-bear { background:var(--bear); }
  .hdr-comb { background:var(--comb); }
  .hdr-fib1 { background:var(--fib1); }
  .hdr-fib2 { background:var(--fib2); }
  .hdr-fib3 { background:var(--fib3); }
  .hdr-w200 { background:var(--w200); }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th    { background:var(--head); color:white; padding:8px 7px;
          text-align:left; white-space:nowrap; }
  /* Sortable column headers — all tables */
  th.sortable { cursor:pointer; user-select:none; }
  th.sortable:hover { background:#3d5166; }
  th.sort-asc::after  { content:' \25B2'; font-size:10px; }
  th.sort-desc::after { content:' \25BC'; font-size:10px; }
  th.sortable:not(.sort-asc):not(.sort-desc)::after
                      { content:' \21C5'; opacity:0.5; font-size:10px; }
  td    { padding:7px; border-bottom:1px solid #e5e5e5; }
  .r-bull td { background:var(--bull-lt); }
  .r-bear td { background:var(--bear-lt); }
  .r-comb td { background:var(--comb-lt); }
  .r-fib1 td { background:var(--fib1-lt); }
  .r-fib2 td { background:var(--fib2-lt); }
  .r-fib3 td { background:var(--fib3-lt); }
  .r-w200 td { background:var(--w200-lt); }
  tr:hover td { filter:brightness(0.95); }
  td.adx-weak    { color:var(--adx-weak) !important; font-style:italic; }
  td.adx-neutral { background:var(--adx-neutral-bg) !important;
                   color:var(--adx-neutral) !important; font-weight:600; }
  td.adx-strong  { background:var(--adx-strong-bg) !important;
                   color:var(--adx-strong) !important; font-weight:700; }
  td.adx-exhaust { background:var(--adx-exhaust-bg) !important;
                   color:var(--adx-exhaust) !important; font-weight:700; }
  /* % Above SMA200: shaded by proximity zones (0-10% range) */
  td.pct-low  { color:#1a8a5a !important; font-weight:600; }   /* 0-3%  */
  td.pct-mid  { color:#0e6655 !important; font-weight:700; }   /* 3-7%  */
  td.pct-high { color:#0a4f40 !important; font-weight:700; font-style:italic; } /* 7-10% */
  .badge { display:inline-block; padding:2px 9px; border-radius:12px;
           font-size:11px; font-weight:700; color:white; }
  .b-bull  { background:var(--bull); }
  .b-bear  { background:var(--bear); }
  .b-gold  { background:#f39c12; }
  .b-death { background:#8e44ad; }
  .no-sig  { color:#aaa; font-style:italic; padding:10px 4px; font-size:12.5px; }
  /* Universal sort bar — present in every section */
  .sort-bar {
    display:flex; align-items:center; gap:5px; flex-wrap:wrap;
    background:#f4f6f8; border:1px solid #d0d7de; border-radius:7px;
    padding:6px 12px; font-size:11px; margin-bottom:8px;
  }
  .sort-bar strong { color:#2c3e50; margin-right:4px; font-size:11px; }
  .sort-btn {
    border:1px solid #7f8c8d; border-radius:4px; background:#fff;
    color:#2c3e50; padding:2px 8px; cursor:pointer; font-size:11px;
    transition:background 0.15s; white-space:nowrap;
  }
  .sort-btn:hover  { background:#d5dce3; }
  .sort-btn.active { background:#2c3e50; color:#fff; font-weight:700; }
  .sort-sep { color:#bdc3c7; margin:0 1px; font-size:10px; }
</style>"""


# ── JavaScript ─────────────────────────────────────────────────────────────────
# Universal client-side sort engine used by ALL tables in every section.
# - Every table has a unique HTML id (e.g. "cross-bull-tbl").
# - Sort buttons call tableSort(tableId, colIndex, defaultDir, btnEl).
# - Column headers with class 'sortable' respond to direct click-to-sort.
# - data-sort attributes on <td> cells carry raw numeric / ISO-date keys for
#   reliable JS ordering (e.g. raw market cap integer, IPO year integer).
# - ETFs with no market cap have _mktcap_raw=0 and sort to the bottom of
#   Market Cap descending sort (i.e. N/A last).
JS_SORT = """<script>
(function () {
  var sortState = {};

  function sortTable(tableId, colIndex, dir) {
    var table = document.getElementById(tableId);
    if (!table) return;
    var tbody = table.querySelector('tbody');
    var rows  = Array.from(tbody.querySelectorAll('tr'));
    var thAll = table.querySelectorAll('th');

    rows.sort(function (a, b) {
      var aCell = a.cells[colIndex];
      var bCell = b.cells[colIndex];
      var aRaw  = aCell ? (aCell.getAttribute('data-sort') || aCell.innerText.trim()) : '';
      var bRaw  = bCell ? (bCell.getAttribute('data-sort') || bCell.innerText.trim()) : '';
      // Numeric sort first; fall back to locale string (handles ISO dates and alpha)
      var aNum  = parseFloat(String(aRaw).replace(/[^0-9.\-]/g, ''));
      var bNum  = parseFloat(String(bRaw).replace(/[^0-9.\-]/g, ''));
      var cmp   = (!isNaN(aNum) && !isNaN(bNum))
                    ? aNum - bNum
                    : String(aRaw).localeCompare(String(bRaw));
      return dir === 'asc' ? cmp : -cmp;
    });

    rows.forEach(function (r) { tbody.appendChild(r); });

    // Update sort-arrow indicators on column headers
    thAll.forEach(function (th, idx) {
      th.classList.remove('sort-asc', 'sort-desc');
      if (idx === colIndex) th.classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
    });

    sortState[tableId] = { col: colIndex, dir: dir };
  }

  // Called by every sort-bar button across all sections
  window.tableSort = function (tableId, colIndex, defaultDir, btnEl) {
    var state = sortState[tableId] || {};
    // Toggle direction if the same column button is clicked again
    var dir = (state.col === colIndex)
                ? (state.dir === 'asc' ? 'desc' : 'asc')
                : defaultDir;
    sortTable(tableId, colIndex, dir);
    // Highlight active button within the parent sort-bar only
    if (btnEl) {
      var bar = btnEl.closest('.sort-bar');
      if (bar) bar.querySelectorAll('.sort-btn').forEach(function (b) {
        b.classList.remove('active');
      });
      btnEl.classList.add('active');
    }
  };

  // Wire sortable column header clicks on page load for all tables
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('table[id]').forEach(function (table) {
      var tid = table.id;
      table.querySelectorAll('th.sortable').forEach(function (th) {
        var idx = Array.from(th.parentNode.children).indexOf(th);
        th.addEventListener('click', function () {
          var state  = sortState[tid] || {};
          var curDir = (state.col === idx)
                         ? (state.dir === 'asc' ? 'desc' : 'asc')
                         : 'asc';
          sortTable(tid, idx, curDir);
          // Deactivate sort-bar buttons when user sorts via header click
          var container = table.closest('.sub') || table.closest('.section');
          if (container) container.querySelectorAll('.sort-btn').forEach(function (b) {
            b.classList.remove('active');
          });
        });
      });
    });
  });
})();
</script>"""


ADX_LEGEND = """
<div class='adx-legend'>
  <strong>ADX (14):</strong>
  <span><span class='adx-legend-dot' style='background:#95a5a6'></span>
        <span style='color:#95a5a6'><strong>0-18</strong> Weak / No Trend</span></span>
  <span><span class='adx-legend-dot' style='background:#2471a3'></span>
        <span style='color:#2471a3'><strong>18-25</strong> Neutral / Developing</span></span>
  <span><span class='adx-legend-dot' style='background:#27ae60'></span>
        <span style='color:#27ae60'><strong>25-45</strong> Strong Trend</span></span>
  <span><span class='adx-legend-dot' style='background:#c0392b'></span>
        <span style='color:#c0392b'><strong>45-100</strong> Exhaustion / Reversal Risk</span></span>
</div>"""


def badge(val):
    s = str(val)
    if s == "Bullish": return "<span class='badge b-bull'>Bullish &#9650;</span>"
    if s == "Bearish": return "<span class='badge b-bear'>Bearish &#9660;</span>"
    if s == "Golden":  return "<span class='badge b-gold'>Golden &#10022;</span>"
    if s == "Death":   return "<span class='badge b-death'>Death &#10006;</span>"
    return str(val)


def adx_cell(val) -> str:
    """ADX cell with colour-coded tier and data-sort for JS numeric sort."""
    try:
        num = float(val)
        ds  = f"data-sort='{num}'"
        if num < 18:
            return f"<td class='adx-weak' {ds}>&#11036; {val}</td>"
        elif num < 25:
            return f"<td class='adx-neutral' {ds}>&#128309; {val}</td>"
        elif num < 45:
            return f"<td class='adx-strong' {ds}>&#128994; {val}</td>"
        else:
            return f"<td class='adx-exhaust' {ds}>&#128308; {val}</td>"
    except (TypeError, ValueError):
        return f"<td data-sort='0'>{val}</td>"


def pct_above_cell(val) -> str:
    """
    Renders the '% Above SMA200' column.
    All qualifying values are in [0, 10%] range.
    Shaded by sub-zone: low (0-3%), mid (3-7%), high (7-10%).
    data-sort holds raw float for JS numeric sorting.
    """
    try:
        num = float(val)
        formatted = f"+{num:.2f}%"
        cls = "pct-low" if num < 3 else ("pct-mid" if num < 7 else "pct-high")
        return f'<td class="{cls}" data-sort="{num}">{formatted}</td>'
    except (TypeError, ValueError):
        return f"<td data-sort='0'>{val}</td>"


# ── Universal table builder ────────────────────────────────────────────────────

# Columns hidden from display but used as JS sort keys via data-sort attributes
_HIDDEN_COLS = {"_mktcap_raw", "_ipo_raw"}

# Columns where a numeric data-sort attribute should be embedded for reliable sorting
_NUMERIC_COLS = {
    "Last Price", "52W High", "52W Low",
    "EMA 20", "EMA 50", "EMA 200",
    "Daily RSI", "Weekly RSI", "Monthly RSI",
    "Fib Level", "% Above SMA200", "Weekly SMA200",
}


def _render_cell(col: str, val, row_dict: dict) -> str:
    """
    Render a single <td> for the given column and value.
    Handles special display columns and embeds data-sort keys where needed.
    """
    if col == "ADX (14)":
        return adx_cell(val)

    if col == "% Above SMA200":
        return pct_above_cell(val)

    # Market Cap: display formatted string, sort by hidden _mktcap_raw (raw USD int).
    # ETFs have _mktcap_raw=0 so they sink to the bottom on Market Cap desc sort.
    if col == "Market Cap":
        raw = row_dict.get("_mktcap_raw", 0) or 0
        return f'<td data-sort="{raw}">{val}</td>'

    # IPO Year: display year, sort numerically.
    # N/A instruments have _ipo_raw=9999 so they sort last on asc sort.
    if col == "IPO Year":
        raw = row_dict.get("_ipo_raw", 9999)
        return f'<td data-sort="{raw}">{val}</td>'

    # Date columns: ISO date strings sort correctly lexicographically
    if col in ("Crossover Date", "ST Date", "Trend Start Date", "First Entry Date"):
        return f'<td data-sort="{val}">{val}</td>'

    # Numeric columns: embed float as data-sort for JS numeric ordering
    if col in _NUMERIC_COLS:
        try:
            return f'<td data-sort="{float(val)}">{val}</td>'
        except (TypeError, ValueError):
            return f'<td data-sort="0">{val}</td>'

    # Default: badge for direction/type values; plain text for everything else
    return f"<td>{badge(val)}</td>"


def _make_sort_bar(table_id: str, buttons: list) -> str:
    """
    Build the sort control bar above a table.
    buttons: list of (label, col_index, default_dir, btn_html_id)
    The first button is rendered as active (default sort on page load).
    """
    html = "<div class='sort-bar'><strong>&#x21C5; Sort:</strong>"
    for i, (lbl, col, ddir, bid) in enumerate(buttons):
        active = " active" if i == 0 else ""
        html += (f"<button id='{bid}' class='sort-btn{active}' "
                 f"onclick=\"tableSort('{table_id}',{col},'{ddir}',this)\">{lbl}</button>")
        if i < len(buttons) - 1:
            html += "<span class='sort-sep'>|</span>"
    html += ("<span style='color:#aaa;font-size:10px;margin-left:5px;'>"
             "&#8505; Click column headers to sort too</span></div>")
    return html


def build_table(rows: list, table_id: str, row_cls: str,
                visible_cols: list, sort_buttons: list,
                default_sort_col: str = None, default_sort_asc: bool = True) -> str:
    """
    Universal HTML table builder used by every section in the report.

    Args:
      rows:             list of row dicts (may contain hidden _* keys)
      table_id:         unique HTML id for the <table> element
      row_cls:          CSS class applied to every <tr> (e.g. 'r-bull')
      visible_cols:     ordered list of column names to render as <th>/<td>
      sort_buttons:     list of (label, col_index, default_dir, btn_id) for the sort bar
      default_sort_col: column used for Python-side pre-sort before rendering
      default_sort_asc: True = ascending, False = descending for the pre-sort
    """
    if not rows:
        return "<p class='no-sig'>No signals detected.</p>"

    df = pd.DataFrame(rows)

    # Pre-sort rows server-side to match the first/active sort button
    if default_sort_col and default_sort_col in df.columns:
        df = df.sort_values(default_sort_col, ascending=default_sort_asc)

    # Columns that are NOT sortable (badge/direction fields + Instrument label)
    non_sortable = {
        "Instrument", "Crossover Type", "ST Type", "Trend Type",
        "Retracement Zone", "Fib Zone",
    }
    sortable_set = set(visible_cols) - non_sortable

    # Build header row
    hdrs = ""
    for col in visible_cols:
        cls = " class='sortable'" if col in sortable_set else ""
        hdrs += f"<th{cls}>{col}</th>"

    # Build body rows
    body = ""
    for _, r in df.iterrows():
        row_dict = r.to_dict()
        cells    = "".join(_render_cell(col, r[col], row_dict) for col in visible_cols)
        body    += f"<tr class='{row_cls}'>{cells}</tr>"

    sort_bar = _make_sort_bar(table_id, sort_buttons)
    return (sort_bar +
            f'<table id="{table_id}"><thead><tr>{hdrs}</tr></thead>'
            f'<tbody>{body}</tbody></table>')


def sub(hdr_cls: str, title: str, content: str) -> str:
    return (f"<div class='sub'><div class='sub-hdr {hdr_cls}'>{title}</div>"
            f"{content}</div>")


# ── HTML Report ────────────────────────────────────────────────────────────────
#
# Sort button tuple format: (display_label, col_index_in_visible_cols, default_dir, btn_id)
# col_index must match the 0-based position of the column in its visible_cols list.
#
# Standard sort options present in EVERY section:
#   Signal/Reference Date — newest first (default)
#   ADX                   — highest first (strongest trend)
#   Daily RSI             — lowest first (most oversold) — pullback sections only
#   A-Z                   — alphabetical by Instrument
#   Market Cap            — largest first; ETFs (N/A) sort to bottom
#   IPO Year              — oldest first (earliest listing); N/A sorts last

def save_html_report(results: dict):
    est     = pytz.timezone("America/New_York")
    now_str = datetime.now(est).strftime("%Y-%m-%d %H:%M %Z")
    secs    = []

    count_w200 = len(results["w200_zone"])

    # ── Section 1 — Weekly SMA200 Zone (TOP SECTION) ─────────────────────────
    # Filter: weekly SMA200 <= last_close <= weekly SMA200 * 1.10
    # Stock A (SMA200=100, close=105): qualifies  (bullish, within +10% ceiling)
    # Stock B (SMA200=100, close=95 ): excluded   (below SMA200, bearish)
    # Stock C (SMA200=100, close=115): excluded   (above +10% ceiling, overheated)
    #
    # Visible column index map (0-based):
    #   0=Instrument  1=Last Price  2=Weekly SMA200  3=% Above SMA200
    #   4=First Entry Date  5=Market Cap  6=IPO Year  7=ADX(14)
    #   8=52W High  9=52W Low
    w200_vis  = ["Instrument", "Last Price", "Weekly SMA200", "% Above SMA200",
                 "First Entry Date", "Market Cap", "IPO Year", "ADX (14)",
                 "52W High", "52W Low"]
    w200_btns = [
        ("% from SMA200 (lowest first)", 3, "asc",  "w200-pct"),
        ("&#128202; ADX (highest first)", 7, "desc", "w200-adx"),
        ("&#128288; A&#8209;Z",           0, "asc",  "w200-alpha"),
        ("&#128197; First Entry (oldest)", 4, "asc", "w200-date"),
        ("&#128176; Market Cap &#9660;",  5, "desc", "w200-mktcap"),
        ("&#128197; IPO Year (oldest)",   6, "asc",  "w200-ipo"),
    ]
    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>&#128198; Section 1 &mdash; Weekly SMA200 Zone
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(Price at or above Weekly SMA({W200_SMA_PERIOD}) &amp; within +{int(W200_UPPER_BUFFER*100)}% ceiling
          &nbsp;|&nbsp; zone: 100%&ndash;{int((1+W200_UPPER_BUFFER)*100)}% of SMA200
          &nbsp;|&nbsp; {count_w200} instrument{"s" if count_w200 != 1 else ""} found)
        </span>
      </div>
      <div style='background:#e8f8f5;border:1px solid #a2d9ce;border-radius:7px;
                  padding:10px 16px;font-size:12px;margin-bottom:14px;line-height:1.9;'>
        <strong style='color:#0e6655;'>&#128270; Filter Logic &mdash; Weekly SMA({W200_SMA_PERIOD}):</strong><br>
        &#9989; <strong>Stock A:</strong> SMA200=$100, Close=$105 &rarr; <em>qualifies</em>
                &nbsp;(above SMA200 &amp; within +10% ceiling &rarr; bullish zone)<br>
        &#10060; <strong>Stock B:</strong> SMA200=$100, Close=$95&nbsp; &rarr; <em>excluded</em>
                &nbsp;(below SMA200 &rarr; bearish)<br>
        &#10060; <strong>Stock C:</strong> SMA200=$100, Close=$115 &rarr; <em>excluded</em>
                &nbsp;(+15%, above +10% ceiling &rarr; overheated / extended)<br>
        &#128197; <strong>First Entry Date</strong> = earliest ever weekly bar in available history
        when this instrument first entered the SMA200 zone
      </div>
      {ADX_LEGEND}
      <div class='pair'>
        {sub("hdr-w200",
             f"&#128200; Instruments in Weekly SMA200 Zone (100%&ndash;{int((1+W200_UPPER_BUFFER)*100)}% of SMA200)",
             build_table(results["w200_zone"], "w200-tbl", "r-w200",
                         w200_vis, w200_btns,
                         default_sort_col="% Above SMA200", default_sort_asc=True))}
      </div>
    </div>""")

    # ── Section 2 — High-Conviction Pullback Entries ──────────────────────────
    # Visible column index map (0-based):
    #   0=Instrument  1=Last Price  2=Fib Zone  3=Fib Level
    #   4=Daily RSI  5=Weekly RSI  6=Monthly RSI  7=ADX(14)
    #   8=52W High  9=52W Low  10=Market Cap  11=IPO Year
    comb_vis  = ["Instrument", "Last Price", "Fib Zone", "Fib Level",
                 "Daily RSI", "Weekly RSI", "Monthly RSI", "ADX (14)",
                 "52W High", "52W Low", "Market Cap", "IPO Year"]
    comb_btns = [
        ("&#128260; Daily RSI (most oversold)", 4,  "asc",  "comb-rsi"),
        ("&#128202; ADX (highest first)",       7,  "desc", "comb-adx"),
        ("&#128288; A&#8209;Z",                 0,  "asc",  "comb-alpha"),
        ("&#128176; Market Cap &#9660;",       10,  "desc", "comb-mktcap"),
        ("&#128197; IPO Year (oldest)",        11,  "asc",  "comb-ipo"),
    ]
    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>&#127919; Section 2 &mdash; High-Conviction Pullback Entries
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(Fibonacci Zone &amp; RSI Pullback simultaneously
          &nbsp;|&nbsp; default sort: lowest Daily RSI first)
        </span>
      </div>
      {ADX_LEGEND}
      <div class='pair'>
        {sub("hdr-comb",
             "&#11088; Instruments in Fibonacci Retracement Zone WITH Daily RSI &lt; 45 "
             "(Monthly RSI &gt;60 &amp; Weekly RSI &gt;60)",
             build_table(results["combined"], "comb-tbl", "r-comb",
                         comb_vis, comb_btns,
                         default_sort_col="Daily RSI", default_sort_asc=True))}
      </div>
    </div>""")

    # ── Section 3 — Fibonacci Retracement Zones ───────────────────────────────
    # Visible column index map (0-based):
    #   0=Instrument  1=Last Price  2=52W High  3=52W Low
    #   4=Fib Level  5=Retracement Zone  6=ADX(14)  7=Market Cap  8=IPO Year
    fib_vis = ["Instrument", "Last Price", "52W High", "52W Low",
               "Fib Level", "Retracement Zone", "ADX (14)", "Market Cap", "IPO Year"]

    def _fib_btns(pfx):
        return [
            ("Fib Level (shallowest first)", 4, "asc",  f"{pfx}-fib"),
            ("&#128202; ADX (highest first)", 6, "desc", f"{pfx}-adx"),
            ("&#128288; A&#8209;Z",           0, "asc",  f"{pfx}-alpha"),
            ("&#128176; Market Cap &#9660;",  7, "desc", f"{pfx}-mktcap"),
            ("&#128197; IPO Year (oldest)",   8, "asc",  f"{pfx}-ipo"),
        ]

    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>&#128208; Section 3 &mdash; Fibonacci Retracement Zones
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(52W High = Level 0 &nbsp;|&nbsp; 52W Low = Level 1
          &nbsp;|&nbsp; default sort: Fib level shallowest first)
        </span>
      </div>
      {ADX_LEGEND}
      <div class='pair'>
        {sub("hdr-fib1",
             "&#128994; Zone 0.382-0.500 &nbsp;<span style='font-weight:400;font-size:11px'>(shallow retracement - strong trend)</span>",
             build_table(results["fib_382_500"], "fib1-tbl", "r-fib1",
                         fib_vis, _fib_btns("fib1"),
                         default_sort_col="Fib Level", default_sort_asc=True))}
        {sub("hdr-fib2",
             "&#128309; Zone 0.500-0.618 &nbsp;<span style='font-weight:400;font-size:11px'>(moderate retracement - key support area)</span>",
             build_table(results["fib_500_618"], "fib2-tbl", "r-fib2",
                         fib_vis, _fib_btns("fib2"),
                         default_sort_col="Fib Level", default_sort_asc=True))}
      </div>
      <div class='pair' style='margin-top:18px'>
        {sub("hdr-fib3",
             "&#128995; Zone 0.618-0.786 &nbsp;<span style='font-weight:400;font-size:11px'>(deep retracement - last major support before trend reversal)</span>",
             build_table(results["fib_618_786"], "fib3-tbl", "r-fib3",
                         fib_vis, _fib_btns("fib3"),
                         default_sort_col="Fib Level", default_sort_asc=True))}
      </div>
    </div>""")

    # ── Section 4 — RSI Pullback Candidates ──────────────────────────────────
    # Visible column index map (0-based):
    #   0=Instrument  1=Last Price  2=Daily RSI  3=Weekly RSI  4=Monthly RSI
    #   5=ADX(14)  6=52W High  7=52W Low  8=Market Cap  9=IPO Year
    pb_vis  = ["Instrument", "Last Price", "Daily RSI", "Weekly RSI", "Monthly RSI",
               "ADX (14)", "52W High", "52W Low", "Market Cap", "IPO Year"]
    pb_btns = [
        ("&#128260; Daily RSI (most oversold)", 2, "asc",  "pb-rsi"),
        ("&#128202; ADX (highest first)",       5, "desc", "pb-adx"),
        ("&#128288; A&#8209;Z",                 0, "asc",  "pb-alpha"),
        ("&#128176; Market Cap &#9660;",        8, "desc", "pb-mktcap"),
        ("&#128197; IPO Year (oldest)",         9, "asc",  "pb-ipo"),
    ]
    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>&#128260; Section 4 &mdash; Possible Pullback
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(Monthly RSI &gt;60 &amp; Weekly RSI &gt;60 &amp; Daily RSI &lt;45
          &nbsp;|&nbsp; default sort: lowest Daily RSI first)
        </span>
      </div>
      {ADX_LEGEND}
      <div class='pair'>
        {sub("hdr-bull", "&#9989; Pullback Candidates",
             build_table(results["pullback"], "pb-tbl", "r-bull",
                         pb_vis, pb_btns,
                         default_sort_col="Daily RSI", default_sort_asc=True))}
      </div>
    </div>""")

    # ── Section 5 — Crossover Signals ────────────────────────────────────────
    # Visible column index map (0-based):
    #   0=Instrument  1=Last Price  2=52W High  3=52W Low  4=ADX(14)
    #   5=Market Cap  6=IPO Year  7=Crossover Type  8=Crossover Date
    cross_vis = ["Instrument", "Last Price", "52W High", "52W Low",
                 "ADX (14)", "Market Cap", "IPO Year", "Crossover Type", "Crossover Date"]

    def _cross_btns(pfx):
        return [
            ("&#128197; Signal Date (newest)", 8, "desc", f"{pfx}-date"),
            ("&#128202; ADX (highest first)",  4, "desc", f"{pfx}-adx"),
            ("&#128288; A&#8209;Z",             0, "asc",  f"{pfx}-alpha"),
            ("&#128176; Market Cap &#9660;",   5, "desc", f"{pfx}-mktcap"),
            ("&#128197; IPO Year (oldest)",    6, "asc",  f"{pfx}-ipo"),
        ]

    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>&#128200; Section 5 &mdash; Crossover Signals
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(SMA {SMA_SHORT} / SMA {SMA_LONG}
          &nbsp;|&nbsp; default sort: most recent signal first)
        </span>
      </div>
      {ADX_LEGEND}
      <div class='pair'>
        {sub("hdr-bull",
             f"&#9989; Bullish &mdash; Golden Crossover (SMA {SMA_SHORT} crosses above SMA {SMA_LONG})",
             build_table(results["bullish_cross"], "cross-bull-tbl", "r-bull",
                         cross_vis, _cross_btns("cross-bull"),
                         default_sort_col="Crossover Date", default_sort_asc=False))}
        {sub("hdr-bear", "&#10060; Bearish &mdash; Death Crossover",
             build_table(results["bearish_cross"], "cross-bear-tbl", "r-bear",
                         cross_vis, _cross_btns("cross-bear"),
                         default_sort_col="Crossover Date", default_sort_asc=False))}
      </div>
    </div>""")

    # ── Section 6 — Supertrend Signals ───────────────────────────────────────
    # Visible column index map (0-based):
    #   0=Instrument  1=Last Price  2=52W High  3=52W Low  4=ADX(14)
    #   5=Market Cap  6=IPO Year  7=ST Type  8=ST Date
    st_vis = ["Instrument", "Last Price", "52W High", "52W Low",
              "ADX (14)", "Market Cap", "IPO Year", "ST Type", "ST Date"]

    def _st_btns(pfx):
        return [
            ("&#128197; Signal Date (newest)", 8, "desc", f"{pfx}-date"),
            ("&#128202; ADX (highest first)",  4, "desc", f"{pfx}-adx"),
            ("&#128288; A&#8209;Z",             0, "asc",  f"{pfx}-alpha"),
            ("&#128176; Market Cap &#9660;",   5, "desc", f"{pfx}-mktcap"),
            ("&#128197; IPO Year (oldest)",    6, "asc",  f"{pfx}-ipo"),
        ]

    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>&#9889; Section 6 &mdash; Supertrend Signals (10, 3)
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(default sort: most recent signal first)
        </span>
      </div>
      {ADX_LEGEND}
      <div class='pair'>
        {sub("hdr-bull", "&#9989; Bullish Supertrend",
             build_table(results["bullish_st"], "st-bull-tbl", "r-bull",
                         st_vis, _st_btns("st-bull"),
                         default_sort_col="ST Date", default_sort_asc=False))}
        {sub("hdr-bear", "&#10060; Bearish Supertrend",
             build_table(results["bearish_st"], "st-bear-tbl", "r-bear",
                         st_vis, _st_btns("st-bear"),
                         default_sort_col="ST Date", default_sort_asc=False))}
      </div>
    </div>""")

    # ── Section 7 — EMA Up/Down Trend ────────────────────────────────────────
    # Visible column index map (0-based):
    #   0=Instrument  1=Last Price  2=EMA 20  3=EMA 50  4=EMA 200
    #   5=52W High  6=52W Low  7=ADX(14)  8=Trend Start Date
    #   9=Trend Type  10=Market Cap  11=IPO Year
    trend_vis = ["Instrument", "Last Price", "EMA 20", "EMA 50", "EMA 200",
                 "52W High", "52W Low", "ADX (14)", "Trend Start Date",
                 "Trend Type", "Market Cap", "IPO Year"]

    def _trend_btns(pfx):
        return [
            ("&#128197; Trend Start (newest)", 8,  "desc", f"{pfx}-date"),
            ("&#128202; ADX (highest first)",  7,  "desc", f"{pfx}-adx"),
            ("&#128288; A&#8209;Z",             0,  "asc",  f"{pfx}-alpha"),
            ("&#128176; Market Cap &#9660;",   10,  "desc", f"{pfx}-mktcap"),
            ("&#128197; IPO Year (oldest)",    11,  "asc",  f"{pfx}-ipo"),
        ]

    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>&#128202; Section 7 &mdash; Up / Down Trend
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(EMA {EMA_FAST} / EMA {EMA_MID} / EMA {EMA_SLOW}
          &nbsp;|&nbsp; default sort: trend start date newest first)
        </span>
      </div>
      {ADX_LEGEND}
      <div class='pair'>
        {sub("hdr-bull",
             f"&#9989; Bullish &mdash; Close &gt; EMA{EMA_FAST} &gt; EMA{EMA_MID} &gt; EMA{EMA_SLOW}",
             build_table(results["bullish_trend"], "trend-bull-tbl", "r-bull",
                         trend_vis, _trend_btns("trend-bull"),
                         default_sort_col="Trend Start Date", default_sort_asc=False))}
        {sub("hdr-bear",
             f"&#10060; Bearish &mdash; Close &lt; EMA{EMA_FAST} &lt; EMA{EMA_MID} &lt; EMA{EMA_SLOW}",
             build_table(results["bearish_trend"], "trend-bear-tbl", "r-bear",
                         trend_vis, _trend_btns("trend-bear"),
                         default_sort_col="Trend Start Date", default_sort_asc=False))}
      </div>
    </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Market Scanner &mdash; {now_str}</title>
  {CSS}
</head>
<body>
  {JS_SORT}
  <h2>&#128269; Market Scanner Report &nbsp;|&nbsp; {now_str}</h2>
  {''.join(secs)}
</body>
</html>"""

    fname = f"Market_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Scan complete. Report saved as: {fname}")


if __name__ == "__main__":
    run_scanner()
