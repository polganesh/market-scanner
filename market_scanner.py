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
SMA_SHORT = 50
SMA_LONG = 200
LOOKBACK_WINDOW = 15
ADX_PERIOD = 14
RSI_PERIOD = 14
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
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
W200_SMA_PERIOD = 200
W200_UPPER_BUFFER = 0.10  # 10% ceiling above weekly SMA200

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


# ── YAML Ticker Loader (Refactored for Multi-File Splitting) ───────────────────

def load_instruments_grouped() -> dict:
    """
    Returns a dictionary of { "filename": [list_of_tickers] }.
    Falls back to {"Default": _DEFAULT_INSTRUMENTS} if no files found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_dir = os.path.join(script_dir, ".yaml-tickers")
    yaml_files = []

    if os.path.isdir(yaml_dir):
        yaml_files = (glob.glob(os.path.join(yaml_dir, "*.yaml")) +
                      glob.glob(os.path.join(yaml_dir, "*.yml")))

    if not yaml_files:
        log.info("No YAML files found in .yaml-tickers/ — using default instrument list.")
        return {"Default": list(_DEFAULT_INSTRUMENTS)}

    groups = {}

    def _extract(node, ticker_set):
        """Recursively pull ticker strings out of any YAML node."""
        if isinstance(node, str):
            stripped = node.strip().upper()
            if stripped:
                ticker_set.add(stripped)
        elif isinstance(node, dict):
            if "ticker" in node:
                _extract(node["ticker"], ticker_set)
            else:
                for v in node.values():
                    _extract(v, ticker_set)
        elif isinstance(node, list):
            for item in node:
                _extract(item, ticker_set)

    for fpath in sorted(yaml_files):
        fname = os.path.splitext(os.path.basename(fpath))[0]
        tickers = set()
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is not None:
                _extract(data, tickers)
            if tickers:
                groups[fname] = sorted(list(tickers))
                log.info(f"Group '{fname}': Loaded {len(tickers)} tickers.")
        except Exception as e:
            log.warning(f"Could not parse {os.path.basename(fpath)}: {e}")

    return groups if groups else {"Default": list(_DEFAULT_INSTRUMENTS)}


# ── Data Helpers ───────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period: str = "2y") -> tuple:
    """
    Returns (df, full_name, market_cap_raw, ipo_year).
    - market_cap_raw : raw USD integer from yfinance; None for ETFs
    - ipo_year       : int year the instrument first traded (from firstTradeDateEpochUtc
                       or firstTradeDateMilliseconds); "N/A" if unavailable
    """
    try:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        name = info.get('longName', ticker)
        mktcap = info.get('marketCap', None)  # raw USD integer; None for ETFs

        # IPO / first trade date — yfinance exposes this via firstTradeDateEpochUtc (seconds)
        # or firstTradeDateMilliseconds. We try both and fall back to "N/A" gracefully.
        ipo_year = "N/A"
        epoch_s = info.get('firstTradeDateEpochUtc', None)
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
            return f"${v / 1e12:.2f}T"
        elif v >= 1e9:
            return f"${v / 1e9:.2f}B"
        elif v >= 1e6:
            return f"${v / 1e6:.2f}M"
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
        prev = float(combined['s'].iloc[-i - 1]) - float(combined['l'].iloc[-i - 1])
        curr = float(combined['s'].iloc[-i]) - float(combined['l'].iloc[-i])
        if prev < 0 and curr >= 0:
            return "Golden", combined.index[-i].date()
        if prev > 0 and curr <= 0:
            return "Death", combined.index[-i].date()
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
        prev = int(direction.iloc[-i - 1])
        curr = int(direction.iloc[-i])
        if prev == -1 and curr == 1:
            return "Bullish", direction.index[-i].date()
        if prev == 1 and curr == -1:
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
    e20 = get_ema(df, EMA_FAST)
    e50 = get_ema(df, EMA_MID)
    e200 = get_ema(df, EMA_SLOW)

    if e20 is None or e50 is None or e200 is None:
        return None, None, None, None, None

    frame = pd.DataFrame({
        'Close': df['Close'],
        'e20': e20,
        'e50': e50,
        'e200': e200,
    }).dropna()

    if len(frame) < 2:
        return None, None, None, None, None

    last = frame.iloc[-1]
    v20 = round(float(last['e20']), 2)
    v50 = round(float(last['e50']), 2)
    v200 = round(float(last['e200']), 2)

    def is_bull(r):
        return r['Close'] > r['e20'] > r['e50'] > r['e200']

    def is_bear(r):
        return r['Close'] < r['e20'] < r['e50'] < r['e200']

    if is_bull(last):
        trend_type = "Bullish"
    elif is_bear(last):
        trend_type = "Bearish"
    else:
        return None, None, v20, v50, v200

    check = is_bull if trend_type == "Bullish" else is_bear
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

    df_w = fetch_weekly_data(ticker)
    if df_w is None or len(df_w) < RSI_PERIOD + 1:
        return None
    w_rsi = get_rsi(df_w)
    if w_rsi is None or w_rsi <= 60:
        return None

    df_m = fetch_monthly_data(ticker)
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
            'Close': df_w['Close'],
            'sma200': w_sma200_series,
        }).dropna()

        if frame.empty:
            return None

        w_sma200_val = round(float(frame['sma200'].iloc[-1]), 2)
        upper_bound = w_sma200_val * (1 + W200_UPPER_BUFFER)

        # Check CURRENT bar qualifies: price in [SMA200, SMA200 * 1.10]
        if not (last_price >= w_sma200_val and last_price <= upper_bound):
            return None

        pct_above = round(((last_price - w_sma200_val) / w_sma200_val) * 100, 2)

        # Find the EARLIEST EVER date in the full history where the zone condition was true.
        first_ever_date = frame.index[-1].date()  # fallback: today's bar
        for i in range(len(frame)):
            row_close = float(frame['Close'].iloc[i])
            row_sma200 = float(frame['sma200'].iloc[i])
            row_upper = row_sma200 * (1 + W200_UPPER_BUFFER)
            if row_close >= row_sma200 and row_close <= row_upper:
                first_ever_date = frame.index[i].date()
                break  # found the earliest qualifying bar — stop

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


# ── HTML/CSS/JS Assets (PRESERVING ORIGINAL FORMAT) ───────────────────────────

CSS = """<style>
  :root {
    --bull:#27ae60; --bull-lt:#eafaf1;
    --bear:#c0392b; --bear-lt:#fdedec;
    --comb:#b7770d; --comb-lt:#fef9e7;
    --fib1:#1a6b3c; --fib1-lt:#e9f7ef;
    --fib2:#2471a3; --fib2-lt:#eaf3fb;
    --fib3:#7d3c98; --fib3-lt:#f5eef8;
    --w200:#0e6655; --w200-lt:#e8f8f5;
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
          text-align:left; white-space:nowrap; cursor:pointer; user-select:none; }
  th:hover { background:#3d5166; }
  th.sort-asc::after  { content:' \\25B2'; font-size:10px; }
  th.sort-desc::after { content:' \\25BC'; font-size:10px; }
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
</style>"""

JS_SORTING = """<script>
function tableSort(tableId, colIdx) {
  const table = document.getElementById(tableId);
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const th = table.tHead.rows[0].cells[colIdx];
  const isAsc = th.classList.contains('sort-asc');

  table.querySelectorAll('th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
  th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');

  rows.sort((a, b) => {
    let valA = a.cells[colIdx].innerText.replace(/[$,%]/g, '');
    let valB = b.cells[colIdx].innerText.replace(/[$,%]/g, '');
    let floatA = parseFloat(valA);
    let floatB = parseFloat(valB);

    let comparison = 0;
    if (!isNaN(floatA) && !isNaN(floatB)) {
        comparison = floatA - floatB;
    } else {
        comparison = valA.localeCompare(valB);
    }
    return isAsc ? -comparison : comparison;
  });

  rows.forEach(row => tbody.appendChild(row));
}
</script>"""


# ── Report Builder ─────────────────────────────────────────────────────────────

def get_table_html(data_list, table_id, row_class=""):
    if not data_list:
        return '<div class="no-sig">No signals detected.</div>'

    headers = [k for k in data_list[0].keys() if not k.startswith('_')]
    html = f'<table id="{table_id}"><thead><tr>'
    for i, h in enumerate(headers):
        html += f'<th onclick="tableSort(\'{table_id}\', {i})">{h}</th>'
    html += '</tr></thead><tbody>'

    for row in data_list:
        html += f'<tr class="{row_class}">'
        for k, v in row.items():
            if k.startswith('_'): continue

            td_style = ""
            if k == "ADX (14)":
                try:
                    val = float(v)
                    if val < 18:
                        td_style = 'adx-weak'
                    elif 18 <= val < 25:
                        td_style = 'adx-neutral'
                    elif 25 <= val < 45:
                        td_style = 'adx-strong'
                    else:
                        td_style = 'adx-exhaust'
                except:
                    pass

            html += f'<td class="{td_style}">{v}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


# ── Scanner Logic ──────────────────────────────────────────────────────────────

def run_scanner_for_group(group_name, ticker_list):
    results = {
        "w200_zone": [],
        "combined": [],
        "bullish_cross": [],
        "bearish_cross": [],
        "bullish_st": [],
        "bearish_st": [],
        "bullish_trend": [],
        "bearish_trend": [],
        "pullback": [],
        "fib_382_500": [],
        "fib_500_618": [],
        "fib_618_786": [],
    }

    log.info(f"Processing group '{group_name}' ({len(ticker_list)} tickers)...")

    for ticker in ticker_list:
        df, full_name, mktcap, ipo_year = fetch_data(ticker)
        if df is None or len(df) < SMA_LONG:
            continue

        last_price = float(df['Close'].iloc[-1])
        hi_52 = float(df['High'].iloc[-252:].max())
        lo_52 = float(df['Low'].iloc[-252:].min())
        adx_val = get_adx(df)
        mktcap_fmt = format_market_cap(mktcap)
        df_weekly_cached = fetch_weekly_data(ticker)

        row_base = {
            "Instrument": f"{full_name} ({ticker})",
            "Last Price": round(last_price, 2),
            "52W High": round(hi_52, 2),
            "52W Low": round(lo_52, 2),
            "ADX (14)": adx_val if adx_val is not None else "N/A",
            "Market Cap": mktcap_fmt,
            "IPO Year": ipo_year,
        }

        # Filters
        w200_result = detect_weekly_sma200_zone(ticker, last_price, df_weekly=df_weekly_cached)
        if w200_result:
            results["w200_zone"].append(
                {**row_base, "Weekly SMA200": w200_result[0], "% Above": w200_result[1], "First Date": w200_result[2]})

        c_type, c_date = detect_crossover(df)
        if c_type:
            results["bullish_cross" if c_type == "Golden" else "bearish_cross"].append(
                {**row_base, "Type": c_type, "Date": c_date})

        st_type, st_date = detect_supertrend_change(df)
        if st_type:
            results["bullish_st" if st_type == "Bullish" else "bearish_st"].append(
                {**row_base, "Type": st_type, "Date": st_date})

        t_type, t_start, _, _, _ = detect_trend(df)
        if t_type:
            results["bullish_trend" if t_type == "Bullish" else "bearish_trend"].append(
                {**row_base, "Type": t_type, "Start": t_start})

        pb = detect_pullback(ticker, df)
        if pb:
            results["pullback"].append({**row_base, "Daily RSI": pb[0], "Weekly RSI": pb[1], "Monthly RSI": pb[2]})

        fib = detect_fib_zone(last_price, hi_52, lo_52)
        if fib:
            if fib[0] == "0.382 - 0.500":
                results["fib_382_500"].append({**row_base, "Fib Level": fib[1]})
            elif fib[0] == "0.500 - 0.618":
                results["fib_500_618"].append({**row_base, "Fib Level": fib[1]})
            elif fib[0] == "0.618 - 0.786":
                results["fib_618_786"].append({**row_base, "Fib Level": fib[1]})

    # Generate the actual HTML content
    report_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">{CSS}</head>
    <body>
    <h2>Market Scanner: {group_name.replace('_', ' ').upper()}</h2>
    <div class="adx-legend">
        <strong>ADX Legend:</strong>
        <span><i class="adx-legend-dot" style="background:#95a5a6"></i> Weak (< 18)</span>
        <span><i class="adx-legend-dot" style="background:#2471a3"></i> Developing (18-25)</span>
        <span><i class="adx-legend-dot" style="background:#27ae60"></i> Strong (25-45)</span>
        <span><i class="adx-legend-dot" style="background:#c0392b"></i> Exhaustion (> 45)</span>
    </div>

    <div class="section">
        <div class="sec-title">Weekly SMA 200 Zone Filter</div>
        {get_table_html(results['w200_zone'], 'w200-tbl', 'r-w200')}
    </div>

    <div class="section">
        <div class="sec-title">SMA Crossovers</div>
        <div class="pair">
            <div class="sub">
                <div class="sub-hdr hdr-bull">Bullish (Golden Cross)</div>
                {get_table_html(results['bullish_cross'], 'c-bull-tbl', 'r-bull')}
            </div>
            <div class="sub">
                <div class="sub-hdr hdr-bear">Bearish (Death Cross)</div>
                {get_table_html(results['bearish_cross'], 'c-bear-tbl', 'r-bear')}
            </div>
        </div>
    </div>

    <div class="section">
        <div class="sec-title">EMA Trend Alignment (20/50/200)</div>
        <div class="pair">
            <div class="sub">
                <div class="sub-hdr hdr-bull">Bullish Trend</div>
                {get_table_html(results['bullish_trend'], 't-bull-tbl', 'r-bull')}
            </div>
            <div class="sub">
                <div class="sub-hdr hdr-bear">Bearish Trend</div>
                {get_table_html(results['bearish_trend'], 't-bear-tbl', 'r-bear')}
            </div>
        </div>
    </div>

    <div class="section">
        <div class="sec-title">RSI Pullbacks & Fibonacci Zones</div>
        <div class="pair">
            <div class="sub">
                <div class="sub-hdr hdr-comb">Pullback Signals</div>
                {get_table_html(results['pullback'], 'pb-tbl', 'r-comb')}
            </div>
            <div class="sub">
                <div class="sub-hdr hdr-fib2">Fib 50.0% - 61.8%</div>
                {get_table_html(results['fib_500_618'], 'fib-mid-tbl', 'r-fib2')}
            </div>
        </div>
    </div>

    {JS_SORTING}
    </body></html>"""

    filename = f"Scanner_{group_name}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_html)
    log.info(f"Report '{filename}' generated.")


if __name__ == "__main__":
    ticker_groups = load_instruments_grouped()
    for name, tickers in ticker_groups.items():
        run_scanner_for_group(name, tickers)
