import logging
import pandas as pd
import numpy as np
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

USER_INSTRUMENTS = [
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

# ── Data Helpers ───────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period: str = "2y") -> tuple:
    try:
        t_obj = yf.Ticker(ticker)
        name  = t_obj.info.get('longName', ticker)
        df    = t_obj.history(period=period, auto_adjust=True)
        if df.empty:
            return None, name
        return df, name
    except Exception:
        return None, ticker


def fetch_weekly_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period="5y", interval="1wk", auto_adjust=True)
        return df if not df.empty else None
    except Exception:
        return None


def fetch_monthly_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period="10y", interval="1mo", auto_adjust=True)
        return df if not df.empty else None
    except Exception:
        return None


# ── Indicators ─────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Crossover Logic (PRESERVED) ────────────────────────────────────────────────

def detect_crossover(df: pd.DataFrame):
    df = df.copy()
    df[f"SMA{SMA_SHORT}"] = df["Close"].rolling(SMA_SHORT).mean()
    df[f"SMA{SMA_LONG}"]  = df["Close"].rolling(SMA_LONG).mean()
    df = df.dropna(subset=[f"SMA{SMA_SHORT}", f"SMA{SMA_LONG}"])
    if len(df) < 2:
        return None, None
    short, long_ = df[f"SMA{SMA_SHORT}"], df[f"SMA{SMA_LONG}"]

    lookback = min(LOOKBACK_WINDOW, len(df) - 1)
    for i in range(1, lookback + 1):
        prev_diff = float(short.iloc[-i - 1]) - float(long_.iloc[-i - 1])
        curr_diff = float(short.iloc[-i])     - float(long_.iloc[-i])
        if prev_diff < 0 and curr_diff >= 0:
            return "Golden", df.index[-i].date()
        if prev_diff > 0 and curr_diff <= 0:
            return "Death",  df.index[-i].date()
    return None, None


# ── Supertrend Logic (PRESERVED) ───────────────────────────────────────────────

def detect_supertrend_change(df, period=10, multiplier=3):
    df   = df.copy()
    high, low, close = df['High'], df['Low'], df['Close']
    tr   = pd.concat([high - low,
                      abs(high - close.shift(1)),
                      abs(low  - close.shift(1))], axis=1).max(axis=1)
    atr  = tr.rolling(period).mean()
    hl2  = (high + low) / 2
    upper_band  = hl2 + (multiplier * atr)
    lower_band  = hl2 - (multiplier * atr)
    final_upper = [0.0] * len(df)
    final_lower = [0.0] * len(df)
    direction   = [1]   * len(df)

    for i in range(1, len(df)):
        final_upper[i] = (upper_band.iloc[i]
                          if upper_band.iloc[i] < final_upper[i-1]
                          or close.iloc[i-1] > final_upper[i-1]
                          else final_upper[i-1])
        final_lower[i] = (lower_band.iloc[i]
                          if lower_band.iloc[i] > final_lower[i-1]
                          or close.iloc[i-1] < final_lower[i-1]
                          else final_lower[i-1])
        direction[i] = (1  if close.iloc[i] > final_upper[i] else
                        -1 if close.iloc[i] < final_lower[i] else
                        direction[i-1])

    lookback = min(LOOKBACK_WINDOW, len(df) - 1)
    for i in range(1, lookback + 1):
        if direction[-i-1] == -1 and direction[-i] == 1:
            return "Bullish", df.index[-i].date()
        if direction[-i-1] ==  1 and direction[-i] == -1:
            return "Bearish", df.index[-i].date()
    return None, None


# ── Up/Down Trend Logic ─────────────────────────────────────────────────────────

def detect_trend(df: pd.DataFrame):
    """
    Bullish : Close > EMA20 > EMA50 > EMA200
    Bearish : Close < EMA20 < EMA50 < EMA200
    Returns (type, start_date, ema20, ema50, ema200)
    type is None when no clear trend alignment detected.
    """
    df = df.copy()
    df['EMA20']  = df['Close'].ewm(span=20,  adjust=False).mean()
    df['EMA50']  = df['Close'].ewm(span=50,  adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df = df.dropna(subset=['EMA20', 'EMA50', 'EMA200'])
    if len(df) < 2:
        return None, None, None, None, None

    last   = df.iloc[-1]
    ema20  = round(float(last['EMA20']),  2)
    ema50  = round(float(last['EMA50']),  2)
    ema200 = round(float(last['EMA200']), 2)

    def is_bullish(row):
        return row['Close'] > row['EMA20'] > row['EMA50'] > row['EMA200']

    def is_bearish(row):
        return row['Close'] < row['EMA20'] < row['EMA50'] < row['EMA200']

    if is_bullish(last):
        trend_type = "Bullish"
    elif is_bearish(last):
        trend_type = "Bearish"
    else:
        return None, None, ema20, ema50, ema200

    # Walk backwards to find when this trend started
    start_date = df.index[-1].date()
    check_fn   = is_bullish if trend_type == "Bullish" else is_bearish
    for i in range(2, len(df) + 1):
        if check_fn(df.iloc[-i]):
            start_date = df.index[-i].date()
        else:
            break

    return trend_type, start_date, ema20, ema50, ema200


# ── Pullback Logic ──────────────────────────────────────────────────────────────

def detect_pullback(ticker: str, df_daily: pd.DataFrame):
    """Monthly RSI > 60 AND Weekly RSI > 60 AND Daily RSI < 45"""
    if df_daily is None or len(df_daily) < 15:
        return None
    daily_rsi = float(compute_rsi(df_daily['Close'], 14).iloc[-1])
    if pd.isna(daily_rsi) or daily_rsi >= 45:
        return None

    df_w = fetch_weekly_data(ticker)
    if df_w is None or len(df_w) < 15:
        return None
    weekly_rsi = float(compute_rsi(df_w['Close'], 14).iloc[-1])
    if pd.isna(weekly_rsi) or weekly_rsi <= 60:
        return None

    df_m = fetch_monthly_data(ticker)
    if df_m is None or len(df_m) < 15:
        return None
    monthly_rsi = float(compute_rsi(df_m['Close'], 14).iloc[-1])
    if pd.isna(monthly_rsi) or monthly_rsi <= 60:
        return None

    return round(daily_rsi, 2), round(weekly_rsi, 2), round(monthly_rsi, 2)


# ── Fibonacci Retracement Logic ───────────────────────────────────────────────

# Retracement zones (measured from swing HIGH down to swing LOW)
# Level 0 = 52W High (swing high), Level 1 = 52W Low (swing low)
# Retracement % = (High - Close) / (High - Low)
# Zone boundaries:
FIB_ZONES = [
    ("0.382 – 0.500", 0.382, 0.500, "#1a6b3c"),   # deep green
    ("0.500 – 0.618", 0.500, 0.618, "#2471a3"),   # blue
    ("0.618 – 0.786", 0.618, 0.786, "#7d3c98"),   # purple
]

def detect_fib_zone(last_price: float, hi_52: float, lo_52: float):
    """
    Returns (zone_label, fib_level, zone_color) if price sits inside one of the
    three defined retracement zones, else None.

    Convention:
        Level 0 = 52W High (top / swing high)
        Level 1 = 52W Low  (bottom / swing low)
        Retracement = (High - Close) / (High - Low)
        0.382 retracement price = High - 0.382 * (High - Low)
    """
    rng = hi_52 - lo_52
    if rng <= 0:
        return None
    retrace = (hi_52 - last_price) / rng          # 0 at high, 1 at low
    retrace = round(retrace, 4)
    for label, lo_bound, hi_bound, color in FIB_ZONES:
        if lo_bound <= retrace < hi_bound:
            return label, round(retrace, 4), color
    return None


# ── Main Scanner ────────────────────────────────────────────────────────────────

def run_scanner():
    results = {
        "combined":      [],   # Fib zone AND RSI pullback simultaneously
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


    for ticker in USER_INSTRUMENTS:
        log.info(f"Scanning {ticker}...")
        df, full_name = fetch_data(ticker)
        if df is None or len(df) < SMA_LONG:
            log.warning(f"  Skipping {ticker} – insufficient data")
            continue

        last_price = float(df['Close'].iloc[-1])
        hi_52      = float(df['High'].iloc[-252:].max())
        lo_52      = float(df['Low'].iloc[-252:].min())
        label      = f"{full_name} ({ticker})"

        row_base = {
            "Instrument": label,
            "Last Price": round(last_price, 2),
            "52W High":   round(hi_52, 2),
            "52W Low":    round(lo_52, 2),
        }

        # ── 1. Crossover ──────────────────────────────────────────────────────
        c_type, c_date = detect_crossover(df)
        if c_type:
            row = {**row_base, "Crossover Type": c_type, "Crossover Date": c_date}
            if c_type == "Golden":
                results["bullish_cross"].append(row)
            else:
                results["bearish_cross"].append(row)

        # ── 2. Supertrend (PRESERVED) ─────────────────────────────────────────
        st_type, st_date = detect_supertrend_change(df)
        if st_type:
            row = {**row_base, "ST Type": st_type, "ST Date": st_date}
            results["bullish_st" if st_type == "Bullish" else "bearish_st"].append(row)

        # ── 3. Trend ──────────────────────────────────────────────────────────
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
                "Trend Start Date": trend_start,
                "Trend Type":       trend_type,
            }
            key = "bullish_trend" if trend_type == "Bullish" else "bearish_trend"
            results[key].append(row)

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
                "52W High":    round(hi_52, 2),
                "52W Low":     round(lo_52, 2),
            }
            results["pullback"].append(row)

        # ── 5. Fibonacci Retracement Zone ────────────────────────────────────
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
            }
            if   fib_label == "0.382 – 0.500":  results["fib_382_500"].append(row)
            elif fib_label == "0.500 – 0.618":  results["fib_500_618"].append(row)
            elif fib_label == "0.618 – 0.786":  results["fib_618_786"].append(row)

        # ── 6. Combined : Fib Zone AND RSI Pullback ───────────────────────────
        if pb and fib:
            d_rsi, w_rsi, m_rsi = pb
            fib_label, fib_level, _ = fib
            # Compute raw daily RSI for "date" sorting proxy (lower = more oversold = more recent entry)
            row = {
                "Instrument":       label,
                "Last Price":       round(last_price, 2),
                "Fib Zone":         fib_label,
                "Fib Level":        fib_level,
                "Daily RSI":        d_rsi,
                "Weekly RSI":       w_rsi,
                "Monthly RSI":      m_rsi,
                "52W High":         round(hi_52, 2),
                "52W Low":          round(lo_52, 2),
            }
            results["combined"].append(row)

    save_html_report(results)


# ── HTML Helpers ───────────────────────────────────────────────────────────────

CSS = """<style>
  :root {
    --bull:#27ae60; --bull-lt:#eafaf1;
    --bear:#c0392b; --bear-lt:#fdedec;
    --comb:#b7770d;  --comb-lt:#fef9e7;
    --fib1:#1a6b3c;  --fib1-lt:#e9f7ef;
    --fib2:#2471a3;  --fib2-lt:#eaf3fb;
    --fib3:#7d3c98;  --fib3-lt:#f5eef8;
    --head:#2c3e50;
  }
  * { box-sizing:border-box; }
  body { font-family:'Segoe UI',Arial,sans-serif; margin:0; padding:28px;
         background:#ecf0f1; color:#333; }
  h2   { color:var(--head); border-bottom:3px solid var(--head);
         padding-bottom:10px; margin-bottom:28px; }

  /* ── Section card ── */
  .section { background:white; border-radius:10px; padding:22px;
             margin-bottom:38px; box-shadow:0 2px 8px rgba(0,0,0,0.09); }
  .sec-title { font-size:16px; font-weight:700; color:var(--head);
               border-bottom:2px solid var(--head);
               padding-bottom:8px; margin-bottom:18px; }

  /* ── Side-by-side layout ── */
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

  /* ── Tables ── */
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th    { background:var(--head); color:white; padding:8px 7px; text-align:left; }
  td    { padding:7px; border-bottom:1px solid #e5e5e5; }
  .r-bull td { background:var(--bull-lt); }
  .r-bear td { background:var(--bear-lt); }
  .r-comb td { background:var(--comb-lt); }
  .r-fib1 td { background:var(--fib1-lt); }
  .r-fib2 td { background:var(--fib2-lt); }
  .r-fib3 td { background:var(--fib3-lt); }
  tr:hover td { filter:brightness(0.95); }

  /* ── Badges ── */
  .badge { display:inline-block; padding:2px 9px; border-radius:12px;
           font-size:11px; font-weight:700; color:white; }
  .b-bull  { background:var(--bull); }
  .b-bear  { background:var(--bear); }
  .b-gold  { background:#f39c12; }
  .b-death { background:#8e44ad; }

  .no-sig { color:#aaa; font-style:italic; padding:10px 4px; font-size:12.5px; }
</style>"""


def badge(val):
    s = str(val)
    if s == "Bullish": return f"<span class='badge b-bull'>Bullish ▲</span>"
    if s == "Bearish": return f"<span class='badge b-bear'>Bearish ▼</span>"
    if s == "Golden":  return f"<span class='badge b-gold'>Golden ✦</span>"
    if s == "Death":   return f"<span class='badge b-death'>Death ✖</span>"
    return s


def rows_to_table(rows: list, sort_col: str = None, row_cls: str = "",
                  ascending: bool = False) -> str:
    if not rows:
        return "<p class='no-sig'>No signals detected.</p>"
    df = pd.DataFrame(rows)
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending)
    hdrs = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        cells = "".join(f"<td>{badge(v)}</td>" for v in r.values)
        body += f"<tr class='{row_cls}'>{cells}</tr>"
    return f"<table><thead><tr>{hdrs}</tr></thead><tbody>{body}</tbody></table>"


def sub(hdr_cls: str, title: str, content: str) -> str:
    return (f"<div class='sub'>"
            f"<div class='sub-hdr {hdr_cls}'>{title}</div>"
            f"{content}"
            f"</div>")


# ── HTML Report ────────────────────────────────────────────────────────────────

def save_html_report(results: dict):
    est     = pytz.timezone("America/New_York")
    now_str = datetime.now(est).strftime("%Y-%m-%d %H:%M %Z")
    secs    = []

    # ── Section 1 : Combined — Fibonacci Zone AND RSI Pullback ───────────────
    # Sorted ascending by Daily RSI (lowest RSI = most oversold = most recently
    # confirmed pullback entry) so the freshest signals appear at the top.
    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>🎯 Section 1 &mdash; High-Conviction Pullback Entries
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(Fibonacci Zone &amp; RSI Pullback simultaneously &nbsp;|&nbsp; sorted by lowest Daily RSI first)
        </span>
      </div>
      <div class='pair'>
        {sub("hdr-comb",
             "⭐ Instruments in Fibonacci Retracement Zone WITH Daily RSI &lt; 45 "
             "(Monthly RSI &gt;60 &amp; Weekly RSI &gt;60)",
             rows_to_table(results["combined"], "Daily RSI", "r-comb", ascending=True))}
      </div>
    </div>""")

    # ── Section 2 : Fibonacci Retracement Zones ───────────────────────────────
    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>📐 Section 2 &mdash; Fibonacci Retracement Zones
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(52W High = Level 0 &nbsp;|&nbsp; 52W Low = Level 1 &nbsp;|&nbsp; sorted by most recently entered zone)
        </span>
      </div>
      <div class='pair'>
        {sub("hdr-fib1", "🟢 Zone 0.382 – 0.500 &nbsp;<span style='font-weight:400;font-size:11px'>(shallow retracement — strong trend)</span>",
             rows_to_table(results["fib_382_500"], "Fib Level", "r-fib1", ascending=True))}
        {sub("hdr-fib2", "🔵 Zone 0.500 – 0.618 &nbsp;<span style='font-weight:400;font-size:11px'>(moderate retracement — key support area)</span>",
             rows_to_table(results["fib_500_618"], "Fib Level", "r-fib2", ascending=True))}
      </div>
      <div class='pair' style='margin-top:18px'>
        {sub("hdr-fib3", "🟣 Zone 0.618 – 0.786 &nbsp;<span style='font-weight:400;font-size:11px'>(deep retracement — last major support before trend reversal)</span>",
             rows_to_table(results["fib_618_786"], "Fib Level", "r-fib3", ascending=True))}
      </div>
    </div>""")

    # ── Section 3 : Possible Pullback (RSI) ───────────────────────────────────
    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>🔄 Section 3 &mdash; Possible Pullback
        <span style='font-weight:400;font-size:13px'>
          &nbsp;(Monthly RSI &gt;60 &amp; Weekly RSI &gt;60 &amp; Daily RSI &lt;45)
        </span>
      </div>
      <div class='pair'>
        {sub("hdr-bull", "✅ Pullback Candidates",
             rows_to_table(results["pullback"], "Daily RSI", "r-bull", ascending=True))}
      </div>
    </div>""")

    # ── Section 4 : Crossover ──────────────────────────────────────────────────
    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>📈 Section 4 &mdash; Crossover Signals</div>
      <div class='pair'>
        {sub("hdr-bull", "✅ Bullish — Golden Crossover (SMA 50 crosses above SMA 200)",
             rows_to_table(results["bullish_cross"], "Crossover Date", "r-bull"))}
        {sub("hdr-bear", "❌ Bearish — Death Crossover",
             rows_to_table(results["bearish_cross"], "Crossover Date", "r-bear"))}
      </div>
    </div>""")

    # ── Section 5 : Supertrend ─────────────────────────────────────────────────
    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>⚡ Section 5 &mdash; Supertrend Signals (10,3)</div>
      <div class='pair'>
        {sub("hdr-bull", "✅ Bullish Supertrend",
             rows_to_table(results["bullish_st"], "ST Date", "r-bull"))}
        {sub("hdr-bear", "❌ Bearish Supertrend",
             rows_to_table(results["bearish_st"], "ST Date", "r-bear"))}
      </div>
    </div>""")

    # ── Section 6 : Up/Down Trend ──────────────────────────────────────────────
    secs.append(f"""
    <div class='section'>
      <div class='sec-title'>📊 Section 6 &mdash; Up / Down Trend (EMA 20 &rsaquo; 50 &rsaquo; 200)</div>
      <div class='pair'>
        {sub("hdr-bull",
             "✅ Bullish — Close &gt; EMA20 &gt; EMA50 &gt; EMA200",
             rows_to_table(results["bullish_trend"], "Trend Start Date", "r-bull"))}
        {sub("hdr-bear",
             "❌ Bearish — Close &lt; EMA20 &lt; EMA50 &lt; EMA200",
             rows_to_table(results["bearish_trend"], "Trend Start Date", "r-bear"))}
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
