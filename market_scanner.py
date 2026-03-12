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
import argparse

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
W200_SMA_PERIOD = 200
W200_UPPER_BUFFER = 0.10  # 10% ceiling above weekly SMA200

# ── Default Instruments ────────────────────────────────────────────────────────
_DEFAULT_INSTRUMENTS = [
    'USO', 'UNG', 'XLE', 'XOP', 'GLD', 'SLV', 'IBIT',
    'QQQ', 'SPY', 'IWM', 'TLT', 'MAGS', 'URTH',
    'MCHI', 'EWY', 'EWJ', 'EWZ', 'VWO', 'TUR', 'EIS',
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


# ── YAML Ticker Loader ────────────────────────────────────────────────────────

def load_instruments_grouped(target_file=None) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_dir = os.path.normpath(os.path.join(script_dir, ".yaml-tickers"))
    yaml_files = []
    if os.path.isdir(yaml_dir):
        if target_file:
            if not target_file.endswith(('.yaml', '.yml')):
                potential_files = [os.path.join(yaml_dir, f"{target_file}.yaml"),
                                   os.path.join(yaml_dir, f"{target_file}.yml")]
            else:
                potential_files = [os.path.join(yaml_dir, target_file)]
            yaml_files = [f for f in potential_files if os.path.exists(f)]
            if not yaml_files:
                log.error(f"Target file '{target_file}' not found in {yaml_dir}")
                return {}
        else:
            for file in os.listdir(yaml_dir):
                if file.lower().endswith((".yaml", ".yml")):
                    yaml_files.append(os.path.join(yaml_dir, file))
    if not yaml_files and not target_file:
        log.info("No YAML files found — using default list.")
        return {"Default": list(_DEFAULT_INSTRUMENTS)}
    groups = {}

    def _extract(node, ticker_set):
        if isinstance(node, str):
            stripped = node.strip().upper()
            if stripped: ticker_set.add(stripped)
        elif isinstance(node, dict):
            if "ticker" in node:
                _extract(node["ticker"], ticker_set)
            else:
                for v in node.values(): _extract(v, ticker_set)
        elif isinstance(node, list):
            for item in node: _extract(item, ticker_set)

    for fpath in sorted(yaml_files):
        base_name = os.path.basename(fpath)
        fname = base_name.rsplit('.', 1)[0]
        tickers = set()
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data: _extract(data, tickers)
            if tickers:
                groups[fname] = sorted(list(tickers))
                log.info(f"Group '{fname}': Loaded {len(tickers)} tickers.")
        except Exception as e:
            log.warning(f"Could not parse {base_name}: {e}")
    return groups


# ── Data Helpers ───────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period: str = "2y") -> tuple:
    try:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        name = info.get('longName', ticker)
        mktcap = info.get('marketCap', None)
        quote_type = info.get('quoteType', 'EQUITY')  # Added to detect ETFs
        ipo_year = "N/A"
        epoch_s = info.get('firstTradeDateEpochUtc', None)
        if epoch_s is None:
            epoch_ms = info.get('firstTradeDateMilliseconds', None)
            if epoch_ms is not None: epoch_s = epoch_ms / 1000
        if epoch_s is not None:
            try:
                ipo_year = datetime.utcfromtimestamp(float(epoch_s)).year
            except Exception:
                ipo_year = "N/A"
        df = t_obj.history(period=period, auto_adjust=True)
        if df.empty: return None, name, mktcap, ipo_year, quote_type
        df.index = pd.DatetimeIndex(df.index)
        return df, name, mktcap, ipo_year, quote_type
    except Exception:
        return None, ticker, None, "N/A", 'EQUITY'


def fetch_weekly_data(ticker: str) -> pd.DataFrame:
    for period in ("max", "10y", "5y"):
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1wk", auto_adjust=True)
            if df.empty: continue
            df.index = pd.DatetimeIndex(df.index)
            if len(df) >= W200_SMA_PERIOD + 50: return df
        except Exception:
            continue
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
        if df.empty: return None
        df.index = pd.DatetimeIndex(df.index)
        return df
    except Exception:
        return None


def format_market_cap(mktcap) -> str:
    if mktcap is None: return "N/A"
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
    try:
        result = ta.rsi(df['Close'], length=period)
        if result is None: return None
        val = result.dropna()
        return round(float(val.iloc[-1]), 2) if not val.empty else None
    except Exception:
        return None


def get_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    try:
        result = ta.adx(df['High'], df['Low'], df['Close'], length=period)
        if result is None: return None
        col = f"ADX_{period}"
        val = result[col].dropna()
        return round(float(val.iloc[-1]), 2) if not val.empty else None
    except Exception:
        return None


def get_sma(df: pd.DataFrame, period: int) -> pd.Series:
    try:
        return ta.sma(df['Close'], length=period)
    except Exception:
        return None


def get_ema(df: pd.DataFrame, period: int) -> pd.Series:
    try:
        return ta.ema(df['Close'], length=period)
    except Exception:
        return None


def get_supertrend_direction(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    try:
        result = ta.supertrend(df['High'], df['Low'], df['Close'], length=period, multiplier=multiplier)
        if result is None: return None
        dir_cols = [c for c in result.columns if c.startswith('SUPERTd')]
        return result[dir_cols[0]] if dir_cols else None
    except Exception:
        return None


# ── Signal Detection ───────────────────────────────────────────────────────────

def detect_crossover(df: pd.DataFrame):
    sma_s = get_sma(df, SMA_SHORT)
    sma_l = get_sma(df, SMA_LONG)
    if sma_s is None or sma_l is None: return None, None
    combined = pd.DataFrame({'s': sma_s, 'l': sma_l}).dropna()
    lookback = min(LOOKBACK_WINDOW, len(combined) - 1)
    for i in range(1, lookback + 1):
        prev = float(combined['s'].iloc[-i - 1]) - float(combined['l'].iloc[-i - 1])
        curr = float(combined['s'].iloc[-i]) - float(combined['l'].iloc[-i])
        if prev < 0 and curr >= 0: return "Golden", combined.index[-i].date()
        if prev > 0 and curr <= 0: return "Death", combined.index[-i].date()
    return None, None


def detect_supertrend_change(df: pd.DataFrame):
    direction = get_supertrend_direction(df)
    if direction is None: return None, None
    direction = direction.dropna()
    lookback = min(LOOKBACK_WINDOW, len(direction) - 1)
    for i in range(1, lookback + 1):
        prev = int(direction.iloc[-i - 1])
        curr = int(direction.iloc[-i])
        if prev == -1 and curr == 1: return "Bullish", direction.index[-i].date()
        if prev == 1 and curr == -1: return "Bearish", direction.index[-i].date()
    return None, None


def detect_trend(df: pd.DataFrame):
    e20 = get_ema(df, EMA_FAST);
    e50 = get_ema(df, EMA_MID);
    e200 = get_ema(df, EMA_SLOW)
    if e20 is None or e50 is None or e200 is None: return None, None, v20, v50, v200
    frame = pd.DataFrame({'Close': df['Close'], 'e20': e20, 'e50': e50, 'e200': e200}).dropna()
    if frame.empty: return None, None, None, None, None
    last = frame.iloc[-1];
    v20 = round(float(last['e20']), 2);
    v50 = round(float(last['e50']), 2);
    v200 = round(float(last['e200']), 2)
    is_bull = lambda r: r['Close'] > r['e20'] > r['e50'] > r['e200']
    is_bear = lambda r: r['Close'] < r['e20'] < r['e50'] < r['e200']
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
    if df_daily is None or len(df_daily) < RSI_PERIOD + 1: return None
    d_rsi = get_rsi(df_daily)
    if d_rsi is None or d_rsi >= 45: return None
    df_w = fetch_weekly_data(ticker)
    if df_w is None or len(df_w) < RSI_PERIOD + 1: return None
    w_rsi = get_rsi(df_w)
    if w_rsi is None or w_rsi <= 60: return None
    df_m = fetch_monthly_data(ticker)
    if df_m is None or len(df_m) < RSI_PERIOD + 1: return None
    m_rsi = get_rsi(df_m)
    if m_rsi is None or m_rsi <= 60: return None
    return round(d_rsi, 2), round(w_rsi, 2), round(m_rsi, 2)


def detect_weekly_sma200_zone(ticker: str, last_price: float, df_weekly: pd.DataFrame = None):
    try:
        df_w = df_weekly if df_weekly is not None else fetch_weekly_data(ticker)
        if df_w is None or len(df_w) < W200_SMA_PERIOD: return None
        w_sma200_series = get_sma(df_w, W200_SMA_PERIOD)
        frame = pd.DataFrame({'Close': df_w['Close'], 'sma200': w_sma200_series}).dropna()
        if frame.empty: return None
        w_sma200_val = round(float(frame['sma200'].iloc[-1]), 2)
        upper_bound = w_sma200_val * (1 + W200_UPPER_BUFFER)
        if not (last_price >= w_sma200_val and last_price <= upper_bound): return None
        pct_above = round(((last_price - w_sma200_val) / w_sma200_val) * 100, 2)
        first_ever_date = frame.index[-1].date()
        for i in range(len(frame)):
            if float(frame['sma200'].iloc[i]) <= float(frame['Close'].iloc[i]) <= float(frame['sma200'].iloc[i]) * (
                    1 + W200_UPPER_BUFFER):
                first_ever_date = frame.index[i].date();
                break
        return w_sma200_val, pct_above, first_ever_date
    except Exception:
        return None


# ── Fibonacci ──────────────────────────────────────────────────────────────────
FIB_ZONES = [("0.382 - 0.500", 0.382, 0.500, "#1a6b3c"), ("0.500 - 0.618", 0.500, 0.618, "#2471a3"),
             ("0.618 - 0.786", 0.618, 0.786, "#7d3c98")]


def detect_fib_zone(last_price: float, hi_52: float, lo_52: float):
    rng = hi_52 - lo_52
    if rng <= 0: return None
    retrace = round((hi_52 - last_price) / rng, 4)
    for label, lo_b, hi_b, color in FIB_ZONES:
        if lo_b <= retrace < hi_b: return label, retrace, color
    return None


# ── HTML/CSS/JS Assets ────────────────────────────────────────────────────────
CSS = """<style>
  :root {
    --bull:#27ae60; --bull-lt:#eafaf1;
    --bear:#c0392b; --bear-lt:#fdedec;
    --comb:#b7770d; --comb-lt:#fef9e7;
    --st:#3498db;   --st-lt:#ebf5fb;
    --fib1:#1a6b3c; --fib1-lt:#e9f7ef;
    --fib2:#2471a3; --fib2-lt:#eaf3fb;
    --fib3:#7d3c98; --fib3-lt:#f5eef8;
    --w200:#0e6655; --w200-lt:#e8f8f5;
    --etf:#6c3483;  --etf-lt:#f4ecf7;
    --head:#2c3e50;
    --adx-weak:#95a5a6;
    --adx-neutral:#2471a3;    --adx-neutral-bg:#eaf3fb;
    --adx-strong:#27ae60;     --adx-strong-bg:#eafaf1;
    --adx-exhaust:#c0392b;    --adx-exhaust-bg:#fdedec;
  }
  * { box-sizing:border-box; }
  body { font-family:'Segoe UI',Arial,sans-serif; margin:0; padding:28px; background:#ecf0f1; color:#333; }
  h2   { color:var(--head); border-bottom:3px solid var(--head); padding-bottom:10px; margin-bottom:28px; }
  .adx-legend { display:inline-flex; align-items:center; gap:14px; flex-wrap:wrap; background:#fff; border:1px solid #ddd; border-radius:8px; padding:6px 14px; font-size:12px; margin-bottom:18px; }
  .adx-legend-dot { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:5px; }
  .section { background:white; border-radius:10px; padding:22px; margin-bottom:38px; box-shadow:0 2px 8px rgba(0,0,0,0.09); }
  .sec-title { font-size:16px; font-weight:700; color:var(--head); border-bottom:2px solid var(--head); padding-bottom:8px; margin-bottom:18px; }
  .pair { display:flex; gap:20px; flex-wrap:wrap; }
  .sub  { flex:1; min-width:300px; }
  .sub-hdr { font-size:13px; font-weight:700; padding:7px 10px; border-radius:5px 5px 0 0; color:white; margin-bottom:0; }
  .hdr-bull { background:var(--bull); }
  .hdr-bear { background:var(--bear); }
  .hdr-comb { background:var(--comb); }
  .hdr-st   { background:var(--st); }
  .hdr-w200 { background:var(--w200); }
  .hdr-etf  { background:var(--etf); }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th    { background:var(--head); color:white; padding:8px 7px; text-align:left; white-space:nowrap; cursor:pointer; user-select:none; }
  td    { padding:7px; border-bottom:1px solid #e5e5e5; }
  .r-bull td { background:var(--bull-lt); }
  .r-bear td { background:var(--bear-lt); }
  .r-st td   { background:var(--st-lt); }
  .r-w200 td { background:var(--w200-lt); }
  .r-etf td  { background:var(--etf-lt); }
  td.adx-weak    { color:var(--adx-weak) !important; font-style:italic; }
  td.adx-neutral { background:var(--adx-neutral-bg) !important; color:var(--adx-neutral) !important; font-weight:600; }
  td.adx-strong  { background:var(--adx-strong-bg) !important; color:var(--adx-strong) !important; font-weight:700; }
  td.adx-exhaust { background:var(--adx-exhaust-bg) !important; color:var(--adx-exhaust) !important; font-weight:700; }
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
    let floatA = parseFloat(valA); let floatB = parseFloat(valB);
    return isAsc ? (isNaN(floatA) ? valB.localeCompare(valA) : floatB - floatA) : (isNaN(floatA) ? valA.localeCompare(valB) : floatA - floatB);
  });
  rows.forEach(row => tbody.appendChild(row));
}
</script>"""


def get_table_html(data_list, table_id, row_class=""):
    if not data_list: return '<div class="no-sig" style="padding:10px; color:#999;">No signals detected.</div>'
    headers = [k for k in data_list[0].keys() if not k.startswith('_')]
    html = f'<table id="{table_id}"><thead><tr>'
    for i, h in enumerate(headers): html += f'<th onclick="tableSort(\'{table_id}\', {i})">{h}</th>'
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


# ── Main Processing Loop ──────────────────────────────────────────────────────

def run_scanner_for_group(group_name, ticker_list):
    results = {
        "w200_zone": [], "bullish_cross": [], "bearish_cross": [],
        "bullish_st": [], "bearish_st": [], "bullish_trend": [], "bearish_trend": [],
        "pullback": [], "fib_500_618": [], "etf_ema_fall": [],
    }

    log.info(f"Processing group '{group_name}' ({len(ticker_list)} tickers)...")

    for ticker in ticker_list:
        df, full_name, mktcap, ipo_year, quote_type = fetch_data(ticker)
        if df is None or len(df) < SMA_LONG: continue

        last_price = float(df['Close'].iloc[-1])
        hi_52 = float(df['High'].iloc[-252:].max())
        lo_52 = float(df['Low'].iloc[-252:].min())
        adx_val = get_adx(df)

        row_base = {
            "Instrument": f"{full_name} ({ticker})", "Last Price": round(last_price, 2),
            "ADX (14)": adx_val if adx_val is not None else "N/A", "Market Cap": format_market_cap(mktcap)
        }

        # ── ETF logic: Top 20 most fallen from 20 EMA ──
        if quote_type == 'ETF':
            e20_series = get_ema(df, 20)
            if e20_series is not None and not e20_series.dropna().empty:
                e20_val = float(e20_series.iloc[-1])
                # Negative value means price is below EMA (fallen)
                dist_pct = round(((last_price - e20_val) / e20_val) * 100, 2)
                results["etf_ema_fall"].append({
                    **row_base,
                    "EMA 20": round(e20_val, 2),
                    "% Dist from EMA20": dist_pct
                })

        # Indicators
        w200 = detect_weekly_sma200_zone(ticker, last_price)
        if w200: results["w200_zone"].append({**row_base, "SMA200": w200[0], "% Dist": w200[1], "First Date": w200[2]})

        c_type, c_date = detect_crossover(df)
        if c_type: results["bullish_cross" if c_type == "Golden" else "bearish_cross"].append(
            {**row_base, "Date": c_date})

        st_type, st_date = detect_supertrend_change(df)
        if st_type: results["bullish_st" if st_type == "Bullish" else "bearish_st"].append(
            {**row_base, "Flip Date": st_date})

        t_type, t_start, _, _, _ = detect_trend(df)
        if t_type: results["bullish_trend" if t_type == "Bullish" else "bearish_trend"].append(
            {**row_base, "Start": t_start})

        pb = detect_pullback(ticker, df)
        if pb: results["pullback"].append({**row_base, "D-RSI": pb[0], "W-RSI": pb[1], "M-RSI": pb[2]})

        fib = detect_fib_zone(last_price, hi_52, lo_52)
        if fib and fib[0] == "0.500 - 0.618": results["fib_500_618"].append({**row_base, "Retrace": fib[1]})

    # Sort ETF results by most fallen (lowest % distance) and take Top 20
    if results["etf_ema_fall"]:
        results["etf_ema_fall"] = sorted(results["etf_ema_fall"], key=lambda x: x["% Dist from EMA20"])[:20]

    report_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">{CSS}</head><body>
    <h2>Market Scanner: {group_name.upper()}</h2>

    <div class="section">
        <div class="sec-title">Top 20 ETFs: Most Fallen from 20 EMA</div>
        {get_table_html(results['etf_ema_fall'], 'etf-tbl', 'r-etf')}
    </div>

    <div class="section"><div class="sec-title">Weekly SMA 200 Zone Filter</div>{get_table_html(results['w200_zone'], 'w200-tbl', 'r-w200')}</div>

    <div class="section">
        <div class="sec-title">Supertrend (10,3) Trend Reversals</div>
        <div class="pair">
            <div class="sub">
                <div class="sub-hdr hdr-st">Bullish Flip</div>
                {get_table_html(results['bullish_st'], 'st-bull-tbl', 'r-st')}
            </div>
            <div class="sub">
                <div class="sub-hdr hdr-bear">Bearish Flip</div>
                {get_table_html(results['bearish_st'], 'st-bear-tbl', 'r-bear')}
            </div>
        </div>
    </div>

    <div class="section"><div class="sec-title">SMA Crossovers</div><div class="pair"><div class="sub"><div class="sub-hdr hdr-bull">Golden Cross</div>{get_table_html(results['bullish_cross'], 'c-bull-tbl', 'r-bull')}</div><div class="sub"><div class="sub-hdr hdr-bear">Death Cross</div>{get_table_html(results['bearish_cross'], 'c-bear-tbl', 'r-bear')}</div></div></div>
    <div class="section"><div class="sec-title">EMA Trend Alignment (20/50/200)</div><div class="pair"><div class="sub"><div class="sub-hdr hdr-bull">Bullish</div>{get_table_html(results['bullish_trend'], 't-bull-tbl', 'r-bull')}</div><div class="sub"><div class="sub-hdr hdr-bear">Bearish</div>{get_table_html(results['bearish_trend'], 't-bear-tbl', 'r-bear')}</div></div></div>
    {JS_SORTING}</body></html>"""

    with open(f"Scanner_{group_name}.html", "w", encoding="utf-8") as f:
        f.write(report_html)
    log.info(f"Report for {group_name} generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("--file", type=str, default=None);
    args = parser.parse_args()
    groups = load_instruments_grouped(target_file=args.file)
    if groups:
        for name, tickers in groups.items(): run_scanner_for_group(name, tickers)
