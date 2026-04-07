"""
SOLiGence IEAP Barometer — Local Web Dashboard
================================================
FastAPI backend that serves the premium web dashboard.

Run with:
    python dashboard_app.py
or:
    uvicorn dashboard_app:app --reload --port 8000

Then open: http://localhost:8000
"""

import os, sys, warnings, logging, datetime, traceback

warnings.filterwarnings("ignore")

import dotenv

dotenv.load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import yfinance as yf

# ── Setup ──────────────────────────────────────────────────────────────────────
log = logging.getLogger("BarometerDashboard")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Suppress TF noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ── Barometer import (after env setup) ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from barometer_core import BarometerSystem, DataPipeline

# ── Constants ──────────────────────────────────────────────────────────────────
TICKERS = ["MSFT", "AMZN", "AMGN", "NVDA"]
MODEL_DIR = "barometer_saved"
WINDOW = 60
DATA_DAYS = 730  # 2 years for warm-up indicators (SMA200, etc.)

# ── State ──────────────────────────────────────────────────────────────────────
_systems: dict[str, BarometerSystem] = {}  # loaded systems cache
_loading: dict[str, bool] = {}  # in-flight load flags
_errors: dict[str, str] = {}  # per-ticker error messages

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SOLiGence Barometer API",
    description="Real-time ensemble stock forecasting dashboard",
    version="2.0.0",
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fetch_live_data(ticker: str):
    """Download 2 years of data and engineer features for one ticker."""
    end = datetime.datetime.today().strftime("%Y-%m-%d")
    start = (datetime.datetime.today() - datetime.timedelta(days=DATA_DAYS)).strftime(
        "%Y-%m-%d"
    )
    pipe = DataPipeline(tickers=[ticker], start=start, end=end)
    pipe.download()
    pipe.prepare_all()
    df = pipe.feature_data[ticker]
    vix = pipe.raw["Close"]["^VIX"].reindex(df.index).ffill().bfill()
    spy = pipe.raw["Close"]["SPY"].pct_change().reindex(df.index).ffill().bfill()
    return df, vix, spy


def _load_system(ticker: str):
    """Synchronously load a BarometerSystem from disk. Call in a thread."""
    path = f"{MODEL_DIR}/{ticker}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}")
    sys_obj = BarometerSystem(ticker=ticker, window=WINDOW)
    sys_obj.load(path)
    return sys_obj


def _ensure_loaded(ticker: str):
    """Load system + live data synchronously if not already cached."""
    if ticker not in _systems:
        if _loading.get(ticker):
            raise HTTPException(
                status_code=503, detail=f"{ticker} is still loading, try again shortly."
            )
        _loading[ticker] = True
        _errors.pop(ticker, None)
        try:
            log.info(f"[{ticker}] Loading pre-trained system …")
            sys_obj = _load_system(ticker)
            log.info(f"[{ticker}] Fetching live market data …")
            df, vix, spy = _fetch_live_data(ticker)
            sys_obj._last_df = df
            sys_obj._last_vix = vix
            sys_obj._last_spy = spy
            _systems[ticker] = sys_obj
            log.info(f"[{ticker}] Ready ✓")
        except Exception as e:
            _errors[ticker] = str(e)
            _loading[ticker] = False
            raise HTTPException(status_code=500, detail=f"Failed to load {ticker}: {e}")
        finally:
            _loading[ticker] = False
    return _systems[ticker]


# ── Request / Response models ─────────────────────────────────────────────────
class WhatIfRequest(BaseModel):
    price_shock: float = 0.0  # e.g. -0.05
    volume_shock: float = 0.0  # e.g.  0.20
    vix_shock: float = 0.0  # e.g. 10.0


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/tickers")
def get_tickers():
    """Return available tickers and their load status."""
    result = {}
    for t in TICKERS:
        result[t] = {
            "loaded": t in _systems,
            "loading": _loading.get(t, False),
            "error": _errors.get(t),
            "model_exists": os.path.exists(f"{MODEL_DIR}/{t}"),
        }
    return result


@app.get("/api/signal/{ticker}")
def get_signal(ticker: str, refresh: bool = False):
    """
    Load model (if needed) and return live BUY/HOLD/SELL signals + predictions.
    Set refresh=true to re-fetch the latest market data.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not supported.")

    # Optionally refresh live data without re-loading the heavy models
    if refresh and ticker in _systems:
        try:
            log.info(f"[{ticker}] Refreshing live data …")
            df, vix, spy = _fetch_live_data(ticker)
            _systems[ticker]._last_df = df
            _systems[ticker]._last_vix = vix
            _systems[ticker]._last_spy = spy
        except Exception as e:
            log.warning(f"[{ticker}] Data refresh failed: {e}")

    system = _ensure_loaded(ticker)
    signals = system.generate_signal(conf_threshold=0.55)

    # Enrich with last close price
    last_close = float(system._last_df["close"].iloc[-1])
    last_date = str(system._last_df.index[-1].date())

    horizons = {}
    for h, v in signals.items():
        horizons[h] = {
            "signal": v["signal"].split()[0],  # BUY / HOLD / SELL
            "emoji": v["signal"].split()[-1] if len(v["signal"].split()) > 1 else "",
            "price_pred": v["price_pred"],
            "up_prob": round(v["up_prob"] * 100, 1),
            "confidence": round(v["confidence"] * 100, 1),
            "pct_change": round((v["price_pred"] - last_close) / last_close * 100, 2),
        }

    return {
        "ticker": ticker,
        "last_close": round(last_close, 2),
        "last_date": last_date,
        "horizons": horizons,
        "status": "ok",
    }


@app.post("/api/whatif/{ticker}")
def run_whatif(ticker: str, body: WhatIfRequest):
    """Run a stress-test scenario and return base vs shocked predictions."""
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not supported.")

    system = _ensure_loaded(ticker)

    try:
        result = system.what_if(
            price_shock=body.price_shock,
            volume_shock=body.volume_shock,
            vix_shock=body.vix_shock,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    output = {}
    for h in ["t1", "t5", "t21", "t63"]:
        if h not in result["base"]:
            continue
        bp = result["base"][h]["price"]
        sp = result["shocked"][h]["price"]
        output[h] = {
            "base_price": round(bp, 2),
            "shocked_price": round(sp, 2),
            "delta": round(sp - bp, 2),
            "delta_pct": round((sp - bp) / abs(bp) * 100, 2) if bp else 0,
            "impact": "PROTECTIVE" if (sp - bp) > -2 else "FOLLOWING",
        }

    return {
        "ticker": ticker,
        "scenario": result["scenario"],
        "results": output,
        "status": "ok",
    }


@app.get("/api/market/{ticker}")
def get_market_data(ticker: str):
    """Return recent OHLCV data for charting (last 90 trading days)."""
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not supported.")

    system = _ensure_loaded(ticker)
    df = system._last_df.tail(90)

    return {
        "ticker": ticker,
        "dates": [str(d.date()) for d in df.index],
        "close": [round(float(v), 2) for v in df["close"]],
        "volume": [int(v) for v in df["volume"]],
        "rsi": [round(float(v), 1) for v in df["rsi_14"].fillna(50)],
        "macd": [round(float(v), 4) for v in df["macd"].fillna(0)],
        "bb_upper": [round(float(v), 2) for v in df["bb_upper"].fillna(0)],
        "bb_lower": [round(float(v), 2) for v in df["bb_lower"].fillna(0)],
    }


# ── Static files + SPA entry point ────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "dashboard_static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Dashboard loading…</h1><p>Static files not found.</p>")


# ── Launcher ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  SOLiGence Barometer Dashboard")
    print("  http://localhost:8000")
    print("═" * 60 + "\n")
    uvicorn.run(
        "dashboard_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="warning",
    )
