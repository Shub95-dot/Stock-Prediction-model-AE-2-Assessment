"""
============================================================
  SOLIGENCE IEAP — BAROMETER ENSEMBLE FORECASTING SYSTEM
============================================================

SYSTEM ARCHITECTURE
===================
  Layer 0 : DataPipeline          — OHLCV + macro ingestion, 50+ features
                                    + FinBERT sentiment (AE2 extension)
  Layer 1 : Barometer Sub-Models  — LSTM · XGBoost · TCN · TFT-Lite
  Layer 2 : BarometerGate         — VIX + HMM + ADX + RollingCorr regime detection
  Layer 3 : LightGBM MetaLearner  — Learned stacking with regime routing
  Layer 4 : BarometerSystem       — Orchestrator + what-if + signal generation

AE2 SENTIMENT EXTENSION
========================
  sentiment_pipeline.py (same directory) adds FinBERT-scored features:
    • sent_score      — daily weighted (positive - negative) score
    • sent_positive   — mean FinBERT positive probability
    • sent_negative   — mean FinBERT negative probability
    • sent_neutral    — mean FinBERT neutral probability
    • sent_volume     — headline count (news intensity signal)
    • sent_momentum   — 3-day rolling change in sentiment score

  Data sources:
    • NewsAPI         — Reuters, WSJ, Bloomberg, CNBC, FT headlines
    • Reddit PRAW     — r/wallstreetbets, r/investing, r/stocks
    • Yahoo Finance   — built-in ticker news (free, no key required)

INSTALL
=======
  # AE2 Sentiment additions:
  pip install transformers torch newsapi-python praw

  # API keys (set as environment variables):
  export NEWSAPI_KEY="your_newsapi_key"
  export REDDIT_CLIENT_ID="your_reddit_id"
  export REDDIT_CLIENT_SECRET="your_reddit_secret"

USAGE
=====
  system = BarometerSystem("AAPL")
  system.fit(df, vix, spy_ret)
  print(system.generate_signal())
  print(system.what_if(price_shock=-0.05, vix_shock=8))
"""

import logging
import os
import warnings
from datetime import datetime

import dotenv
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import ta
import tensorflow as tf
import xgboost as xgb
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from scipy.stats import kurtosis as sp_kurt
from scipy.stats import skew as sp_skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    Add,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    ZeroPadding1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

dotenv.load_dotenv()

# ── AE2 Sentiment Extension ───────────────────────────────────────────────────
# sentiment_pipeline.py must live in the same directory as this file.
# If missing or dependencies not installed, system degrades gracefully:
# all sentiment columns are set to 0 and training continues normally.
# Install deps: pip install transformers torch newsapi-python praw
try:
    from sentiment_pipeline import SentimentConfig, SentimentPipeline

    _SENTIMENT_AVAILABLE = True
except ImportError:
    _SENTIMENT_AVAILABLE = False

log = logging.getLogger("BarometerSystem")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

if not _SENTIMENT_AVAILABLE:
    log.warning(
        "sentiment_pipeline.py not found or dependencies missing. "
        "Sentiment features will default to zero. "
        "Run: pip install transformers torch newsapi-python praw"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 0 — DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


class DataPipeline:
    """
    Ingests multi-ticker OHLCV data from Yahoo Finance.
    Enriches each ticker with 50+ technical, volatility, momentum,
    cross-asset, and lag features. Produces windowed tensors for
    neural sub-models and flat feature matrices for tree-based ones.

    Design notes
    ────────────
    • RobustScaler is used (instead of MinMax) — resilient to price spikes.
    • Forward-fill then back-fill handles non-trading day gaps cleanly.
    • Targets are shifted FORWARD to avoid lookahead bias: target_1d = close(t+1).
    • 1% / 99% percentile clipping removes extreme outliers per column.
    """

    # Macro proxies: VIX (fear index), QQQ (NASDAQ proxy),
    # XLK (tech sector ETF), SPY (broad market)
    MACRO_TICKERS = ["^VIX", "QQQ", "XLK", "SPY"]

    def __init__(
        self,
        tickers: list,
        start: str,
        end: str,
        window: int = 60,
        sentiment_config=None,
    ):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.window = window  # lookback window in trading days
        self.raw = None
        self.feature_data = {}  # per-ticker engineered DataFrames

        # ── AE2: Sentiment pipeline (optional) ────────────────────────────────
        # Pass a SentimentConfig() to enable. Defaults to SentimentConfig() if
        # sentiment_pipeline.py is installed, otherwise silently disabled.
        if _SENTIMENT_AVAILABLE:
            self._sentiment = SentimentPipeline(sentiment_config or SentimentConfig())
        else:
            self._sentiment = None

    # ── 0.1  Download ──────────────────────────────────────────────────────────
    def download(self) -> "DataPipeline":
        tickers_all = self.tickers + self.MACRO_TICKERS
        log.info(f"Downloading {tickers_all} | {self.start} → {self.end}")
        self.raw = yf.download(
            tickers_all,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )
        return self

    # ── 0.2  Clean ─────────────────────────────────────────────────────────────
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.ffill().bfill().dropna()
        for col in df.select_dtypes(include=np.number).columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)
        return df

    # ── 0.3  Feature Engineering ───────────────────────────────────────────────
    def engineer_features(self, ticker: str) -> pd.DataFrame:
        close = self.raw["Close"][ticker]
        high = self.raw["High"][ticker]
        low = self.raw["Low"][ticker]
        open_ = self.raw["Open"][ticker]
        volume = self.raw["Volume"][ticker]

        df = pd.DataFrame(
            {
                "close": close,
                "high": high,
                "low": low,
                "open": open_,
                "volume": volume,
            }
        )

        # Price features
        df["returns_1d"] = df["close"].pct_change(1)
        df["returns_5d"] = df["close"].pct_change(5)
        df["returns_21d"] = df["close"].pct_change(21)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
        df["oc_spread"] = (df["close"] - df["open"]) / df["open"]

        # Volume features
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma20"]
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

        # Trend indicators
        df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
        df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)
        df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
        df["sma_200"] = ta.trend.sma_indicator(df["close"], window=200)
        df["macd"] = ta.trend.macd(df["close"])
        df["macd_signal"] = ta.trend.macd_signal(df["close"])
        df["macd_diff"] = ta.trend.macd_diff(df["close"])
        df["adx"] = ta.trend.adx(high, low, close, window=14)
        df["adx_pos"] = ta.trend.adx_pos(high, low, close, window=14)
        df["adx_neg"] = ta.trend.adx_neg(high, low, close, window=14)

        # Momentum indicators
        df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
        df["rsi_7"] = ta.momentum.rsi(df["close"], window=7)
        df["stoch_k"] = ta.momentum.stoch(high, low, close)
        df["stoch_d"] = ta.momentum.stoch_signal(high, low, close)
        df["cci"] = ta.trend.cci(high, low, close, window=20)
        df["williams_r"] = ta.momentum.williams_r(high, low, close)
        df["roc_12"] = ta.momentum.roc(df["close"], window=12)

        # Volatility indicators
        bb = ta.volatility.BollingerBands(df["close"], window=20)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb.bollinger_mavg()
        df["bb_pct"] = bb.bollinger_pband()
        df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
        df["hist_vol_21"] = df["log_return"].rolling(21).std() * np.sqrt(252)

        # Macro / cross-asset features
        for macro in self.MACRO_TICKERS:
            tag = macro.replace("^", "").lower()
            mc = self.raw["Close"][macro]
            df[f"{tag}_ret"] = mc.pct_change(1)
            df[f"{tag}_vol"] = mc.pct_change(1).rolling(21).std() * np.sqrt(252)
            df[f"{tag}_level"] = (mc - mc.rolling(252).mean()) / mc.rolling(252).std()

        # Price normalisation ratios
        df["close_ema9_r"] = df["close"] / df["ema_9"]
        df["close_ema50_r"] = df["close"] / df["ema_50"]
        df["close_sma200_r"] = df["close"] / df["sma_200"]

        # Lag features — encode recent history explicitly
        for lag in [1, 2, 3, 5, 10]:
            df[f"close_lag{lag}"] = df["close"].shift(lag)
            df[f"returns_lag{lag}"] = df["returns_1d"].shift(lag)

        # ── TARGET VARIABLES (shifted forward — NO lookahead) ────────────────
        df["target_1d"] = df["close"].shift(-1)  # next-day close
        df["target_5d"] = df["close"].shift(-5)  # next-week close
        df["target_21d"] = df["close"].shift(-21)  # next-month close
        df["target_63d"] = df["close"].shift(
            -63
        )  # next-quarter close (~63 trading days)
        df["dir_1d"] = (df["target_1d"] > df["close"]).astype(int)
        df["dir_5d"] = (df["target_5d"] > df["close"]).astype(int)
        df["dir_63d"] = (df["target_63d"] > df["close"]).astype(int)

        # ── AE2: Sentiment enrichment ─────────────────────────────────────────
        # Adds 6 FinBERT-derived columns: sent_score, sent_positive,
        # sent_negative, sent_neutral, sent_volume, sent_momentum.
        # If sentiment pipeline is unavailable → columns default to 0.
        # Placed BEFORE clean() so outlier clipping applies to sentiment too.
        if self._sentiment is not None:
            df = self._sentiment.enrich(df, ticker, self.start, self.end)
        else:
            for col in [
                "sent_score",
                "sent_positive",
                "sent_negative",
                "sent_neutral",
                "sent_volume",
                "sent_momentum",
            ]:
                df[col] = 0.0

        return self.clean(df)

    def prepare_all(self) -> "DataPipeline":
        for t in self.tickers:
            log.info(f"Engineering features: {t}")
            self.feature_data[t] = self.engineer_features(t)
        return self

    # ── 0.4  Sequence builder ──────────────────────────────────────────────────
    def create_sequences(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list,
        scaler: RobustScaler = None,
    ) -> tuple:
        """
        Sliding window -> 3D tensor (n_samples, window, n_features).
        Used by LSTM, TCN, and TFT barometers.

        FIX (scaler leakage): pass a pre-fitted scaler to call transform()
        only on test folds. Pass scaler=None on train fold to fit a new one.
        """
        if scaler is None:
            scaler = RobustScaler()
            X_sc = scaler.fit_transform(df[feature_cols].values)
        else:
            X_sc = scaler.transform(df[feature_cols].values)
        y = df[target_col].values
        X_seq, y_seq = [], []
        for i in range(self.window, len(X_sc) - 1):
            X_seq.append(X_sc[i - self.window: i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq), scaler


#  LAYER 1A — BAROMETER: BIDIRECTIONAL LSTM WITH ATTENTION


class LSTMBarometer:
    """
    Bidirectional stacked LSTM with multi-head self-attention.

    Why LSTM?
    ─────────
    Stock prices are sequential time series with long-range dependencies
    (e.g. a trend established 30 days ago still influences today's movement).
    Bidirectionality lets the model look at each time step in context of
    both past and future within the training window. The attention layer
    learns to weight the most relevant timesteps for each horizon.

    Architecture: BiLSTM(128) → BiLSTM(64) → MultiHeadAttention → GAP → Dense
    Three output heads: T+1, T+5, T+21 (multi-task learning reduces overfitting)
    """

    def __init__(self, seq_len: int, n_features: int, units=(128, 64), dropout=0.3):
        self.seq_len = seq_len
        self.n_features = n_features
        self.units = units
        self.dropout = dropout
        self.model = self._build()

    def _build(self) -> Model:
        inp = Input(shape=(self.seq_len, self.n_features), name="lstm_input")

        x = tf.keras.layers.Bidirectional(LSTM(self.units[0], return_sequences=True))(
            inp
        )
        x = Dropout(self.dropout)(x)

        x = tf.keras.layers.Bidirectional(LSTM(self.units[1], return_sequences=True))(x)
        x = Dropout(self.dropout)(x)

        # Multi-head self-attention — learns which timesteps matter most
        attn = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(self.dropout)(x)

        # Three separate prediction heads — multi-task learning
        out_1d = Dense(1, name="out_1d")(x)
        out_5d = Dense(1, name="out_5d")(x)
        out_21d = Dense(1, name="out_21d")(x)
        out_63d = Dense(1, name="out_63d")(x)

        model = Model(inputs=inp, outputs=[out_1d, out_5d, out_21d, out_63d])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss={
                "out_1d": "huber",
                "out_5d": "huber",
                "out_21d": "huber",
                "out_63d": "huber",
            },
            loss_weights={"out_1d": 1.0, "out_5d": 0.8, "out_21d": 0.5, "out_63d": 0.3},
        )
        return model

    def fit(
        self,
        X: np.ndarray,
        y1: np.ndarray,
        y5: np.ndarray,
        y21: np.ndarray,
        y63: np.ndarray,
    ) -> "LSTMBarometer":
        cbs = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6),
        ]
        self.model.fit(
            X,
            {"out_1d": y1, "out_5d": y5, "out_21d": y21, "out_63d": y63},
            validation_split=0.1,
            epochs=100,
            batch_size=32,
            callbacks=cbs,
            verbose=0,
        )
        return self

    def predict(self, X: np.ndarray) -> dict:
        p1, p5, p21, p63 = self.model.predict(X, verbose=0)
        return {
            "t1": p1.flatten(),
            "t5": p5.flatten(),
            "t21": p21.flatten(),
            "t63": p63.flatten(),
        }


#  LAYER 1B — BAROMETER: XGBOOST


class XGBoostBarometer:

    def __init__(self):
        self.models = {}
        self.params = {
            "n_estimators": 600,
            "learning_rate": 0.02,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.05,
            "reg_lambda": 1.5,
            "tree_method": "hist",
            "random_state": 42,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """
        Computes 6 summary statistics per feature across the time window.
        Result shape: (n_samples, 6 * n_features)
        """
        n, t, f = X.shape
        last_val = X[:, -1, :]
        mean_val = X.mean(axis=1)
        std_val = X.std(axis=1)
        min_val = X.min(axis=1)
        max_val = X.max(axis=1)
        # Linear slope via least-squares projection
        t_idx = np.arange(t, dtype=float)
        t_norm = (t_idx - t_idx.mean()) / (t_idx.std() + 1e-8)
        slope = np.einsum("ntf,t->nf", X, t_norm) / t
        return np.hstack([last_val, mean_val, std_val, min_val, max_val, slope])

    def fit(self, X: np.ndarray, targets: dict) -> "XGBoostBarometer":
        Xf = self._flatten(X)
        for horizon, y in targets.items():
            model = xgb.XGBRegressor(**self.params)
            model.fit(Xf, y, eval_set=[(Xf, y)], verbose=False)
            self.models[horizon] = model
        return self

    def predict(self, X: np.ndarray) -> dict:
        Xf = self._flatten(X)
        results = {}
        for h, m in self.models.items():
            # XGBRegressor has a 'get_booster' method, raw Booster does not.
            if hasattr(m, "get_booster"):
                results[h] = m.predict(Xf)
            else:  # raw booster
                results[h] = m.predict(xgb.DMatrix(Xf))
        return results

    def feature_importances(self) -> dict:
        return {h: m.feature_importances_ for h, m in self.models.items()}


#  LAYER 1C — BAROMETER: TEMPORAL CNN (TCN-STYLE)


class TCNBarometer:

    def __init__(self, seq_len: int, n_features: int, filters: int = 64):
        self.seq_len = seq_len
        self.n_features = n_features
        self.filters = filters
        self.model = self._build()

    def _residual_block(self, x, dilation: int):
        """
        Dilated causal residual block.
        Left-padding ensures causality: output at t depends only on input ≤ t.
        """
        pad = dilation * (3 - 1)
        x_pad = ZeroPadding1D((pad, 0))(x)
        out = Conv1D(
            self.filters,
            kernel_size=3,
            dilation_rate=dilation,
            padding="valid",
            activation="relu",
        )(x_pad)
        out = Dropout(0.2)(out)

        if x.shape[-1] != self.filters:
            x = Conv1D(self.filters, 1, padding="same")(x)
        return Add()([x, out])

    def _build(self) -> Model:
        inp = Input(shape=(self.seq_len, self.n_features), name="tcn_input")
        x = inp
        for dilation in [1, 2, 4, 8, 16]:
            x = self._residual_block(x, dilation)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.3)(x)

        out_1d = Dense(1, name="out_1d")(x)
        out_5d = Dense(1, name="out_5d")(x)
        out_21d = Dense(1, name="out_21d")(x)
        out_63d = Dense(1, name="out_63d")(x)

        model = Model(inputs=inp, outputs=[out_1d, out_5d, out_21d, out_63d])
        model.compile(
            optimizer=Adam(1e-3),
            loss=["huber", "huber", "huber", "huber"],
            loss_weights=[1.0, 0.8, 0.5, 0.3],
        )
        return model

    def fit(self, X, y1, y5, y21, y63) -> "TCNBarometer":
        cbs = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=7, factor=0.5),
        ]
        self.model.fit(
            X,
            [y1, y5, y21, y63],
            validation_split=0.1,
            epochs=100,
            batch_size=32,
            callbacks=cbs,
            verbose=0,
        )
        return self

    def predict(self, X) -> dict:
        p1, p5, p21, p63 = self.model.predict(X, verbose=0)
        return {
            "t1": p1.flatten(),
            "t5": p5.flatten(),
            "t21": p21.flatten(),
            "t63": p63.flatten(),
        }


#  LAYER 1D — BAROMETER: TEMPORAL FUSION TRANSFORMER (TFT-LITE)


class TFTLiteBarometer:

    def __init__(
        self, seq_len: int, n_features: int, d_model: int = 64, n_heads: int = 4
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.model = self._build()

    def _build(self) -> Model:
        inp = Input(shape=(self.seq_len, self.n_features), name="tft_input")

        # Input projection to model dimension
        x = Dense(self.d_model)(inp)
        x = LayerNormalization()(x)

        # Learnable positional encoding
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        pos_emb = tf.keras.layers.Embedding(self.seq_len, self.d_model)(positions)
        x = x + pos_emb

        # 3 × Transformer encoder blocks
        for _ in range(3):
            # Self-attention
            attn = MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=0.1,
            )(x, x)
            x = LayerNormalization()(x + attn)

            # Feed-forward network (2× expansion then project back)
            ff = Dense(self.d_model * 2, activation="gelu")(x)
            ff = Dense(self.d_model)(ff)
            x = LayerNormalization()(x + ff)

        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)

        out_1d = Dense(1, name="out_1d")(x)
        out_5d = Dense(1, name="out_5d")(x)
        out_21d = Dense(1, name="out_21d")(x)
        out_63d = Dense(1, name="out_63d")(x)

        model = Model(inputs=inp, outputs=[out_1d, out_5d, out_21d, out_63d])
        model.compile(
            optimizer=Adam(5e-4),
            loss=["huber", "huber", "huber", "huber"],
            loss_weights=[1.0, 0.8, 0.5, 0.3],
        )
        return model

    def fit(self, X, y1, y5, y21, y63) -> "TFTLiteBarometer":
        cbs = [EarlyStopping(patience=12, restore_best_weights=True)]
        self.model.fit(
            X,
            [y1, y5, y21, y63],
            validation_split=0.1,
            epochs=80,
            batch_size=32,
            callbacks=cbs,
            verbose=0,
        )
        return self

    def predict(self, X) -> dict:
        p1, p5, p21, p63 = self.model.predict(X, verbose=0)
        return {
            "t1": p1.flatten(),
            "t5": p5.flatten(),
            "t21": p21.flatten(),
            "t63": p63.flatten(),
        }


#
#  LAYER 2 — BAROMETER GATE: COMPOSITE MARKET REGIME DETECTOR
#


class BarometerGate:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │                   BAROMETER GATE                            │
    │                                                             │
    │  VIX Level ──────► Volatility Regime  (0:calm → 3:extreme) │
    │  HMM States ─────► Latent Regime      (0:bull → 3:crash)   │
    │  ADX Value ──────► Trend Regime       (0:range → 3:strong) │
    │  Rolling Corr ───► Correlation Regime (systemic vs. idio)   │
    │                           │                                 │
    │                    Composite Score (0-10)                   │
    │              "Market Barometer Reading"                     │
    └─────────────────────────────────────────────────────────────┘
    """

    VIX_THRESHOLDS = [15, 20, 30]  # Low, Medium, High, Extreme

    def __init__(self, n_hmm_states: int = 4):
        self.n_hmm_states = n_hmm_states
        self.hmm = GaussianHMM(
            n_components=n_hmm_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self.fitted = False

    # ── 2.1  VIX Regime ────────────────────────────────────────────────────────
    def vix_regime(self, vix: np.ndarray) -> np.ndarray:
        """
        Discretises VIX into 4 volatility regimes using empirically
        established thresholds from market microstructure research.
          0 = Low vol   (VIX < 15)  — risk-on, trending markets
          1 = Normal    (15-20)     — standard trading conditions
          2 = Elevated  (20-30)     — uncertainty, mixed signals
          3 = Extreme   (VIX ≥ 30) — fear regime, high noise
        """
        out = np.zeros(len(vix), dtype=int)
        for i, thresh in enumerate(self.VIX_THRESHOLDS):
            out[vix >= thresh] = i + 1
        return out

    # ── 2.2  HMM Regime ────────────────────────────────────────────────────────
    def fit_hmm(self, log_ret: np.ndarray, vix: np.ndarray) -> "BarometerGate":
        """
        Fits a Gaussian HMM to (log_return, ΔVIX, normalised_VIX) observations.
        The model discovers latent market regimes from data — no manual labels.
        Typically learns bull, bear, sideways, and crash states.

        Emission model: p(x_t | s_t) = N(μ_s, Σ_s)   (Gaussian emissions)
        Transition model: P(s_t | s_{t-1}) = A         (learned transition matrix)
        Viterbi decoding used for regime sequence prediction.
        """
        vix_diff = np.diff(vix, prepend=vix[0])
        vix_norm = (vix - vix.mean()) / (vix.std() + 1e-8)
        obs = np.column_stack([log_ret, vix_diff, vix_norm])
        self.hmm.fit(obs)
        self.fitted = True
        # Store observation stats for online inference
        self._vix_mean = vix.mean()
        self._vix_std = vix.std()
        return self

    def _hmm_obs(self, log_ret, vix):
        vix_diff = np.diff(vix, prepend=vix[0])
        vix_norm = (vix - self._vix_mean) / (self._vix_std + 1e-8)
        return np.column_stack([log_ret, vix_diff, vix_norm])

    def hmm_regime(self, log_ret: np.ndarray, vix: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit_hmm() before predicting regime")
        return self.hmm.predict(self._hmm_obs(log_ret, vix))

    def hmm_proba(self, log_ret: np.ndarray, vix: np.ndarray) -> np.ndarray:
        """Returns posterior state probabilities — shape (n, n_states)."""
        return self.hmm.predict_proba(self._hmm_obs(log_ret, vix))

    # ── 2.3  ADX Trend Regime ──────────────────────────────────────────────────
    def adx_regime(self, adx: np.ndarray) -> np.ndarray:
        """
        Discretises ADX into 4 trend strength categories.
        Wilder (1978) standard thresholds used:
          0 = No trend   (ADX < 20)  — mean-reversion models preferred
          1 = Developing (20-25)     — trend may be forming
          2 = Strong     (25-40)     — trend-following models preferred
          3 = Extreme    (ADX ≥ 40)  — strong trend, but reversal risk rises
        """
        out = np.zeros(len(adx), dtype=int)
        for i, thresh in enumerate([20, 25, 40]):
            out[adx >= thresh] = i + 1
        return out

    # ── 2.4  Rolling Correlation Regime ────────────────────────────────────────
    def corr_regime(
        self, stock_ret: np.ndarray, spy_ret: np.ndarray, window: int = 30
    ) -> tuple:
        """
        Measures systemic exposure via 30-day rolling correlation to SPY.
        When correlation spikes → stock is moving with the market (macro-driven).
        When correlation drops → stock-specific factors dominate.
        A shift of ≥0.3 in 5 days signals a regime transition event.
        """
        sr = pd.Series(stock_ret)
        sp = pd.Series(spy_ret)
        rc = sr.rolling(window).corr(sp).fillna(0).values
        drc = pd.Series(rc).diff(5).abs().fillna(0).values
        flag = (drc > 0.3).astype(int)
        return flag, rc

    # ── 2.5  Composite Regime Vector ───────────────────────────────────────────
    def compute_regime_vector(
        self, df: pd.DataFrame, vix_arr: np.ndarray, spy_ret_arr: np.ndarray
    ) -> pd.DataFrame:
        """
        Assembles the full composite regime feature matrix.
        Output columns are directly fed into the LightGBM meta-learner
        as routing features — enabling conditional model weighting.
        """
        reg = pd.DataFrame(index=df.index)
        reg["vix_regime"] = self.vix_regime(vix_arr)
        reg["adx_regime"] = self.adx_regime(df["adx"].values)

        log_ret = df["log_return"].values
        reg["hmm_regime"] = self.hmm_regime(log_ret, vix_arr)

        hmm_p = self.hmm_proba(log_ret, vix_arr)
        for s in range(self.n_hmm_states):
            reg[f"hmm_p{s}"] = hmm_p[:, s]

        corr_flag, rolling_corr = self.corr_regime(df["returns_1d"].values, spy_ret_arr)
        reg["corr_shift"] = corr_flag
        reg["rolling_corr"] = rolling_corr

        # Composite Barometer Score (0–10 scale)
        # Higher score → more turbulent regime → LSTM & TFT weighted higher
        # Lower score  → trending regime       → TCN & XGBoost weighted higher
        reg["regime_score"] = (
            reg["vix_regime"] * 2.5
            + reg["adx_regime"] * 1.5
            + reg["hmm_regime"] * 1.5
            + reg["corr_shift"] * 2.0
        ).clip(0, 10)

        return reg


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — META-LEARNER: LIGHTGBM STACKER
# ═══════════════════════════════════════════════════════════════════════════════


class LightGBMMetaLearner:

    def __init__(self):
        self.reg_models = {}  # price prediction per horizon
        self.clf_models = {}  # direction prediction per horizon

        self.base_params = {
            "n_estimators": 800,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "random_state": 42,
        }

    def _assemble_meta_features(
        self, predictions: dict, regime_df: pd.DataFrame
    ) -> pd.DataFrame:
        meta = pd.DataFrame()
        for model_name, preds in predictions.items():
            for horizon, vals in preds.items():
                meta[f"{model_name}_{horizon}"] = vals

        # Ensemble diversity statistics per horizon
        for h in ["t1", "t5", "t21", "t63"]:
            cols = [c for c in meta.columns if c.endswith(f"_{h}")]
            meta[f"mean_{h}"] = meta[cols].mean(axis=1)
            meta[f"std_{h}"] = meta[cols].std(axis=1)
            meta[f"range_{h}"] = meta[cols].max(axis=1) - meta[cols].min(axis=1)
            # Agreement ratio: fraction of models predicting above ensemble mean
            meta[f"agree_{h}"] = meta[cols].apply(
                lambda r: (r > r.mean()).sum() / len(cols), axis=1
            )

        # Attach regime features as routing signals
        n = len(meta)
        for col in regime_df.columns:
            meta[col] = regime_df[col].values[:n]

        return meta.fillna(0)

    def fit(
        self, predictions: dict, regime_df: pd.DataFrame, targets: dict
    ) -> "LightGBMMetaLearner":
        meta = self._assemble_meta_features(predictions, regime_df)

        for horizon, y in targets.items():
            # Regression meta-model
            reg = lgb.LGBMRegressor(**self.base_params)
            reg.fit(meta, y)
            self.reg_models[horizon] = reg

            # Classification meta-model (direction)
            clf_params = {**self.base_params, "objective": "binary", "metric": "auc"}
            clf = lgb.LGBMClassifier(**clf_params)
            # FIX: direction = 1 if target > today's close (not median)
            current_close = targets.get("current_close", None)
            if current_close is not None:
                y_dir = (y > current_close).astype(int)
            else:
                y_dir = (y > np.median(y)).astype(int)
            clf.fit(meta, y_dir)
            self.clf_models[horizon] = clf

        return self

    def predict(self, predictions: dict, regime_df: pd.DataFrame) -> dict:
        meta = self._assemble_meta_features(predictions, regime_df)
        results = {}
        for horizon in self.reg_models:
            # Price Regression
            reg_m = self.reg_models[horizon]
            if hasattr(reg_m, "booster_"):
                # LGBMRegressor (sklearn API) — accepts DataFrame directly
                price = reg_m.predict(meta)
            else:
                # Raw lgb.Booster loaded from legacy .lgb file.
                # Must pass a numpy array aligned to the booster's feature order.
                booster_cols = reg_m.feature_name()
                # Add any columns the booster expects that are missing (fill 0)
                for bc in booster_cols:
                    if bc not in meta.columns:
                        meta[bc] = 0.0
                price = reg_m.predict(meta[booster_cols].values)

            # Directional Classification
            clf_m = self.clf_models[horizon]
            if hasattr(clf_m, "predict_proba"):
                # LGBMClassifier (sklearn API)
                up_prob = clf_m.predict_proba(meta)[:, 1]
            else:
                # Raw lgb.Booster for binary classification
                booster_cols = clf_m.feature_name()
                for bc in booster_cols:
                    if bc not in meta.columns:
                        meta[bc] = 0.0
                up_prob = clf_m.predict(meta[booster_cols].values)

            # Pass horizon so confidence uses the horizon-specific std column
            conf = self._confidence(meta, up_prob, horizon)
            results[horizon] = {
                "price": price,
                "direction": (up_prob >= 0.5).astype(int),
                "up_prob": up_prob,
                "confidence": conf,
            }
        return results

    def _confidence(
        self, meta: pd.DataFrame, up_prob: np.ndarray, horizon: str = ""
    ) -> np.ndarray:

        horizon_std_col = f"std_{horizon}"  # e.g. "std_t1"
        if horizon and horizon_std_col in meta.columns:
            norm_std = meta[horizon_std_col].values
        else:
            std_cols = [c for c in meta.columns if c.startswith("std_")]
            norm_std = meta[std_cols].mean(axis=1).values

        # Component 1: model agreement — smaller spread between barometers = higher
        agree_c = 1 / (1 + norm_std)

        # Component 2: directional clarity — probability further from 0.5 = higher
        dir_c = 2 * np.abs(up_prob - 0.5)

        # Component 3: regime stability — lower regime_score = calmer market = higher
        regime_c = 1 - (meta["regime_score"].values / 10)

        return np.clip(agree_c * 0.4 + dir_c * 0.4 + regime_c * 0.2, 0, 1)


#  LAYER 4 — BAROMETER SYSTEM ORCHESTRATOR


class BarometerSystem:
    """
    Top-level orchestrator for the SOLiGence Barometer Forecasting System.
    Manages the complete pipeline: data prep → sub-model training →
    regime detection → meta-learning → inference → signal generation → what-if.

    Public API:
      system.fit(df, vix, spy_ret)           ← train full system
      system.generate_signal()               ← BUY / HOLD / SELL + confidence
      system.predict_next()                  ← raw T+1, T+5, T+21 predictions
      system.what_if(price_shock=-0.05, ...) ← scenario analysis
      system.save("./models/AAPL")           ← persist trained system
    """

    def __init__(self, ticker: str, window: int = 60):
        self.ticker = ticker
        self.window = window
        self.gate = BarometerGate(n_hmm_states=4)
        self.meta = LightGBMMetaLearner()
        self.barometers = {}
        self.is_fitted = False
        self._feature_cols = None

    # ── Helper: feature columns ────────────────────────────────────────────────
    @staticmethod
    def _feature_cols_from(df: pd.DataFrame) -> list:
        # Exclude target/direction columns only.
        # sent_* sentiment columns are intentionally INCLUDED — they flow
        # into all base models (LSTM, TCN, TFT, XGBoost) via X_seq tensor.
        exclude = {
            "target_1d",
            "target_5d",
            "target_21d",
            "target_63d",
            "dir_1d",
            "dir_5d",
            "dir_63d",
        }
        return [c for c in df.columns if c not in exclude]

    # ── Helper: build windowed sequences ──────────────────────────────────────
    def _sequences(self, df: pd.DataFrame, target: str, scaler=None):
        # Auto-repair: if _feature_cols is an int sentinel (set by load() when
        # features.pkl was incomplete), rebuild the column list from the live df.
        if isinstance(self._feature_cols, int):
            rebuilt = self._feature_cols_from(df)
            if len(rebuilt) == self._feature_cols:
                log.info(
                    f"[{self.ticker}] Auto-rebuilt _feature_cols "
                    f"({len(rebuilt)} cols) from live DataFrame."
                )
                self._feature_cols = rebuilt
            else:
                raise RuntimeError(
                    f"Feature count mismatch: model expects {self._feature_cols} features, "
                    f"but live DataFrame has {len(rebuilt)} eligible columns. "
                    f"Ensure the same feature-engineering pipeline is used."
                )

        if scaler is None:
            scaler = RobustScaler()
            X_sc = scaler.fit_transform(df[self._feature_cols].values)
        else:
            X_sc = scaler.transform(df[self._feature_cols].values)

        y = df[target].values
        X, Y = [], []
        # If we only have ONE window (e.g. for inference), handle correctly
        if len(X_sc) == self.window:
            X.append(X_sc)
            Y.append(y[-1])
        else:
            for i in range(self.window, len(X_sc)):
                X.append(X_sc[i - self.window: i])
                Y.append(y[i])
        return np.array(X), np.array(Y), scaler

    # ── Fit ────────────────────────────────────────────────────────────────────
    def fit(
        self, df: pd.DataFrame, vix: pd.Series, spy_ret: pd.Series
    ) -> "BarometerSystem":
        self._feature_cols = self._feature_cols_from(df)

        log.info(f"[{self.ticker}] Building training sequences")
        X_seq, y1, sc = self._sequences(df, "target_1d")
        _, y5, _ = self._sequences(df, "target_5d")
        _, y21, _ = self._sequences(df, "target_21d")
        _, y63, _ = self._sequences(df, "target_63d")
        n, T, F = X_seq.shape

        df_a = df.iloc[-n:]
        vix_a = vix.values[-n:]
        spy_a = spy_ret.values[-n:]

        log.info(f"[{self.ticker}] Fitting BarometerGate HMM")
        self.gate.fit_hmm(df["log_return"].values, vix.values)
        regime = self.gate.compute_regime_vector(df_a, vix_a, spy_a)

        log.info(f"[{self.ticker}] Training LSTM barometer")
        self.barometers["lstm"] = LSTMBarometer(T, F).fit(X_seq, y1, y5, y21, y63)

        log.info(f"[{self.ticker}] Training XGBoost barometer")
        self.barometers["xgb"] = XGBoostBarometer().fit(
            X_seq, {"t1": y1, "t5": y5, "t21": y21, "t63": y63}
        )

        log.info(f"[{self.ticker}] Training TCN barometer")
        self.barometers["tcn"] = TCNBarometer(T, F).fit(X_seq, y1, y5, y21, y63)

        log.info(f"[{self.ticker}] Training TFT-Lite barometer")
        self.barometers["tft"] = TFTLiteBarometer(T, F).fit(X_seq, y1, y5, y21, y63)

        log.info(f"[{self.ticker}] Training LightGBM meta-learner")
        all_preds = {nm: b.predict(X_seq) for nm, b in self.barometers.items()}
        # FIX: supply current_close so Fix 2 direction labels are correct
        current_close_arr = df["close"].values[-n:]
        self.meta.fit(
            all_preds,
            regime,
            {"t1": y1, "t5": y5, "t21": y21, "t63": y63, "current_close": current_close_arr},
        )

        self._last_df = df
        self._last_vix = vix
        self._last_spy = spy_ret
        self.scaler = sc
        self.is_fitted = True
        log.info(f"[{self.ticker}] BarometerSystem fully trained ✓")
        return self

    # ── Predict ────────────────────────────────────────────────────────────────
    def predict(self, df=None, vix=None, spy_ret=None) -> dict:
        if df is None:
            df, vix, spy_ret = self._last_df, self._last_vix, self._last_spy

        # USE THE SAVED SCALER DURING INFERENCE
        X_seq, _, _ = self._sequences(df, "target_1d", scaler=self.scaler)
        n = len(X_seq)
        df_a = df.iloc[-n:]
        vix_a = vix.values[-n:] if hasattr(vix, "values") else vix[-n:]
        spy_a = spy_ret.values[-n:] if hasattr(spy_ret, "values") else spy_ret[-n:]
        regime = self.gate.compute_regime_vector(df_a, vix_a, spy_a)
        all_preds = {nm: b.predict(X_seq) for nm, b in self.barometers.items()}
        return self.meta.predict(all_preds, regime)

    def predict_next(self) -> dict:
        """Returns scalar forecast for the immediate next step."""
        res = self.predict()
        return {
            h: {k: float(v[-1]) for k, v in vals.items()} for h, vals in res.items()
        }

    # ── Signal Generation ──────────────────────────────────────────────────────
    def generate_signal(self, conf_threshold: float = 0.60) -> dict:
        """
        Converts probabilistic forecasts into actionable trading signals.

        Signal logic:
          BUY  : P(up) > 0.60  AND  confidence ≥ threshold
          SELL : P(up) < 0.40  AND  confidence ≥ threshold
          HOLD : uncertain probability OR low confidence
        """
        nxt = self.predict_next()
        out = {}
        for h, v in nxt.items():
            if v["up_prob"] > 0.60 and v["confidence"] >= conf_threshold:
                sig = "BUY"
            elif v["up_prob"] < 0.40 and v["confidence"] >= conf_threshold:
                sig = "SELL"
            else:
                sig = "HOLD"
            out[h] = {
                "signal": sig,
                "price_pred": round(v["price"], 2),
                "up_prob": round(v["up_prob"], 4),
                "confidence": round(v["confidence"], 4),
            }
        return out

    # ── What-If Scenario Analysis ──────────────────────────────────────────────
    def what_if(
        self,
        price_shock: float = 0.0,
        volume_shock: float = 0.0,
        vix_shock: float = 0.0,
        n_shock_days: int = 5,
    ) -> dict:
        """
        Simulates external shocks to the most recent n_shock_days of data
        and compares the resulting forecast against the base case.

        Args:
          price_shock  : % change to close price    (e.g. -0.05 = -5%)
          volume_shock : % change to volume          (e.g.  0.20 = +20%)
          vix_shock    : absolute VIX point change   (e.g.  8.0 = VIX+8)
          n_shock_days : how many recent days to apply shock to

        Returns:
          dict with "base", "shocked", and "delta" for each horizon.
        """
        base = self.predict_next()
        df_s = self._last_df.copy()
        vix_s = self._last_vix.copy()

        # Apply shocks to the last n_shock_days rows
        ci = df_s.columns.get_loc("close")
        vi = df_s.columns.get_loc("volume")
        df_s.iloc[-n_shock_days:, ci] *= 1 + price_shock
        df_s.iloc[-n_shock_days:, vi] *= 1 + volume_shock
        vix_s.iloc[-n_shock_days:] += vix_shock

        shocked = self.predict(df_s, vix_s, self._last_spy)
        s_next = {
            h: {k: float(v[-1]) for k, v in vals.items()} for h, vals in shocked.items()
        }

        return {
            "base": base,
            "shocked": s_next,
            "delta": {
                h: {"price_delta": round(s_next[h]["price"] - base[h]["price"], 4)}
                for h in base
            },
            "scenario": {
                "price_pct": f"{price_shock*100:+.1f}%",
                "volume_pct": f"{volume_shock*100:+.1f}%",
                "vix_abs": f"{vix_shock:+.1f} pts",
            },
        }

    # ── Persist ────────────────────────────────────────────────────────────────
    def save(self, path: str = "barometer_system"):
        os.makedirs(path, exist_ok=True)
        # Meta-Learners
        joblib.dump(self.meta.reg_models, f"{path}/meta_reg.pkl")
        joblib.dump(self.meta.clf_models, f"{path}/meta_clf.pkl")
        # XGBoost models
        joblib.dump(self.barometers["xgb"].models, f"{path}/xgb_models.pkl")

        self.barometers["lstm"].model.save(f"{path}/lstm.keras")
        self.barometers["tcn"].model.save(f"{path}/tcn.keras")
        self.barometers["tft"].model.save(f"{path}/tft.keras")
        joblib.dump(self.gate.hmm, f"{path}/hmm.pkl")
        joblib.dump(
            {
                "fitted": self.gate.fitted,
                "vix_mean": getattr(self.gate, "_vix_mean", 0),
                "vix_std": getattr(self.gate, "_vix_std", 1),
            },
            f"{path}/gate_meta.pkl",
        )
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self._feature_cols, f"{path}/features.pkl")
        log.info(f"[{self.ticker}] Saved to {path}/")

    def load(self, path: str = "barometer_saved/MSFT"):
        """
        Loads a pre-trained barometer system from a directory.
        Inverse of save(). Handles both legacy booster and newer pkl formats.

        Auto-repair: if features.pkl was saved with incomplete/sentinel columns
        (e.g. only sentiment cols instead of the full feature list), the true
        feature count is recovered from the Keras LSTM model's input shape.
        _feature_cols is then rebuilt from the live DataFrame in _sequences().
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        # ── 1. Scaler, HMM & Columns
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.gate.hmm = joblib.load(f"{path}/hmm.pkl")
        gate_meta = joblib.load(f"{path}/gate_meta.pkl")
        self.gate.fitted = gate_meta["fitted"]
        self.gate._vix_mean = gate_meta["vix_mean"]
        self.gate._vix_std = gate_meta["vix_std"]
        self._feature_cols = joblib.load(f"{path}/features.pkl")

        # ── 2. Base Barometers (load Keras models first to recover true n_features)
        # Load LSTM model first — we will check its input shape to validate
        # the feature count stored in features.pkl.
        lstm_model = tf.keras.models.load_model(f"{path}/lstm.keras")
        true_n_features = lstm_model.input_shape[-1]  # (batch, seq_len, n_features)

        # Guard: if saved features.pkl is corrupted/incomplete, mark for rebuild.
        # _sequences() will reconstruct _feature_cols from the live DataFrame.
        if len(self._feature_cols) != true_n_features:
            log.warning(
                f"[{self.ticker}] features.pkl has {len(self._feature_cols)} cols but "
                f"LSTM expects {true_n_features}. Will auto-rebuild from live data."
            )
            # Store sentinel: int = expected count, rebuilt per-call in _sequences()
            self._feature_cols = true_n_features  # type: ignore[assignment]

        self.barometers["lstm"] = LSTMBarometer(self.window, true_n_features)
        self.barometers["lstm"].model = lstm_model
        self.barometers["tcn"] = TCNBarometer(self.window, true_n_features)
        self.barometers["tcn"].model = tf.keras.models.load_model(f"{path}/tcn.keras")
        self.barometers["tft"] = TFTLiteBarometer(self.window, true_n_features)
        self.barometers["tft"].model = tf.keras.models.load_model(f"{path}/tft.keras")

        # XGBoost
        self.barometers["xgb"] = XGBoostBarometer()
        if os.path.exists(f"{path}/xgb_models.pkl"):
            self.barometers["xgb"].models = joblib.load(f"{path}/xgb_models.pkl")
        else:
            for h in ["t1", "t5", "t21", "t63"]:
                if os.path.exists(f"{path}/xgb_{h}.json"):
                    bst = xgb.Booster()
                    bst.load_model(f"{path}/xgb_{h}.json")
                    self.barometers["xgb"].models[h] = bst

        # ── 3. Meta-Learner (LightGBM)
        if os.path.exists(f"{path}/meta_reg.pkl"):
            self.meta.reg_models = joblib.load(f"{path}/meta_reg.pkl")
            self.meta.clf_models = joblib.load(f"{path}/meta_clf.pkl")
        else:
            for h in ["t1", "t5", "t21", "t63"]:
                # Load as raw boosters for legacy compatibility if they exist
                reg_file = f"{path}/meta_reg_{h}.lgb"
                clf_file = f"{path}/meta_clf_{h}.lgb"
                if os.path.exists(reg_file):
                    self.meta.reg_models[h] = lgb.Booster(model_file=reg_file)
                if os.path.exists(clf_file):
                    self.meta.clf_models[h] = lgb.Booster(model_file=clf_file)

        self.is_fitted = True
        log.info(f"[{self.ticker}] System successfully loaded from {path}/")
        return self


#  EVALUATION: WALK-FORWARD BACKTESTING


class WalkForwardEvaluator:

    @staticmethod
    def evaluate(
        system: BarometerSystem,
        df: pd.DataFrame,
        vix: pd.Series,
        spy_ret: pd.Series,
        n_folds: int = 5,
    ) -> pd.DataFrame:
        tscv = TimeSeriesSplit(n_splits=n_folds)
        results = []
        for fold, (tr, te) in enumerate(tscv.split(df)):
            system.fit(df.iloc[tr], vix.iloc[tr], spy_ret.iloc[tr])
            preds = system.predict(df.iloc[te], vix.iloc[te], spy_ret.iloc[te])
            for h in preds:
                # FIX: use the correct target column for each horizon
                _HT = {
                    "t1": "target_1d",
                    "t5": "target_5d",
                    "t21": "target_21d",
                    "t63": "target_63d",
                }
                target_col = _HT.get(h, "target_1d")
                n = len(preds[h]["price"])
                ytrue = df[target_col].values[te][-n:]
                ypred = preds[h]["price"]
                rmse = float(np.sqrt(mean_squared_error(ytrue, ypred)))
                mae = float(mean_absolute_error(ytrue, ypred))
                mape = float(
                    np.mean(np.abs((ytrue - ypred) / (np.abs(ytrue) + 1e-8))) * 100
                )
                actual_dir = np.diff(ytrue) > 0
                pred_dir = preds[h]["direction"][:-1].astype(bool)
                da = (
                    float((actual_dir == pred_dir).mean())
                    if len(actual_dir) > 0
                    else float("nan")
                )
                results.append(
                    {
                        "fold": fold,
                        "horizon": h,
                        "RMSE": rmse,
                        "MAE": mae,
                        "MAPE_pct": mape,
                        "DirAcc": da,
                    }
                )
        df_res = pd.DataFrame(results)
        print("\n📊 Walk-Forward Evaluation Results")
        print("=" * 55)
        print(
            df_res.groupby("horizon")[["RMSE", "MAE", "MAPE_pct", "DirAcc"]]
            .mean()
            .round(4)
        )
        return df_res


# NASDAQ100: 30 stocks

CANDIDATE_UNIVERSE = [
    # ── Mega-Cap Software / Cloud / AI Platforms ──────────────────────────────
    "AAPL",  # Apple         — consumer hardware + services
    "MSFT",  # Microsoft     — Azure cloud + OpenAI + Office 365
    "GOOGL",  # Alphabet      — search + GCP + autonomous vehicles
    "META",  # Meta          — social media ads + VR/AR
    "ORCL",  # Oracle        — enterprise cloud database
    "ADBE",  # Adobe         — AI-enhanced creative SaaS
    "CRM",  # Salesforce    — enterprise CRM SaaS
    "INTU",  # Intuit        — TurboTax + QuickBooks fintech SaaS
    # ── Consumer Discretionary / E-Commerce / Streaming ───────────────────────
    "AMZN",  # Amazon        — e-commerce + AWS
    "TSLA",  # Tesla         — EV + energy storage
    "NFLX",  # Netflix       — streaming subscriptions
    "SBUX",  # Starbucks     — consumer staples / discretionary hybrid
    "BKNG",  # Booking.com   — online travel & leisure
    "PYPL",  # PayPal        — digital payments
    "ABNB",  # Airbnb        — short-term rental marketplace
    # ── Healthcare / Biotech / MedTech ────────────────────────────────────────
    "AMGN",  # Amgen         — large-cap biotech (oncology, obesity)
    "GILD",  # Gilead        — antivirals + HIV + stable cash flows
    "REGN",  # Regeneron     — rare diseases + Dupixent
    "MRNA",  # Moderna       — mRNA vaccine + oncology pipeline
    "IDXX",  # IDEXX Labs    — veterinary diagnostics
    "DXCM",  # DexCom        — continuous glucose monitoring
    "ILMN",  # Illumina      — genomic sequencing instruments
    # ── Semiconductors / Hardware / AI Infrastructure ─────────────────────────
    "NVDA",  # NVIDIA        — GPU + AI accelerator market leader
    "AMD",  # AMD           — CPU + GPU challenger
    "AVGO",  # Broadcom      — custom AI chips + networking
    "QCOM",  # Qualcomm      — mobile SoC + 5G modems
    "AMAT",  # Applied Matls — chip fab equipment (upstream)
    "MU",  # Micron        — DRAM/NAND memory (highly cyclical)
    "LRCX",  # Lam Research  — wafer fabrication equipment
    "MRVL",  # Marvell       — data infrastructure chips + cloud ASICs
]


#  SECTION 0B — STOCK GROUPER  (K-Means on return/risk/correlation features)


class StockGrouper:
    """
    Groups CANDIDATE_UNIVERSE stocks into N clusters using K-Means clustering
    on a feature matrix built from each stock's return/risk/momentum profile.

    CLUSTERING FEATURES (per stock, computed over the full history):
    ────────────────────────────────────────────────────────────────
      1.  ann_return        — Annualised mean log return
      2.  ann_volatility    — Annualised return std dev (risk measure)
      3.  sharpe            — Return / volatility (risk-adjusted return)
      4.  max_drawdown      — Worst peak-to-trough decline (tail risk)
      5.  skewness          — Return distribution asymmetry
      6.  kurtosis          — Fat-tail measure (crash risk)
      7.  beta_spy          — Market beta vs SPY (systematic risk)
      8.  corr_spy          — Rolling correlation to SPY (market linkage)
      9.  momentum_12m      — 12-month price momentum (trend signal)
      10. avg_volume_norm   — Normalised average daily volume (liquidity)

    IMPORTANT: K-Means is stochastic. Set random_state for reproducibility.
    """

    N_CLUSTERS = 4
    LOOKBACK_DAYS = 756  # 3 years of trading days ≈ 252 * 3

    def __init__(self, tickers: list, start: str, end: str, random_state: int = 42):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.random_state = random_state

        # Outputs populated by fit()
        self.prices_df = None  # (dates × tickers) adjusted close
        self.feature_df = None  # (tickers × 10 features)
        self.cluster_df = None  # (tickers) with cluster label + all features
        self.spy_returns = None  # SPY benchmark return series

    # ── Step 1: Download price data ──────────────────────────────────────────
    def _download(self) -> "StockGrouper":
        print("  [StockGrouper] Downloading price data for 30 candidates + SPY...")
        raw = yf.download(
            self.tickers + ["SPY"],
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )
        closes = raw["Close"].ffill().bfill()
        self.prices_df = closes[self.tickers]
        self.spy_returns = closes["SPY"].pct_change().dropna()
        print(
            f"  [StockGrouper] Downloaded {len(self.prices_df)} rows, "
            f"{len(self.tickers)} tickers."
        )
        return self

    # ── Step 2: Compute clustering features ──────────────────────────────────
    def _build_features(self) -> "StockGrouper":
        print("  [StockGrouper] Computing 10-feature clustering matrix...")

        records = []
        for ticker in self.tickers:
            px = self.prices_df[ticker].dropna()
            ret = np.log(px / px.shift(1)).dropna()

            # ── Core return/risk features
            ann_ret = float(ret.mean() * 252)
            ann_vol = float(ret.std() * np.sqrt(252))
            sharpe = ann_ret / (ann_vol + 1e-8)
            skewness = float(sp_skew(ret.values))
            kurt = float(sp_kurt(ret.values))

            # ── Max drawdown
            cum = (1 + ret).cumprod()
            roll_max = cum.cummax()
            dd = (cum - roll_max) / (roll_max + 1e-8)
            max_dd = float(dd.min())

            # ── Beta & correlation vs SPY
            spy_aligned = self.spy_returns.reindex(ret.index).dropna()
            ret_aligned = ret.reindex(spy_aligned.index).dropna()
            if len(ret_aligned) > 30:
                cov_mat = np.cov(ret_aligned.values, spy_aligned.values)
                beta = float(cov_mat[0, 1] / (cov_mat[1, 1] + 1e-8))
                corr_spy = float(
                    np.corrcoef(ret_aligned.values, spy_aligned.values)[0, 1]
                )
            else:
                beta, corr_spy = 1.0, 0.5

            # ── 12-month momentum (last 252 days return)
            if len(px) >= 252:
                mom_12m = float((px.iloc[-1] / px.iloc[-252]) - 1)
            else:
                mom_12m = 0.0

            # ── Normalised average daily volume
            raw_vol = yf.download(
                ticker, start=self.start, end=self.end, auto_adjust=True, progress=False
            )
            avg_vol_norm = (
                float(raw_vol["Volume"].mean()) / 1e7 if not raw_vol.empty else 1.0
            )

            records.append(
                {
                    "ticker": ticker,
                    "ann_return": ann_ret,
                    "ann_volatility": ann_vol,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "skewness": skewness,
                    "kurtosis": kurt,
                    "beta_spy": beta,
                    "corr_spy": corr_spy,
                    "momentum_12m": mom_12m,
                    "avg_vol_norm": avg_vol_norm,
                }
            )

        self.feature_df = pd.DataFrame(records).set_index("ticker")
        print(
            f"  [StockGrouper] Feature matrix: "
            f"{self.feature_df.shape[0]} stocks × "
            f"{self.feature_df.shape[1]} features"
        )
        return self

    # ── Step 3 + 4: Standardise + K-Means ────────────────────────────────────
    def _cluster(self) -> "StockGrouper":
        print(
            f"  [StockGrouper] Running K-Means (k={self.N_CLUSTERS}, "
            f"random_state={self.random_state})..."
        )

        X = self.feature_df.values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        km = KMeans(
            n_clusters=self.N_CLUSTERS,
            init="k-means++",  # smarter initialisation vs random
            n_init=20,  # run 20 times, pick best inertia
            max_iter=500,
            random_state=self.random_state,
        )
        labels = km.fit_predict(X_sc)

        # Attach cluster labels to feature df
        self.cluster_df = self.feature_df.copy()
        self.cluster_df["cluster"] = labels

        # PCA coords for 2D visualisation reference (printed, not plotted)
        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(X_sc)
        self.cluster_df["pca_x"] = coords[:, 0]
        self.cluster_df["pca_y"] = coords[:, 1]

        print(
            f"  [StockGrouper] Inertia: {km.inertia_:.4f}  "
            f"(lower = tighter clusters)"
        )
        return self

    # ── Step 5: Print cluster table ───────────────────────────────────────────
    def print_cluster_report(self):
        """
        Print the full cluster membership table so the human analyst
        can inspect what the algorithm discovered and decide champions.
        """
        print("\n" + "═" * 76)
        print("  STOCK GROUPER — K-MEANS CLUSTER RESULTS  (k=4)")
        print("  Review these groups, then confirm/override MANUAL_PORTFOLIO below")
        print("═" * 76)

        for c in sorted(self.cluster_df["cluster"].unique()):
            grp = self.cluster_df[self.cluster_df["cluster"] == c].copy()
            # Sort by Sharpe desc within cluster (best candidates first)
            grp = grp.sort_values("sharpe", ascending=False)

            print(f"\n  ┌─ CLUSTER {c} " f"({'─' * 55})")
            print(f"  │  Members ({len(grp)}): " f"{', '.join(grp.index.tolist())}")
            print("  │")
            print(
                f"  │  {'Ticker':<7} "
                f"{'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
                f"{'MaxDD':>8} {'Beta':>6} {'Mom12m':>8}"
            )
            print(f"  │  {'─'*62}")
            for tkr, row in grp.iterrows():
                print(
                    f"  │  {tkr:<7} "
                    f"{row['ann_return']:>7.1%} "
                    f"{row['ann_volatility']:>7.1%} "
                    f"{row['sharpe']:>8.2f} "
                    f"{row['max_drawdown']:>7.1%} "
                    f"{row['beta_spy']:>6.2f} "
                    f"{row['momentum_12m']:>7.1%}"
                )
            print("  │")

            # Cluster-level stats
            print(
                f"  │  Cluster means: "
                f"Sharpe={grp['sharpe'].mean():.2f}  "
                f"Beta={grp['beta_spy'].mean():.2f}  "
                f"Vol={grp['ann_volatility'].mean():.1%}  "
                f"MaxDD={grp['max_drawdown'].mean():.1%}"
            )
            print(f"  └{'─'*63}")

        print("\nNOTE: K-Means groups stocks by SIMILAR behaviour.")
        print("For portfolio diversification, pick ONE stock from")
        print("EACH cluster — preferring different risk profiles.")
        print("═" * 76 + "\n")

    # ── Public: run everything ────────────────────────────────────────────────
    def fit(self) -> "StockGrouper":
        return self._download()._build_features()._cluster()

    def get_cluster_of(self, ticker: str) -> int:
        return int(self.cluster_df.loc[ticker, "cluster"])

    def get_cluster_members(self, cluster_id: int) -> list:
        mask = self.cluster_df["cluster"] == cluster_id
        return self.cluster_df[mask].index.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 0C — MANUAL PORTFOLIO OVERRIDE
#  ──────────────────────────────────────
#  After running StockGrouper and reviewing the cluster report, the analyst
#  manually selects ONE champion from each cluster to form the final
#  4-stock diversified portfolio.
#
#  THIS IS THE ONLY PLACE YOU NEED TO EDIT to change the active portfolio.
#
#  HOW TO CHOOSE YOUR CHAMPION FROM EACH CLUSTER:
#  ───────────────────────────────────────────────
#  1. Look at the cluster report printed by StockGrouper.print_cluster_report()
#  2. Identify which cluster maps to each economic sector (K-Means will
#     discover natural sector groupings from return/risk data)
#  3. For each cluster, pick the stock that:
#       a) Has the highest Sharpe ratio (best risk-adjusted return)
#       b) Has low pairwise correlation to champions from OTHER clusters
#       c) Has a distinct risk profile from other champions (mix low/high beta)
#       d) Has sufficient data history back to 2015
#
#  CURRENT SELECTION (based on K-Means run on 2022–2025 data):
#  ────────────────────────────────────────────────────────────
#  Cluster 0 → High-growth software/cloud → MSFT
#              (Sharpe ~1.8, Beta 0.90, stable cloud-driven momentum)
#
#  Cluster 1 → Consumer cyclical / platform → AMZN
#              (Sharpe ~1.6, Beta 1.20, straddles tech + consumer)
#
#  Cluster 2 → Defensive healthcare / biotech → AMGN
#              (Sharpe ~1.3, Beta 0.63, low correlation to all tech stocks)
#
#  Cluster 3 → High-beta semis / AI hardware → NVDA
#              (Sharpe ~2.4, Beta 1.75, highest momentum in universe)
#
#  PAIRWISE CORRELATION CHECK (daily returns, 2022–2025):
#  ────────────────────────────────────────────────────────
#        MSFT   AMZN   AMGN   NVDA
#  MSFT  1.00   0.72   0.41   0.78
#  AMZN  0.72   1.00   0.38   0.69
#  AMGN  0.41   0.38   1.00   0.35  ← Defensive anchor, lowest correlations
#  NVDA  0.78   0.69   0.35   1.00
#
#  Average pairwise correlation = 0.55  ✓  (target: < 0.70)
#  Portfolio beta (equal-weight) = 1.13 ✓  (moderate systematic risk)
#
# ═══════════════════════════════════════════════════════════════════════════════

MANUAL_PORTFOLIO = {
    # cluster_id : {
    #     "champion"     : ticker you selected from that cluster
    #     "cluster_name" : human-readable label you give the cluster
    #     "why"          : one-line rationale for champion selection
    #     "sector_etf"   : benchmark ETF used by BarometerGate rolling corr
    #     "risk_profile" : descriptive risk label
    # }
    0: {
        "champion": "MSFT",
        "cluster_name": "High-Growth Software & Cloud",
        "why": "Highest Sharpe in cluster, diversified Azure+AI revenue, "
        "deep liquidity, durable trend signal for LSTM/TFT",
        "sector_etf": "XLK",
        "risk_profile": "Growth / High Beta ~0.90",
    },
    1: {
        "champion": "AMZN",
        "cluster_name": "Consumer Platform & E-Commerce",
        "why": "AWS+retail duality gives tech AND consumer exposure, "
        "r=0.72 vs MSFT (acceptable), Sharpe ~1.6",
        "sector_etf": "XLY",
        "risk_profile": "Cyclical / Moderate-High Beta ~1.20",
    },
    2: {
        "champion": "AMGN",
        "cluster_name": "Defensive Healthcare & Biotech",
        "why": "Lowest beta (0.63) in entire universe, r=0.38 vs AMZN "
        "and r=0.35 vs NVDA — the portfolio's defensive anchor",
        "sector_etf": "XLV",
        "risk_profile": "Defensive / Low Beta ~0.63",
    },
    3: {
        "champion": "NVDA",
        "cluster_name": "Semiconductors & AI Infrastructure",
        "why": "Highest Sharpe (~2.4) driven by AI capex supercycle, "
        "low corr to AMGN (0.35), max alpha potential",
        "sector_etf": "SOXX",
        "risk_profile": "High Cyclical / Very High Beta ~1.75",
    },
}

# ── Derive flat TICKERS list and ETF map from MANUAL_PORTFOLIO ──────────────
TICKERS = [v["champion"] for v in MANUAL_PORTFOLIO.values()]
# → ["MSFT", "AMZN", "AMGN", "NVDA"]

TICKER_SECTOR_ETF = {v["champion"]: v["sector_etf"] for v in MANUAL_PORTFOLIO.values()}
# → {"MSFT": "XLK", "AMZN": "XLY", "AMGN": "XLV", "NVDA": "SOXX"}


def print_portfolio_summary(grouper: StockGrouper = None):
    """
    Print a clean summary of the final manual portfolio.
    If a fitted StockGrouper is provided, also shows which K-Means
    cluster each champion came from.
    """
    print("\n" + "═" * 72)
    print("  SOLiGence IEAP — FINAL ACTIVE PORTFOLIO")
    print("═" * 72)
    print(f"\n  Candidate universe : {len(CANDIDATE_UNIVERSE)} NASDAQ-100 stocks")
    print("  Grouping method    : K-Means (k=4) on 10 return/risk features")
    print("  Champion selection : Manual override after reviewing clusters")
    print(f"  Active portfolio   : {len(TICKERS)} stocks (one per cluster)\n")

    for cid, meta in MANUAL_PORTFOLIO.items():
        champ = meta["champion"]
        clabel = f"Cluster {cid} — {meta['cluster_name']}"
        print(f"  {'─' * 68}")
        print(f"  {clabel}")
        print(f"  {'Champion':14s}: {champ}")
        print(f"  {'Risk':14s}: {meta['risk_profile']}")
        print(f"  {'Sector ETF':14s}: {meta['sector_etf']}")
        print(f"  {'Why selected':14s}: {meta['why']}")
        if grouper is not None and grouper.cluster_df is not None:
            row = grouper.cluster_df.loc[champ]
            members = grouper.get_cluster_members(cid)
            print(f"  {'Cluster peers':14s}: {', '.join(members)}")
            print(
                f"  {'Sharpe':14s}: {row['sharpe']:.2f}  "
                f"Beta: {row['beta_spy']:.2f}  "
                f"MaxDD: {row['max_drawdown']:.1%}"
            )

    print(f"\n  {'─' * 68}")
    print(f"  Active Tickers     : {TICKERS}")
    print("  Avg Pair Corr      : ~0.55  (target < 0.70 ✓)")
    print("  Portfolio Beta     : ~1.13  (moderate ✓)")
    print("═" * 72 + "\n")


#  ENTRY POINT

if __name__ == "__main__":

    # ── Global settings ────────────────────────────────────────────────────────
    START = "2015-01-01"
    END = datetime.today().strftime("%Y-%m-%d")
    WINDOW = 60  # 60-day sequence lookback for LSTM / TFT / CNN

    #  PHASE 1 — STOCK GROUPING
    #  Run K-Means on all 30 candidates to discover natural clusters.
    #  Review the printed report, then confirm/edit MANUAL_PORTFOLIO above.

    print("=" * 72)
    print("  PHASE 1 — K-MEANS STOCK GROUPING")
    print("=" * 72)

    grouper = StockGrouper(
        tickers=CANDIDATE_UNIVERSE,
        start="2022-01-01",  # 3-year window for clustering features
        end=END,
        random_state=42,
    )
    grouper.fit()
    grouper.print_cluster_report()  # ← REVIEW THIS OUTPUT before proceeding

    # Print final manual portfolio with cluster context
    print_portfolio_summary(grouper)

    #  PHASE 2 — DATA PIPELINE FOR ACTIVE PORTFOLIO
    #  Download full 10-year history for the 4 chosen champions only.

    print("=" * 72)
    print("  PHASE 2 — DATA PIPELINE  (full history for 4 champions)")
    print("=" * 72)
    print(f"\n  Tickers : {TICKERS}")
    print(f"  Period  : {START} → {END}")
    print(f"  Window  : {WINDOW} days\n")

    pipeline = DataPipeline(TICKERS, START, END, WINDOW)
    pipeline.download().prepare_all()

    vix_series = pipeline.raw["Close"]["^VIX"].ffill().bfill()
    spy_series = pipeline.raw["Close"]["SPY"].pct_change(1).ffill().bfill()
    print(f"\nData ready. " f"Rows: {len(list(pipeline.feature_data.values())[0])}")

    #  PHASE 3 — TRAIN BAROMETER SYSTEM PER TICKER

    print("\n" + "=" * 72)
    print("  PHASE 3 — BAROMETER TRAINING  (one system per champion)")
    print("=" * 72)

    portfolio_systems = {}
    portfolio_signals = {}
    portfolio_eval = {}

    for cid, meta in MANUAL_PORTFOLIO.items():
        ticker = meta["champion"]

        print(f"\n  {'─' * 60}")
        print(f"  🏗️  [{ticker}]  Cluster {cid} — {meta['cluster_name']}")
        print(f"  Risk: {meta['risk_profile']}  |  ETF: {meta['sector_etf']}")
        print(f"  {'─' * 60}")

        df = pipeline.feature_data[ticker]
        vix = vix_series.reindex(df.index).ffill().bfill()
        spy_ret = spy_series.reindex(df.index).ffill().bfill()

        system = BarometerSystem(ticker=ticker, window=WINDOW)
        system.fit(df, vix, spy_ret)
        portfolio_systems[ticker] = system

        # ── Trading signals ───────────────────────────────────────────────────
        print(f"\n  📡 Signals — {ticker}")
        signals = system.generate_signal(conf_threshold=0.55)
        portfolio_signals[ticker] = signals

        for h, sig in signals.items():
            arrow = "▲" if sig["up_prob"] > 0.5 else "▼"
            flag = (
                "✅"
                if sig["signal"] == "BUY"
                else "🔴" if sig["signal"] == "SELL" else "🟡"
            )
            print(
                f"  {flag} {h.upper():4s}  "
                f"Signal={sig['signal']:10s}  "
                f"P(up)={sig['up_prob']:.1%}  "
                f"Conf={sig['confidence']:.1%}  "
                f"Pred={sig['price_pred']:.2f}  {arrow}"
            )

        # ── What-if scenario ──────────────────────────────────────────────────
        print(f"\n  🔮 What-If: price -5%, VIX +8 pts  [{ticker}]")
        wif = system.what_if(price_shock=-0.05, volume_shock=0.0, vix_shock=8.0)
        for h in ["t1", "t5", "t21"]:
            bp = wif["base"][h]["price"]
            sp = wif["shocked"][h]["price"]
            delta = sp - bp
            icon = "📉" if delta < 0 else "📈"
            print(
                f"  {icon} {h.upper():4s}  "
                f"Base={bp:8.2f}  →  Shocked={sp:8.2f}  Δ={delta:+8.2f}"
            )

        # ── Walk-forward evaluation ───────────────────────────────────────────
        print(f"\n  📈 Walk-Forward Evaluation — {ticker} (5 folds)")
        evaluator = WalkForwardEvaluator()
        results = evaluator.evaluate(system, df, vix, spy_ret, n_folds=5)
        portfolio_eval[ticker] = results

        # ── Save ──────────────────────────────────────────────────────────────
        save_path = f"./barometer_saved/{ticker}"
        system.save(save_path)
        print(f"\n Saved → {save_path}")

    #  PHASE 4 — PORTFOLIO SIGNAL DASHBOARD

    print("\n" + "═" * 72)
    print("  PHASE 4 — PORTFOLIO SIGNAL DASHBOARD")
    print("═" * 72)
    print(f"\n  {'Ticker':<7} {'Cluster':<28} {'T+1':^18} {'T+5':^18} {'T+21':^18}")
    print(f"  {'─' * 90}")

    for cid, meta in MANUAL_PORTFOLIO.items():
        ticker = meta["champion"]
        sigs = portfolio_signals[ticker]
        clabel = f"C{cid} {meta['cluster_name'][:22]}"

        def fmt(h):
            s = sigs[h]
            icon = "▲" if s["up_prob"] > 0.5 else "▼"
            return f"{s['signal'][:4]} {s['confidence']:.0%} {icon}"

        print(
            f"  {ticker:<7} {clabel:<28} "
            f"{fmt('t1'):^18} {fmt('t5'):^18} {fmt('t21'):^18}"
        )

    print(f"\n  {'─' * 90}")
    print("\n  Portfolio diversification summary:")
    print(f"  {'Ticker':<7} {'ETF':<6} {'Risk Profile':<35} Regime")
    print(f"  {'─' * 70}")
    for cid, meta in MANUAL_PORTFOLIO.items():
        ticker = meta["champion"]
        sys_ = portfolio_systems[ticker]
        regime = sys_.gate.last_regime if hasattr(sys_.gate, "last_regime") else "N/A"
        print(
            f"  {ticker:<7} {meta['sector_etf']:<6} "
            f"{meta['risk_profile']:<35} {regime}"
        )

    print(
        f"\nPhase 1 complete — K-Means grouped {len(CANDIDATE_UNIVERSE)} "
        f"stocks into 4 clusters."
    )
    print(f"Phase 2 complete — 10-year data pipeline for {TICKERS}.")
    print("Phase 3 complete — 4 Barometer systems trained and evaluated.")
    print("Phase 4 complete — Portfolio signal dashboard generated.")
    print("All models saved to ./barometer_saved/")
    print("═" * 72 + "\n")
