"""
============================================================
  SOLIGENCE IEAP — FINBERT SENTIMENT PIPELINE
  IBM Data Science Team | AE2 Sentiment Extension
============================================================

MODULE PURPOSE
==============
  This module is a SELF-CONTAINED extension to barometer_core.py.
  It does NOT modify any existing code — it produces a sentiment
  DataFrame that is merged into DataPipeline.engineer_features()
  via a single one-line call.

WHAT THIS MODULE DOES
=====================
  1. Fetches financial news headlines from THREE sources:
       • NewsAPI        — mainstream financial news (WSJ, Reuters, Bloomberg)
       • Reddit PRAW    — r/wallstreetbets, r/investing, r/stocks
       • yfinance news  — Yahoo Finance built-in ticker news (free, no key needed)

  2. Runs each headline through FinBERT:
       • ProsusAI/finbert — financial domain fine-tuned BERT model
       • Outputs: positive / negative / neutral probability per headline

  3. Aggregates sentiment per trading day into 6 features:
       • sent_score      — weighted mean (positive - negative)
       • sent_positive   — mean positive probability
       • sent_negative   — mean negative probability
       • sent_neutral    — mean neutral probability
       • sent_volume     — number of headlines that day (news intensity)
       • sent_momentum   — 3-day rolling change in score (sentiment shift)

  4. Returns a DatetimeIndex DataFrame that merges directly into
     the existing feature matrix with zero lookahead bias.

INSTALL
=======
  pip install transformers torch newsapi-python praw

API KEYS REQUIRED
=================
  NewsAPI  : https://newsapi.org/register  (free tier: 100 req/day)
  Reddit   : https://www.reddit.com/prefs/apps  (free, create "script" app)

  Set as environment variables OR pass directly to SentimentConfig:
    export NEWSAPI_KEY="your_key_here"
    export REDDIT_CLIENT_ID="your_id"
    export REDDIT_CLIENT_SECRET="your_secret"

USAGE IN barometer_core.py
===========================
  # In DataPipeline.engineer_features(), after building df, add ONE line:
  df = SentimentPipeline(config).enrich(df, ticker, start, end)
  # That's it. The sentiment columns flow into all 4 base models automatically.

LOOKAHEAD BIAS PROTECTION
==========================
  All sentiment scores are assigned to date T using headlines published
  BEFORE market open on day T (i.e. headlines from T-1 evening + T morning
  up to 09:25 ET). Headlines published after market open are assigned to T+1.
  This strictly prevents lookahead bias.

FALLBACK BEHAVIOUR
==================
  If API keys are missing or API calls fail, the pipeline returns a DataFrame
  of zeros for all sentiment columns. The system continues running normally —
  sentiment is additive, never blocking.
"""

import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger("SentimentPipeline")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SentimentConfig:
    """
    Central configuration for the sentiment pipeline.
    All API keys and pipeline settings live here — edit once, applies everywhere.
    """

    # ── API Keys (read from env vars by default) ──────────────────────────────
    newsapi_key:          Optional[str] = field(
        default_factory=lambda: os.getenv("NEWSAPI_KEY", "")
    )
    reddit_client_id:     Optional[str] = field(
        default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", "")
    )
    reddit_client_secret: Optional[str] = field(
        default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", "")
    )
    reddit_user_agent:    str = "BarometerSentiment/2.0 (SOLiGence IEAP AE2)"

    # ── FinBERT model ─────────────────────────────────────────────────────────
    finbert_model: str = "ProsusAI/finbert"   # HuggingFace model ID
    batch_size:    int = 32                    # headlines per inference batch
    max_length:    int = 128                   # token limit (FinBERT max = 512)
    device:        str = "cpu"                 # "cuda" if GPU available

    # ── News fetch settings ───────────────────────────────────────────────────
    newsapi_page_size:   int = 100    # max per request (NewsAPI free tier limit)
    reddit_post_limit:   int = 100    # posts per subreddit per fetch
    reddit_subreddits:   list = field(default_factory=lambda: [
        "wallstreetbets", "investing", "stocks", "SecurityAnalysis", "finance"
    ])
    market_open_hour:    int = 9      # ET — headlines after this → assigned T+1
    market_open_minute:  int = 25

    # ── Cache settings ────────────────────────────────────────────────────────
    cache_dir:     str  = "./sentiment_cache"   # local parquet cache
    use_cache:     bool = True                  # skip API calls if cache exists

    # ── Sentiment aggregation ─────────────────────────────────────────────────
    sentiment_window: int = 3   # days for rolling momentum calculation
    min_headlines:    int = 1   # minimum headlines to compute score (else NaN→0)


# ═══════════════════════════════════════════════════════════════════════════════
#  FINBERT INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FinBERTScorer:
    """
    Wraps the ProsusAI/finbert model for batch inference.

    FinBERT is a BERT model fine-tuned on ~10,000 financial news sentences
    from Reuters, The Motley Fool, and financial analyst reports.
    It outputs three probabilities: positive, negative, neutral.

    Reference: Araci, D. (2019). FinBERT: Financial Sentiment Analysis
               with Pre-trained Language Models. arXiv:1908.10063
    """

    def __init__(self, config: SentimentConfig):
        self.config  = config
        self._model  = None
        self._tok    = None
        self._loaded = False

    def _load(self):
        """Lazy load — only imports transformers when actually called."""
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            self._torch = torch
            log.info(f"Loading FinBERT: {self.config.finbert_model}")
            self._tok   = AutoTokenizer.from_pretrained(self.config.finbert_model)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.finbert_model
            )
            self._model.eval()
            self._model.to(self.config.device)
            self._loaded = True
            log.info("FinBERT loaded ✓")
        except ImportError:
            log.error("transformers/torch not installed. Run: pip install transformers torch")
            raise
        except Exception as e:
            log.error(f"FinBERT load failed: {e}")
            raise

    def score_batch(self, texts: list[str]) -> pd.DataFrame:
        """
        Scores a list of headline strings.

        Returns:
          DataFrame with columns [positive, negative, neutral]
          One row per input text.
          All values are probabilities summing to 1.0.
        """
        self._load()
        results = []
        import torch

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i: i + self.config.batch_size]
            # Tokenise
            enc = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.config.device)

            with torch.no_grad():
                logits = self._model(**enc).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()

            # FinBERT label order: positive=0, negative=1, neutral=2
            for row in probs:
                results.append({
                    "positive": float(row[0]),
                    "negative": float(row[1]),
                    "neutral":  float(row[2])
                })

        return pd.DataFrame(results)

    def score_single(self, text: str) -> dict:
        """Convenience wrapper for a single headline."""
        return self.score_batch([text]).iloc[0].to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
#  NEWS FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

class NewsAPIFetcher:
    """
    Fetches financial news from NewsAPI.org.
    Free tier: 100 requests/day, headlines up to 30 days back.
    Paid tier: full historical archive.

    Targets financial sources: wsj.com, reuters.com, bloomberg.com,
    ft.com, cnbc.com, marketwatch.com, seekingalpha.com
    """

    FINANCIAL_SOURCES = (
        "the-wall-street-journal,reuters,bloomberg,financial-times,"
        "cnbc,marketwatch,business-insider,fortune,the-verge"
    )

    def __init__(self, config: SentimentConfig):
        self.config = config

    def fetch(self, query: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Args:
          query     : ticker symbol or company name (e.g. "MSFT Microsoft")
          from_date : "YYYY-MM-DD"
          to_date   : "YYYY-MM-DD"

        Returns:
          DataFrame [published_at, title, description, source]
          published_at is timezone-aware UTC datetime.
        """
        if not self.config.newsapi_key:
            log.warning("NEWSAPI_KEY not set — skipping NewsAPI fetch")
            return pd.DataFrame()

        try:
            from newsapi import NewsApiClient
            client = NewsApiClient(api_key=self.config.newsapi_key)
            rows   = []

            # Paginate through results
            for page in range(1, 6):   # max 5 pages × 100 = 500 headlines
                resp = client.get_everything(
                    q            = query,
                    sources      = self.FINANCIAL_SOURCES,
                    from_param   = from_date,
                    to           = to_date,
                    language     = "en",
                    sort_by      = "publishedAt",
                    page_size    = self.config.newsapi_page_size,
                    page         = page
                )
                articles = resp.get("articles", [])
                if not articles:
                    break
                for art in articles:
                    rows.append({
                        "published_at": pd.to_datetime(art["publishedAt"]),
                        "title":        art.get("title", "") or "",
                        "description":  art.get("description", "") or "",
                        "source":       art.get("source", {}).get("name", ""),
                        "data_source":  "newsapi"
                    })
                time.sleep(0.2)   # polite rate limiting

            log.info(f"NewsAPI: fetched {len(rows)} articles for '{query}'")
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        except Exception as e:
            log.warning(f"NewsAPI fetch failed: {e}")
            return pd.DataFrame()


class RedditFetcher:
    """
    Fetches posts from financial subreddits using PRAW (Python Reddit API Wrapper).

    Why Reddit?
    ───────────
    r/wallstreetbets and r/investing are proven leading indicators for
    retail sentiment shifts, particularly for high-profile NASDAQ-100 names.
    Academic research (Bollen et al. 2011, Oliveira et al. 2017) shows
    social media sentiment has statistically significant predictive power
    for next-day returns, especially for stocks with high retail ownership.

    Free tier: 1,000 requests/10 min. Historical data: up to ~1,000 posts
    per subreddit. For full historical data, use PushShift API instead.
    """

    def __init__(self, config: SentimentConfig):
        self.config = config

    def _build_client(self):
        try:
            import praw
            return praw.Reddit(
                client_id     = self.config.reddit_client_id,
                client_secret = self.config.reddit_client_secret,
                user_agent    = self.config.reddit_user_agent
            )
        except ImportError:
            log.error("praw not installed. Run: pip install praw")
            raise

    def fetch(self, ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Searches each configured subreddit for posts mentioning the ticker.

        Args:
          ticker    : stock ticker (e.g. "MSFT")
          from_date : "YYYY-MM-DD"
          to_date   : "YYYY-MM-DD"

        Returns:
          DataFrame [published_at, title, description, source, score, upvote_ratio]
          score = Reddit upvote count (proxy for post reach/virality)
        """
        if not self.config.reddit_client_id or not self.config.reddit_client_secret:
            log.warning("Reddit credentials not set — skipping Reddit fetch")
            return pd.DataFrame()

        try:
            reddit  = self._build_client()
            rows    = []
            ts_from = pd.Timestamp(from_date).timestamp()
            ts_to   = pd.Timestamp(to_date).timestamp()

            for sub_name in self.config.reddit_subreddits:
                sub = reddit.subreddit(sub_name)
                # Search for ticker mentions
                for post in sub.search(
                    ticker,
                    limit   = self.config.reddit_post_limit,
                    sort    = "new",
                    time_filter = "year"
                ):
                    if ts_from <= post.created_utc <= ts_to:
                        rows.append({
                            "published_at": pd.to_datetime(
                                post.created_utc, unit="s", utc=True
                            ),
                            "title":        post.title,
                            "description":  post.selftext[:500],   # first 500 chars
                            "source":       f"reddit/r/{sub_name}",
                            "score":        post.score,
                            "upvote_ratio": post.upvote_ratio,
                            "data_source":  "reddit"
                        })
                time.sleep(0.5)   # Reddit rate limit: polite delay

            log.info(f"Reddit: fetched {len(rows)} posts for '{ticker}'")
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        except Exception as e:
            log.warning(f"Reddit fetch failed: {e}")
            return pd.DataFrame()


class YahooNewsFetcher:
    """
    Fetches news directly from yfinance — no API key required.
    yfinance.Ticker.news returns the most recent ~20 news items.
    This is the fallback source that always works.

    Limitation: only returns recent headlines (~1 week), not historical.
    For historical sentiment, NewsAPI or Reddit are required.
    """

    def fetch(self, ticker: str) -> pd.DataFrame:
        """
        Returns recent Yahoo Finance news for a ticker.
        No date filtering — returns whatever yfinance provides.
        """
        try:
            import yfinance as yf
            t    = yf.Ticker(ticker)
            news = t.news
            if not news:
                return pd.DataFrame()

            rows = []
            for item in news:
                # yfinance news item keys: uuid, title, publisher, link,
                #                          providerPublishTime, type, thumbnail
                rows.append({
                    "published_at": pd.to_datetime(
                        item.get("providerPublishTime", 0), unit="s", utc=True
                    ),
                    "title":       item.get("title", "") or "",
                    "description": "",   # yfinance doesn't provide body text
                    "source":      item.get("publisher", "Yahoo Finance"),
                    "data_source": "yahoo"
                })

            log.info(f"Yahoo Finance: fetched {len(rows)} articles for '{ticker}'")
            return pd.DataFrame(rows)

        except Exception as e:
            log.warning(f"Yahoo news fetch failed: {e}")
            return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentAggregator:
    """
    Aggregates per-headline FinBERT scores into daily trading features.

    Lookahead bias prevention:
    ──────────────────────────
    Headlines published BEFORE market open (09:25 ET) on day T are assigned
    to trading day T (they reflect pre-market sentiment that drives the open).
    Headlines published AFTER 09:25 ET on day T are assigned to day T+1
    (they reflect intraday news whose impact is seen at the NEXT open).

    This mirrors how a real trading desk would use news:
      "What did I know before the market opened this morning?"
    """

    def __init__(self, config: SentimentConfig):
        self.config = config

    def _assign_trading_day(self, published_at: pd.Series,
                             trading_days: pd.DatetimeIndex) -> pd.Series:
        """
        Maps each headline's publication timestamp to its correct trading day.
        Uses market open cutoff (09:25 ET) as the boundary.
        """
        # Convert all timestamps to US Eastern time
        try:
            et_times = published_at.dt.tz_convert("US/Eastern")
        except Exception:
            et_times = published_at.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")

        cutoff_hour   = self.config.market_open_hour
        cutoff_minute = self.config.market_open_minute

        assigned = []
        for ts in et_times:
            # Date of this headline in ET
            date = ts.date()
            # If published after market open → assign to next trading day
            if ts.hour > cutoff_hour or (ts.hour == cutoff_hour
                                          and ts.minute >= cutoff_minute):
                date = date + timedelta(days=1)
            # Find nearest trading day (forward fill to skip weekends/holidays)
            target = pd.Timestamp(date)
            # Find the first trading day >= target
            future = trading_days[trading_days >= target]
            assigned.append(future[0] if len(future) > 0 else pd.NaT)

        return pd.Series(assigned, index=published_at.index)

    def aggregate(self, headlines_df: pd.DataFrame,
                  scores_df: pd.DataFrame,
                  trading_days: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Combines headlines + FinBERT scores into a daily sentiment DataFrame.

        Args:
          headlines_df  : DataFrame with [published_at, title, source, ...]
          scores_df     : DataFrame with [positive, negative, neutral] per headline
          trading_days  : Index of valid trading days from the price data

        Returns:
          DataFrame indexed by trading_days with columns:
            sent_score, sent_positive, sent_negative, sent_neutral,
            sent_volume, sent_momentum
        """
        # Build result skeleton — one row per trading day, default zeros
        result = pd.DataFrame(index=trading_days)
        result.index = pd.DatetimeIndex(result.index).tz_localize(None)

        if headlines_df.empty or scores_df.empty:
            log.warning("No headlines to aggregate — returning zero sentiment")
            for col in ["sent_score", "sent_positive", "sent_negative",
                        "sent_neutral", "sent_volume", "sent_momentum"]:
                result[col] = 0.0
            return result

        # Merge headlines with scores
        df = headlines_df.reset_index(drop=True).copy()
        df = pd.concat([df, scores_df.reset_index(drop=True)], axis=1)

        # Assign each headline to a trading day
        df["trading_day"] = self._assign_trading_day(
            df["published_at"], trading_days
        )
        df = df.dropna(subset=["trading_day"])
        df["trading_day"] = pd.DatetimeIndex(df["trading_day"]).tz_localize(None)

        # Compute sentiment score = positive - negative (range: [-1, +1])
        df["sent_score_raw"] = df["positive"] - df["negative"]

        # Weight Reddit posts by upvote count (virality weighting)
        # Higher upvotes = more people saw and agreed with this post
        if "score" in df.columns:
            # Normalise upvote score to [1, 5] weight range
            max_score = df["score"].fillna(0).max()
            if max_score > 0:
                df["weight"] = 1 + 4 * (df["score"].fillna(0) / max_score)
            else:
                df["weight"] = 1.0
        else:
            df["weight"] = 1.0

        # Daily aggregation with weighted mean
        def weighted_mean(grp, col):
            w = grp["weight"].values
            v = grp[col].values
            return np.average(v, weights=w)

        daily = df.groupby("trading_day").apply(
            lambda g: pd.Series({
                "sent_score":    weighted_mean(g, "sent_score_raw"),
                "sent_positive": weighted_mean(g, "positive"),
                "sent_negative": weighted_mean(g, "negative"),
                "sent_neutral":  weighted_mean(g, "neutral"),
                "sent_volume":   len(g),          # headline count = news intensity
            })
        )

        # Merge into trading day skeleton
        result = result.join(daily, how="left")
        result = result.fillna({
            "sent_score":    0.0,
            "sent_positive": 0.333,   # neutral default
            "sent_negative": 0.333,
            "sent_neutral":  0.333,
            "sent_volume":   0.0
        })

        # Sentiment momentum: 3-day rolling change in score
        # Captures shifts in market narrative (e.g. positive → negative = warning)
        result["sent_momentum"] = (
            result["sent_score"]
            .rolling(self.config.sentiment_window, min_periods=1)
            .mean()
            .diff(1)
            .fillna(0)
        )

        # Clip extremes — same 1%/99% policy as DataPipeline.clean()
        for col in ["sent_score", "sent_momentum"]:
            lo, hi = result[col].quantile(0.01), result[col].quantile(0.99)
            result[col] = result[col].clip(lo, hi)

        log.info(f"Aggregated sentiment: {len(result)} trading days, "
                 f"avg {result['sent_volume'].mean():.1f} headlines/day")
        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  CACHE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentCache:
    """
    Parquet-based local cache for scored headlines.
    Avoids re-running FinBERT inference and API calls on reruns.

    Cache key: {ticker}_{start}_{end}.parquet
    Stored in: config.cache_dir/
    """

    def __init__(self, config: SentimentConfig):
        self.config   = config
        self.cache_dir = config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, ticker: str, start: str, end: str) -> str:
        safe_start = start.replace("-", "")
        safe_end   = end.replace("-", "")
        return os.path.join(self.cache_dir, f"{ticker}_{safe_start}_{safe_end}.parquet")

    def exists(self, ticker: str, start: str, end: str) -> bool:
        return os.path.exists(self._path(ticker, start, end))

    def load(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        path = self._path(ticker, start, end)
        log.info(f"Loading sentiment cache: {path}")
        return pd.read_parquet(path)

    def save(self, df: pd.DataFrame, ticker: str, start: str, end: str):
        path = self._path(ticker, start, end)
        df.to_parquet(path)
        log.info(f"Saved sentiment cache: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentPipeline:
    """
    Top-level orchestrator for the FinBERT sentiment pipeline.

    This is the ONLY class you need to interact with from barometer_core.py.
    Everything else in this module is internal.

    Design principles:
    ──────────────────
    • ADDITIVE: adds columns to existing DataFrame, never removes anything
    • GRACEFUL DEGRADATION: if any source fails, others continue; if all fail,
      zeros are returned and the system keeps running
    • CACHED: results stored locally so FinBERT doesn't re-run on each training call
    • LOOKAHEAD-FREE: strict pre-market cutoff assignment
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config     = config or SentimentConfig()
        self.scorer     = FinBERTScorer(self.config)
        self.aggregator = SentimentAggregator(self.config)
        self.cache      = SentimentCache(self.config)

        self._newsapi = NewsAPIFetcher(self.config)
        self._reddit  = RedditFetcher(self.config)
        self._yahoo   = YahooNewsFetcher()

    # ── Public API ─────────────────────────────────────────────────────────────

    def enrich(self, df: pd.DataFrame, ticker: str,
               start: str, end: str) -> pd.DataFrame:
        """
        Main entry point. Adds 6 sentiment columns to an existing feature DataFrame.

        Args:
          df     : existing feature DataFrame from DataPipeline.engineer_features()
                   Must have a DatetimeIndex.
          ticker : stock ticker (e.g. "MSFT")
          start  : data start date "YYYY-MM-DD"
          end    : data end date   "YYYY-MM-DD"

        Returns:
          The same df with 6 new columns appended:
            sent_score, sent_positive, sent_negative, sent_neutral,
            sent_volume, sent_momentum

        Usage in barometer_core.py:
          # Add this ONE line at the end of engineer_features(), before return:
          df = SentimentPipeline(config).enrich(df, ticker, self.start, self.end)
        """
        log.info(f"[{ticker}] Starting sentiment enrichment: {start} → {end}")

        # Check cache first
        if self.config.use_cache and self.cache.exists(ticker, start, end):
            sentiment_df = self.cache.load(ticker, start, end)
            log.info(f"[{ticker}] Loaded sentiment from cache")
        else:
            sentiment_df = self._build_sentiment(ticker, start, end, df.index)
            if self.config.use_cache:
                self.cache.save(sentiment_df, ticker, start, end)

        # Merge sentiment into feature DataFrame
        # Use left join on index — trading days in df are the authority
        df_index_naive = df.index.tz_localize(None) if df.index.tz else df.index
        sentiment_df.index = pd.DatetimeIndex(sentiment_df.index).tz_localize(None)

        df = df.copy()
        df.index = df_index_naive

        sentiment_cols = ["sent_score", "sent_positive", "sent_negative",
                          "sent_neutral", "sent_volume", "sent_momentum"]

        for col in sentiment_cols:
            if col in sentiment_df.columns:
                df[col] = sentiment_df[col].reindex(df.index).fillna(0)
            else:
                df[col] = 0.0

        log.info(f"[{ticker}] Sentiment enrichment complete — "
                 f"added {len(sentiment_cols)} features")
        return df

    def score_live(self, ticker: str) -> dict:
        """
        Fetches and scores the most recent headlines for live signal generation.
        Called at inference time (generate_signal) to get current sentiment.

        Returns:
          dict with keys: sent_score, sent_positive, sent_negative,
                          sent_neutral, sent_volume, timestamp
        """
        log.info(f"[{ticker}] Fetching live sentiment")
        today = datetime.today().strftime("%Y-%m-%d")
        week_ago = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

        headlines = self._fetch_all(ticker, week_ago, today)
        if headlines.empty:
            log.warning(f"[{ticker}] No live headlines found — returning neutral")
            return {
                "sent_score":    0.0,
                "sent_positive": 0.333,
                "sent_negative": 0.333,
                "sent_neutral":  0.333,
                "sent_volume":   0,
                "timestamp":     today
            }

        texts  = self._build_texts(headlines)
        scores = self._run_finbert(texts)

        # Most recent 24h only for live signal
        recent_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
        if "published_at" in headlines.columns:
            mask     = headlines["published_at"] >= recent_cutoff
            recent   = scores[mask.values] if mask.any() else scores
        else:
            recent = scores

        return {
            "sent_score":    float((recent["positive"] - recent["negative"]).mean()),
            "sent_positive": float(recent["positive"].mean()),
            "sent_negative": float(recent["negative"].mean()),
            "sent_neutral":  float(recent["neutral"].mean()),
            "sent_volume":   len(recent),
            "timestamp":     today
        }

    # ── Internal Methods ───────────────────────────────────────────────────────

    def _build_sentiment(self, ticker: str, start: str, end: str,
                          trading_days: pd.DatetimeIndex) -> pd.DataFrame:
        """Fetches all sources, runs FinBERT, aggregates to daily features."""
        headlines = self._fetch_all(ticker, start, end)

        if headlines.empty:
            log.warning(f"[{ticker}] No headlines found from any source")
            # Return zeros for all trading days
            result = pd.DataFrame(index=trading_days)
            for col in ["sent_score", "sent_positive", "sent_negative",
                        "sent_neutral", "sent_volume", "sent_momentum"]:
                result[col] = 0.0
            return result

        texts  = self._build_texts(headlines)
        scores = self._run_finbert(texts)

        return self.aggregator.aggregate(headlines, scores, trading_days)

    def _fetch_all(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetches from all three sources and combines.
        Each source is tried independently — one failure doesn't stop others.
        """
        dfs = []

        # Source 1: NewsAPI (most comprehensive financial coverage)
        # Query uses both ticker symbol and common company name for better recall
        company_name = self._ticker_to_name(ticker)
        query        = f"{ticker} OR {company_name}"
        news_df      = self._newsapi.fetch(query, start, end)
        if not news_df.empty:
            dfs.append(news_df)
            log.info(f"[{ticker}] NewsAPI: {len(news_df)} articles")

        # Source 2: Reddit (retail sentiment signal)
        reddit_df = self._reddit.fetch(ticker, start, end)
        if not reddit_df.empty:
            dfs.append(reddit_df)
            log.info(f"[{ticker}] Reddit: {len(reddit_df)} posts")

        # Source 3: Yahoo Finance (always available, recent only)
        yahoo_df = self._yahoo.fetch(ticker)
        if not yahoo_df.empty:
            dfs.append(yahoo_df)
            log.info(f"[{ticker}] Yahoo: {len(yahoo_df)} articles")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        # Deduplicate by title (same story from multiple sources)
        combined = combined.drop_duplicates(subset=["title"])
        combined = combined.dropna(subset=["title"])
        combined = combined[combined["title"].str.len() > 10]   # filter junk

        log.info(f"[{ticker}] Total unique headlines: {len(combined)}")
        return combined

    def _build_texts(self, headlines: pd.DataFrame) -> list[str]:
        """
        Builds the text to score for each headline.
        Uses title + description when available for richer context.
        FinBERT is sensitive to financial terminology so more context = better.
        """
        texts = []
        for _, row in headlines.iterrows():
            title = str(row.get("title", ""))
            desc  = str(row.get("description", ""))
            # Combine title and description, truncated to avoid exceeding max_length
            text  = title if not desc or desc == "nan" else f"{title}. {desc}"
            texts.append(text[:400])    # ~400 chars ≈ 80 tokens, safe for FinBERT
        return texts

    def _run_finbert(self, texts: list[str]) -> pd.DataFrame:
        """
        Runs FinBERT inference with error handling.
        Returns neutral 0.333/0.333/0.333 if model fails.
        """
        if not texts:
            return pd.DataFrame()
        try:
            return self.scorer.score_batch(texts)
        except Exception as e:
            log.error(f"FinBERT inference failed: {e}")
            # Return neutral scores — system continues without sentiment
            return pd.DataFrame({
                "positive": [0.333] * len(texts),
                "negative": [0.333] * len(texts),
                "neutral":  [0.333] * len(texts)
            })

    @staticmethod
    def _ticker_to_name(ticker: str) -> str:
        """
        Maps ticker symbols to company names for richer news queries.
        Covers the MANUAL_PORTFOLIO tickers from barometer_core.py.
        """
        names = {
            "MSFT":  "Microsoft",
            "AMZN":  "Amazon",
            "AMGN":  "Amgen",
            "NVDA":  "Nvidia",
            "AAPL":  "Apple",
            "GOOGL": "Google Alphabet",
            "META":  "Meta Facebook",
            "TSLA":  "Tesla",
            "NFLX":  "Netflix",
            "ADBE":  "Adobe",
            "CRM":   "Salesforce",
            "QCOM":  "Qualcomm",
            "INTC":  "Intel",
            "AMD":   "AMD Advanced Micro",
            "COST":  "Costco",
        }
        return names.get(ticker.upper(), ticker)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION INSTRUCTIONS FOR barometer_core.py
# ═══════════════════════════════════════════════════════════════════════════════
"""
STEP 1 — Add import at top of barometer_core.py (after existing imports):
─────────────────────────────────────────────────────────────────────────
    from sentiment_pipeline import SentimentPipeline, SentimentConfig

STEP 2 — Add config to DataPipeline.__init__():
────────────────────────────────────────────────
    def __init__(self, tickers, start, end, window=60,
                 sentiment_config: SentimentConfig = None):
        ...existing code...
        self.sentiment_config = sentiment_config or SentimentConfig()
        self._sentiment       = SentimentPipeline(self.sentiment_config)

STEP 3 — Add ONE line at end of DataPipeline.engineer_features(),
          just before   return self.clean(df):
──────────────────────────────────────────────────────────────────
    df = self._sentiment.enrich(df, ticker, self.start, self.end)

STEP 4 — Update BarometerSystem._feature_cols_from() exclude set:
──────────────────────────────────────────────────────────────────
    exclude = {"target_1d", "target_5d", "target_21d", "target_63d",
               "dir_1d", "dir_5d", "dir_63d"}
    # sent_* columns are NOT excluded — they flow into all base models

STEP 5 — (Optional) Enrich generate_signal() with live sentiment:
──────────────────────────────────────────────────────────────────
    def generate_signal(self, conf_threshold=0.60):
        live_sent = self._sentiment.score_live(self.ticker)
        # Log it alongside signal output
        log.info(f"Live sentiment score: {live_sent['sent_score']:+.3f} "
                 f"({live_sent['sent_volume']} headlines)")
        ...rest of existing code unchanged...

That is ALL. The main barometer_core.py code is untouched.
The 6 new sentiment features flow automatically into:
  • LSTMBarometer     (via X_seq tensor)
  • TCNBarometer      (via X_seq tensor)
  • TFTLiteBarometer  (via X_seq tensor)
  • XGBoostBarometer  (via flattened stats)
  • LightGBMMetaLearner (via meta-feature matrix)
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST — run this file directly to validate the pipeline
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 65)
    print("  SENTIMENT PIPELINE — STANDALONE TEST")
    print("=" * 65)

    # ── Test 1: FinBERT scorer ─────────────────────────────────────────────────
    print("\n[TEST 1] FinBERT scorer on sample headlines")
    config = SentimentConfig()
    scorer = FinBERTScorer(config)

    test_headlines = [
        "Microsoft reports record quarterly earnings, beats all estimates",
        "NVDA stock crashes 15% on disappointing guidance and weak demand",
        "Amazon AWS revenue growth slows amid cloud spending uncertainty",
        "Amgen announces positive Phase 3 trial results for new drug",
        "Federal Reserve signals further rate hikes amid persistent inflation",
    ]

    try:
        scores = scorer.score_batch(test_headlines)
        print(f"\n  {'Headline':<55} {'Pos':>6} {'Neg':>6} {'Neu':>6} {'Score':>7}")
        print(f"  {'─' * 80}")
        for i, row in scores.iterrows():
            txt   = test_headlines[i][:52] + "..."
            score = row["positive"] - row["negative"]
            icon  = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
            print(f"  {icon} {txt:<55} "
                  f"{row['positive']:>6.3f} {row['negative']:>6.3f} "
                  f"{row['neutral']:>6.3f} {score:>+7.3f}")
    except Exception as e:
        print(f"  ⚠️  FinBERT test failed: {e}")
        print("  Install: pip install transformers torch")

    # ── Test 2: Yahoo Finance fetcher (no key required) ────────────────────────
    print("\n[TEST 2] Yahoo Finance news fetch for MSFT")
    fetcher = YahooNewsFetcher()
    df_news = fetcher.fetch("MSFT")
    if df_news.empty:
        print("  ⚠️  No Yahoo news returned")
    else:
        print(f"  ✅ Fetched {len(df_news)} headlines")
        for _, row in df_news.head(3).iterrows():
            print(f"  • [{row['published_at'].strftime('%Y-%m-%d')}] {row['title'][:70]}")

    # ── Test 3: Full pipeline on dummy DataFrame ───────────────────────────────
    print("\n[TEST 3] Full pipeline.enrich() on dummy price DataFrame")
    idx     = pd.bdate_range("2024-01-01", "2024-03-31")    # business days only
    dummy   = pd.DataFrame({"close": np.random.randn(len(idx)) + 100}, index=idx)
    pipeline = SentimentPipeline(config)
    enriched = pipeline.enrich(dummy, "MSFT", "2024-01-01", "2024-03-31")
    sent_cols = [c for c in enriched.columns if c.startswith("sent_")]
    print(f"  ✅ Sentiment columns added: {sent_cols}")
    print(f"  Sample values (last 3 rows):")
    print(enriched[sent_cols].tail(3).to_string(float_format="{:.4f}".format))

    print("\n" + "=" * 65)
    print("  All tests complete.")
    print("  Next step: follow INTEGRATION INSTRUCTIONS above")
    print("=" * 65)
