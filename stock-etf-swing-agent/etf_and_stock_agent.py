#!/usr/bin/env python3
"""
ETF Swing Trading Agent
Evaluates ETFs and individual stocks based on technical, fundamental, and
sentiment factors to identify top candidates for near-term outperformance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import logging
import time
import re
import os
import argparse
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
import warnings
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from textblob import TextBlob
warnings.filterwarnings('ignore', category=FutureWarning, module='ta')


# Valid ETF symbol pattern: alphanumeric, dots, hyphens, underscores (max 10 chars)
VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9._-]{1,10}$')


def validate_etf_symbol(symbol: str) -> bool:
    """
    Validate ETF symbol format.
    
    Args:
        symbol: ETF ticker symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    return bool(VALID_SYMBOL_PATTERN.match(symbol.upper()))


class RateLimiter:
    """
    Token bucket rate limiter for HTTP requests.
    
    Prevents overwhelming servers with too many requests in a short time.
    Uses a token bucket algorithm with configurable rate and burst capacity.
    """
    
    def __init__(self, rate_per_second: float = 1.0, burst: int = 5):
        """
        Initialize rate limiter.
        
        Args:
            rate_per_second: Maximum requests per second (e.g., 1.0 = 1 request/sec)
            burst: Maximum burst size (tokens accumulated during idle periods)
        """
        self.rate = rate_per_second
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()  # Thread-safe token bucket
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Caller must hold the lock."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now
    
    def acquire(self) -> float:
        """
        Acquire a token, waiting if necessary. Thread-safe.
        
        Returns:
            Time waited in seconds
        """
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return waited
                # Not enough tokens; compute how long to wait
                sleep_time = (1.0 - self.tokens) / self.rate
            # Sleep outside the lock so other threads can proceed
            time.sleep(sleep_time)
            waited += sleep_time
    
    def __call__(self) -> 'RateLimiter':
        """Context manager support for rate limiting."""
        self.acquire()
        return self


# Shared modules
from indicators import (
    calculate_technical_indicators,
    calculate_short_term_indicators,
    calculate_short_term_score,
    calculate_day_trade_indicators,
    calculate_day_trade_score,
    calculate_4week_growth_outlook,
)
from scoring import calculate_technical_score, calculate_fundamental_score
from retry import retry, retry_call

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETFSwingAgent:
    def __init__(self, config_path: str = "config.yaml", mode: str = "stock",
                 horizon: str = "swing") -> None:
        """Initialize the ETF Swing Agent with configuration.

        Args:
            config_path: Path to YAML configuration file.
            mode: 'etf', 'stock', 'all', or 'owned-etf'.
            horizon: 'swing' (3-20 day hold, default) or 'day' (1-5 day hold).
                     'day' uses ultra-short indicators (RSI(2), 1-3d ROC,
                     Bollinger squeeze, gaps) optimized for day-scale entries.
        """
        self.horizon = horizon
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Validate config before using it (fail fast with clear messages)
        self._validate_config(self.config)

        if mode == "etf":
            self.etf_universe: List[str] = self.config['etf_universe']
            self._stock_symbols: Set[str] = set()
            self._owned_etf_symbols: Set[str] = set(self._load_symbols_from_file('currently_own_etf.dat'))
        elif mode == "stock":
            self.etf_universe = self._load_symbols_from_file('corrently_own_stocks.dat')
            if not self.etf_universe:
                self.etf_universe = self.config['etf_universe']
            self._stock_symbols = set(self.etf_universe)
            self._owned_etf_symbols = set()
        elif mode == "all":
            etf_symbols = set(self.config['etf_universe'])
            stock_symbols = set(self._load_symbols_from_file('corrently_own_stocks.dat'))
            self.etf_universe = list(etf_symbols.union(stock_symbols))
            self._stock_symbols = stock_symbols
            self._owned_etf_symbols = set(self._load_symbols_from_file('currently_own_etf.dat'))
        elif mode == "owned-etf":
            # Screen ONLY the ETFs listed in currently_own_etf.dat.
            # These are also treated as owned so the Sell guardrail applies.
            owned = self._load_symbols_from_file('currently_own_etf.dat')
            if not owned:
                raise ValueError(
                    "No symbols found in currently_own_etf.dat. "
                    "Add at least one ETF (one symbol per line) to use --mode owned-etf."
                )
            self.etf_universe = owned
            self._stock_symbols = set()
            self._owned_etf_symbols = set(owned)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'etf', 'stock', 'all', 'owned-etf'.")

        # Store mode so evaluation/ranking can apply short-term logic for stocks
        self.mode = mode

        # Human-readable label for the asset type being evaluated
        if mode == "stock":
            self._asset_label = "Stock"
            self._asset_label_plural = "Stocks"
        elif mode in ("etf", "owned-etf"):
            self._asset_label = "ETF"
            self._asset_label_plural = "ETFs"
        else:
            self._asset_label = "Symbol"
            self._asset_label_plural = "Symbols"

        # If after all that we still have no symbols, raise an error.
        if not self.etf_universe:
            raise ValueError("No symbols to evaluate. Check your data sources.")

        self.market_regime: Dict[str, Any] = self.config['market_regime']
        self.technical_weights: Dict[str, float] = self.config['technical_weights']
        self.fundamental_weights: Dict[str, float] = self.config['fundamental_weights']
        self.news_config: Dict[str, Any] = self.config['news']
        self.risk_config: Dict[str, Any] = self.config['risk']
        self.day_trade_config: Dict[str, Any] = self.config.get('day_trade', {})
        self.output_config: Dict[str, Any] = self.config['output']

        # Composite score weights (must match backtester for consistency)
        self.composite_weights: Dict[str, float] = self.config.get('composite_weights', {
            'technical': 0.50,
            'fundamental': 0.30,
            'sentiment': 0.20,
        })
        
        # Cache for data to avoid redundant downloads
        # Each entry: {'data': DataFrame, 'fetched_at': datetime, 'period': str}
        # Using OrderedDict for LRU eviction
        self._data_cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_maxsize = 100  # Maximum number of cached entries
        self._cache_lock = threading.Lock()  # Thread-safe cache access
        
        # Cache TTL by period (shorter periods = fresher data needed)
        self._cache_ttl: Dict[str, timedelta] = {
            '1mo': timedelta(hours=1),
            '3mo': timedelta(hours=4),
            '6mo': timedelta(hours=8),
            '1y': timedelta(hours=24),
            '2y': timedelta(hours=48),
        }
        self._default_cache_ttl = timedelta(hours=6)
        
        # Rate limiter for web scraping (configurable via news config)
        news_rate = self.news_config.get('rate_limit_per_sec', 0.5)  # Default: 1 request per 2 seconds
        self._rate_limiter = RateLimiter(rate_per_second=news_rate, burst=3)

        # Max worker threads for parallel ETF evaluation
        self._max_workers = self.output_config.get('max_workers', 8)

    @staticmethod
    def _validate_config(config: dict) -> None:
        """
        Validate the configuration structure and weight sums.

        Raises:
            ValueError: If required sections are missing or weights don't sum to 1.0.
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a mapping (invalid YAML root).")

        required_sections = [
            'etf_universe', 'market_regime', 'technical_weights',
            'fundamental_weights', 'news', 'risk', 'output',
        ]
        missing = [s for s in required_sections if s not in config]
        if missing:
            raise ValueError(f"Config missing required section(s): {', '.join(missing)}")

        # Note: etf_universe validation is skipped here since we load from file
        # but we'll check it's not empty after loading in __init__

        # Validate weight groups sum to ~1.0 (tolerance for float rounding)
        weight_groups = {
            'technical_weights': config.get('technical_weights', {}),
            'fundamental_weights': config.get('fundamental_weights', {}),
            'composite_weights': config.get('composite_weights', {}),
        }
        tolerance = 0.01
        for name, weights in weight_groups.items():
            if not weights:
                continue  # composite_weights is optional (has a default)
            total = sum(weights.values())
            if abs(total - 1.0) > tolerance:
                raise ValueError(
                    f"Config '{name}' weights must sum to 1.0 (got {total:.4f})."
                )

    def _load_symbols_from_file(self, filename: str) -> List[str]:
        """Load symbols from a file, one per line, ignoring empty lines and comments."""
        try:
            with open(filename, 'r') as f:
                symbols = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        symbols.append(line.upper())
                return symbols
        except FileNotFoundError:
            return []

    def fetch_etf_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch historical data for an ETF with TTL-based caching."""
        # Validate symbol format
        if not validate_etf_symbol(symbol):
            logger.warning(f"Invalid ETF symbol format: {symbol}")
            return None
        
        cache_key = f"{symbol}:{period}"
        now = datetime.now()
        
        # Check cache with TTL (thread-safe)
        with self._cache_lock:
            if cache_key in self._data_cache:
                entry = self._data_cache[cache_key]
                ttl = self._cache_ttl.get(period, self._default_cache_ttl)
                age = now - entry['fetched_at']
                if age < ttl:
                    # Move to end for LRU (most recently used)
                    self._data_cache.move_to_end(cache_key)
                    logger.debug(f"Cache hit for {cache_key} (age: {age})")
                    return entry['data']  # type: ignore[no-any-return]
                else:
                    logger.debug(f"Cache expired for {cache_key} (age: {age} > ttl: {ttl})")
                    # Remove expired entry
                    del self._data_cache[cache_key]
            
        try:
            ticker = yf.Ticker(symbol)
            data = retry_call(ticker.history, period=period)
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
                
            # Store in cache (thread-safe), evicting oldest if full (LRU)
            with self._cache_lock:
                while len(self._data_cache) >= self._cache_maxsize:
                    oldest_key = next(iter(self._data_cache))
                    del self._data_cache[oldest_key]
                    logger.debug(f"Cache eviction: removed {oldest_key}")

                self._data_cache[cache_key] = {
                    'data': data,
                    'fetched_at': now,
                    'period': period,
                }
            return data  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} after retries: {e}")
            return None
    
    def clear_cache(self, older_than: Optional[timedelta] = None) -> None:
        """
        Clear cached data.
        
        Args:
            older_than: If provided, only clear entries older than this duration.
                        If None, clear all cached data.
        """
        with self._cache_lock:
            if older_than is None:
                self._data_cache.clear()
                logger.info("Cache cleared (all entries)")
            else:
                now = datetime.now()
                expired = [
                    k for k, v in self._data_cache.items()
                    if now - v['fetched_at'] > older_than
                ]
                for k in expired:
                    del self._data_cache[k]
                logger.info(f"Cache cleared ({len(expired)} expired entries removed, "
                           f"{len(self._data_cache)} remaining)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "size": len(self._data_cache),
            "maxsize": self._cache_maxsize,
            "utilization_pct": round(len(self._data_cache) / self._cache_maxsize * 100, 1)
        }
    
    def get_market_regime(self) -> Dict[str, Any]:
        """Determine current market regime."""
        spy_data = self.fetch_etf_data("SPY", "1y")
        if spy_data is None or len(spy_data) < 200:
            return {"regime": "unknown", "volatility": "unknown", "confidence": 0.0}
        
        close = spy_data['Close']
        sma_200 = close.rolling(200).mean().iloc[-1]
        price = close.iloc[-1]
        
        # Trend determination
        price_vs_sma200 = price / sma_200 - 1
        
        if price_vs_sma200 > self.market_regime['trend_threshold']:
            trend = "bull"
        elif price_vs_sma200 < -self.market_regime['trend_threshold']:
            trend = "bear"
        else:
            trend = "sideways"
        
        # Volatility regime (using VIX if available, otherwise price volatility)
        try:
            vix_data = retry_call(yf.Ticker("^VIX").history, period="1mo")
            if not vix_data.empty:
                vix_level = vix_data['Close'].iloc[-1]
                if vix_level > self.market_regime['vix_high_threshold']:
                    vol_regime = "high"
                elif vix_level < self.market_regime['vix_low_threshold']:
                    vol_regime = "low"
                else:
                    vol_regime = "moderate"
            else:
                # Fallback to price volatility
                returns = close.pct_change(fill_method=None).dropna()
                vol_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                if vol_20 > 0.3:
                    vol_regime = "high"
                elif vol_20 < 0.15:
                    vol_regime = "low"
                else:
                    vol_regime = "moderate"
        except (requests.RequestException, ValueError, KeyError, IndexError) as e:
            logger.debug(f"VIX/volatility regime detection failed: {e}")
            vol_regime = "unknown"

        # Confidence: how far price is from the trend threshold (0 = at threshold, 1 = far)
        threshold = self.market_regime['trend_threshold']
        confidence = min(abs(price_vs_sma200) / (threshold * 2), 1.0) if threshold > 0 else 0.0

        return {
            "regime": trend,
            "volatility": vol_regime,
            "price_vs_sma200": price_vs_sma200,
            "confidence": confidence,
        }
    
    def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment for a symbol using yfinance news + TextBlob.

        Uses yfinance's built-in ``ticker.news`` method (fast, free, no scraping
        needed) to fetch recent news articles, then scores each title + publisher
        summary with TextBlob polarity. Falls back to a price-momentum proxy if
        no news is available.

        Returns:
            dict with keys 'score' (0-1), 'articles' (int), 'source' (str).
            'source' is 'news' if real articles were found, 'fallback' otherwise.
        """
        if not self.news_config['enabled']:
            return {"score": 0.5, "articles": 0, "source": "disabled"}

        # Validate symbol format
        if not validate_etf_symbol(symbol):
            logger.warning(f"Invalid symbol format: {symbol}")
            return {"score": 0.5, "articles": 0, "source": "invalid"}

        try:
            ticker = yf.Ticker(symbol)
            news_items = retry_call(lambda: ticker.news) or []

            polarities = []
            for item in news_items[:20]:  # Cap at 20 most recent
                title = (item.get('content', {}) or {}).get('title', '')
                summary = (item.get('content', {}) or {}).get('summary', '')
                text = f"{title}. {summary}".strip()
                if len(text) < 20:
                    # Some items have only a title — use that if it's substantive
                    text = title.strip() if len(title) > 15 else ""
                if not text:
                    continue

                polarity = TextBlob(text).sentiment.polarity
                if polarity != 0.0:
                    polarities.append(polarity)

            if polarities:
                mean_polarity = sum(polarities) / len(polarities)
                # Map polarity [-1, 1] -> sentiment [0, 1]
                sentiment = 0.5 + 0.5 * max(-1.0, min(1.0, mean_polarity))
                return {"score": max(0.0, min(1.0, sentiment)), "articles": len(polarities), "source": "news"}
            else:
                fb = self._get_fallback_sentiment(symbol)
                return {"score": fb, "articles": 0, "source": "fallback"}

        except Exception as e:
            logger.warning(f"Error in news sentiment analysis for {symbol}: {e}")
            fb = self._get_fallback_sentiment(symbol)
            return {"score": fb, "articles": 0, "source": "error"}
    
    def _get_fallback_sentiment(self, symbol: str) -> float:
        """Fallback sentiment based on recent price action when news is unavailable."""
        try:
            data = self.fetch_etf_data(symbol, "1mo")
            if data is None or len(data) < 5:
                return 0.5
            
            # Simple sentiment based on recent price action
            recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) if len(data) >= 5 else 0
            
            # Convert return to sentiment score (0-1)
            # -5% return = 0.0, +5% return = 1.0
            sentiment = 0.5 + recent_return * 10
            return float(min(max(sentiment, 0), 1))
        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Fallback sentiment error for {symbol}: {e}")
            return 0.5

    def evaluate_etf(
        self,
        symbol: str,
        market_regime: Optional[Dict[str, Any]] = None,
        bench_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single ETF and return scores.

        Args:
            symbol: ETF ticker to evaluate.
            market_regime: Pre-computed market regime (avoids re-fetching per ETF).
            bench_data: Pre-fetched SPY 6mo data for tracking-error calculation.
        """
        # Validate symbol format early
        if not validate_etf_symbol(symbol):
            logger.warning(f"Invalid ETF symbol format: {symbol}")
            return {"symbol": symbol, "error": "Invalid symbol format"}

        logger.info(f"Evaluating {symbol}...")

        # Fetch data
        data = self.fetch_etf_data(symbol, "6mo")
        if data is None:
            return {"symbol": symbol, "error": "No data available"}

        # Calculate components
        technical_indicators = calculate_technical_indicators(data)

        # Determine if this symbol is a stock (not an ETF) for scoring
        is_stock = symbol in self._stock_symbols

        # For stocks, reduce mean-reversion weight (it penalizes trending
        # stocks near upper Bollinger Band) and redistribute to momentum/trend.
        if is_stock:
            stock_tech_weights = dict(self.technical_weights)
            stock_tech_weights['mean_reversion_score'] = 0.05
            stock_tech_weights['momentum_score'] = 0.35
            stock_tech_weights['trend_score'] = 0.35
            technical_score = calculate_technical_score(technical_indicators, stock_tech_weights)
        else:
            technical_score = calculate_technical_score(technical_indicators, self.technical_weights)

        # Use pre-fetched benchmark data if provided, else fetch on demand
        if bench_data is None:
            bench_data = self.fetch_etf_data("SPY", "6mo")
        fundamental_score = calculate_fundamental_score(
            symbol, self.fundamental_weights, price_data=data,
            benchmark_data=bench_data, is_stock=is_stock,
        )
        sentiment_result = self.get_news_sentiment(symbol)
        sentiment_score = sentiment_result["score"]
        sentiment_articles = sentiment_result["articles"]
        sentiment_source = sentiment_result["source"]

        # Use pre-computed market regime if provided
        if market_regime is None:
            market_regime = self.get_market_regime()

        # Composite score: for stocks, down-weight fundamental (ETF-centric
        # fields are meaningless) and sentiment (mostly noise from failed
        # news scraping). Redistribute to technical.
        cw = self.composite_weights
        if is_stock:
            # Stocks: 80% technical, 10% fundamental, 10% sentiment
            composite_score = (
                technical_score * 0.80 +
                fundamental_score * 0.10 +
                sentiment_score * 0.10
            )
        else:
            composite_score = (
                technical_score * cw['technical'] +
                fundamental_score * cw['fundamental'] +
                sentiment_score * cw['sentiment']
            )

        # Current price info
        current_price = data['Close'].iloc[-1]
        price_change_1d = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100 if len(data) >= 2 else 0
        price_change_1w = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100 if len(data) >= 5 else 0

        # Short-term (days-scale) score — computed for stocks in any mode.
        # Uses SPY 1mo data for relative-strength comparison.
        # Also compute for ETFs to power the 4-week growth outlook.
        short_term_score = 0.0
        day_trade_score = 0.0
        growth_outlook = None
        short_data = self.fetch_etf_data(symbol, "1mo")
        if short_data is not None and len(short_data) >= 20:
            short_ind = calculate_short_term_indicators(short_data)
            # Fetch SPY 1mo once for relative strength (cached, so cheap after first call)
            spy_short = self.fetch_etf_data("SPY", "1mo")
            spy_ind = calculate_short_term_indicators(spy_short) if spy_short is not None and len(spy_short) >= 20 else None
            if is_stock:
                short_term_score = calculate_short_term_score(short_ind, spy_indicators=spy_ind)
            else:
                # ETFs: compute growth outlook from short-term data + sentiment + regime
                growth_outlook = calculate_4week_growth_outlook(
                    short_ind,
                    spy_indicators=spy_ind,
                    sentiment_score=sentiment_score,
                    price_change_1w=price_change_1w,
                    market_regime=market_regime['regime'],
                )

        # Day-trade (1-5 day) score — computed when horizon == "day".
        # Uses ~2 weeks of data for ultra-short indicators.
        if self.horizon == "day":
            day_data = self.fetch_etf_data(symbol, "2wk")
            if day_data is not None and len(day_data) >= 10:
                day_ind = calculate_day_trade_indicators(day_data)
                spy_day = self.fetch_etf_data("SPY", "2wk")
                spy_day_ind = calculate_day_trade_indicators(spy_day) if spy_day is not None and len(spy_day) >= 10 else None
                day_trade_score = calculate_day_trade_score(day_ind, spy_indicators=spy_day_ind)

        # Dividend yield — compute once here so main() doesn't need extra calls
        dividend_yield_pct = None
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            rate = info.get('trailingAnnualDividendRate')
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            if rate and price:
                dividend_yield_pct = (rate / price) * 100
            if dividend_yield_pct is None:
                raw = info.get('dividendYield', 0) or 0
                dividend_yield_pct = raw  # raw is already in percentage from yfinance
        except Exception:
            dividend_yield_pct = None

        return {
            "symbol": symbol,
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "sentiment_score": sentiment_score,
            "sentiment_articles": sentiment_articles,
            "sentiment_source": sentiment_source,
            "composite_score": composite_score,
            "short_term_score": short_term_score,
            "day_trade_score": day_trade_score,
            "growth_outlook": growth_outlook,
            "dividend_yield_pct": dividend_yield_pct,
            "current_price": current_price,
            "price_change_1d": price_change_1d,
            "price_change_1w": price_change_1w,
            "market_regime": market_regime['regime'],
            "volatility_regime": market_regime['volatility'],
            "atr": technical_indicators.get('atr_14', 0.0),
            "data_points": len(data)
        }

    def run_screening(self) -> List[Dict[str, Any]]:
        """Run the full screening process (symbols evaluated in parallel)."""
        label = self._asset_label_plural
        logger.info(f"Starting {label} screening for {len(self.etf_universe)} {label}...")

        # Compute market regime and SPY benchmark ONCE (not per-ETF)
        market_regime = self.get_market_regime()
        spy_bench = self.fetch_etf_data("SPY", "6mo")

        results: List[Dict[str, Any]] = []
        errors = 0

        # Evaluate ETFs concurrently (I/O-bound: network + scraping)
        max_workers = min(self._max_workers, len(self.etf_universe)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self.evaluate_etf, symbol,
                    market_regime=market_regime, bench_data=spy_bench
                ): symbol
                for symbol in self.etf_universe
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if "error" not in result:
                        results.append(result)
                    else:
                        errors += 1
                        logger.warning(f"Skipping {symbol}: {result['error']}")
                except Exception as e:
                    errors += 1
                    logger.error(f"Error evaluating {symbol}: {e}")

        logger.info(
            f"Evaluated {len(self.etf_universe)} {label}: "
            f"{len(results)} scored, {errors} errors"
        )

        # Rank: stocks by short_term_score (swing) or day_trade_score (day),
        # ETFs by growth_score (primary criteria).
        # In "all" mode, both are 0-1 so they're comparable on the same scale.
        for r in results:
            is_stock = r['symbol'] in self._stock_symbols
            if is_stock:
                if self.horizon == "day":
                    r['_rank_score'] = r.get('day_trade_score', 0.0)
                else:
                    r['_rank_score'] = r['short_term_score']
            else:
                growth = r.get('growth_outlook')
                r['_rank_score'] = growth['growth_score'] if (growth and growth.get('growth_score', 0.0) > 0) else r['composite_score']
        results.sort(key=lambda x: x['_rank_score'], reverse=True)

        # Filter by minimum threshold — mode-aware:
        #   ETFs: 0.55 (config value)
        #   Stocks: 0.35 (lower bar; short-term scores cluster lower)
        #   owned-etf: 0.0 (user explicitly wants to evaluate these specific
        #              holdings, so never filter them out by score)
        etf_threshold = self.risk_config['min_score_threshold']
        stock_threshold = 0.35
        owned_etf_threshold = 0.0
        # Day-trade threshold (lower; day-trade scores cluster lower)
        day_trade_threshold = self.day_trade_config.get('min_score_threshold', 0.30)
        filtered_results: List[Dict[str, Any]] = []
        for r in results:
            is_stock = r['symbol'] in self._stock_symbols
            if self.mode == "owned-etf":
                thresh = owned_etf_threshold
            elif self.horizon == "day":
                thresh = day_trade_threshold
            elif is_stock:
                thresh = stock_threshold
            else:
                thresh = etf_threshold
            if r['_rank_score'] >= thresh:
                filtered_results.append(r)

        # Fallback: if threshold is too strict, use top N by score regardless
        top_n = self.output_config['top_n']
        if len(filtered_results) < top_n:
            logger.info(
                f"Only {len(filtered_results)} symbols above threshold; "
                f"falling back to top {top_n} by score"
            )
            filtered_results = results[:top_n]

        # Apply correlation filter to limit highly correlated positions
        filtered_results = self._apply_correlation_filter(filtered_results)

        # Return top N
        top_results = filtered_results[:top_n]

        # Apply position sizing based on max_position_pct
        self._apply_position_sizing(top_results)

        logger.info(f"Screening complete. Found {len(filtered_results)} {label} above threshold, returning top {len(top_results)}")
        return top_results

    def _apply_position_sizing(self, results: List[Dict[str, Any]]) -> None:
        """
        Assign a position weight to each selected ETF, capped at
        ``risk.max_position_pct`` and normalized so weights sum to 1.0.

        Mutates ``results`` in place, adding a ``position_pct`` key.
        """
        max_pct = self.risk_config.get('max_position_pct', 0.20)
        n = len(results)
        if n == 0:
            return

        # Equal weight capped at max_position_pct, then normalize to 100%
        raw_weight = min(1.0 / n, max_pct)
        total = raw_weight * n
        for r in results:
            r['position_pct'] = round(raw_weight / total, 4)  # sums to ~1.0

    def _apply_correlation_filter(self, etf_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter ETFs to limit highly correlated positions.
        
        Uses pairwise price correlation over the lookback period. If more than
        max_correlated_positions ETFs are correlated above the threshold,
        lower-scoring ones are removed.
        """
        max_corr = self.risk_config.get('max_correlated_positions', 3)
        corr_threshold = 0.7
        
        if len(etf_scores) <= 1:
            return etf_scores
        
        # Build correlation matrix from price returns (read from cache, no refetch)
        price_series: Dict[str, pd.Series] = {}
        for etf in etf_scores:
            cache_key = f"{etf['symbol']}:6mo"
            entry = self._data_cache.get(cache_key)
            data = entry['data'] if entry else self.fetch_etf_data(etf['symbol'], "6mo")
            if data is not None and len(data) >= 20:
                price_series[etf['symbol']] = data['Close'].pct_change(fill_method=None).dropna()
        
        if len(price_series) < 2:
            return etf_scores
        
        returns_df = pd.DataFrame(price_series)
        corr_matrix = returns_df.corr()
        
        # Greedy selection: start with highest score, add if not too correlated.
        # Once max_correlated_positions correlated symbols are selected, reject
        # all further correlated symbols (global cap, not per-symbol).
        selected: List[Dict[str, Any]] = []
        correlated_count = 0
        for etf in etf_scores:
            symbol = etf['symbol']
            if symbol not in corr_matrix.columns:
                selected.append(etf)
                continue

            high_corr_count = 0
            for sel in selected:
                sel_sym = sel['symbol']
                if sel_sym in corr_matrix.columns and symbol in corr_matrix.columns:
                    if abs(corr_matrix.loc[symbol, sel_sym]) > corr_threshold:
                        high_corr_count += 1

            if high_corr_count == 0:
                selected.append(etf)
            elif correlated_count < max_corr:
                selected.append(etf)
                correlated_count += 1

        return selected
    
    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Display results in a formatted way."""
        label = self._asset_label_plural
        if not results:
            print(f"No {label} met the screening criteria.")
            return

        print("\n" + "="*80)
        horizon_label = "DAY-TRADE" if self.horizon == "day" else "SWING"
        print(f"{label.upper()} {horizon_label} AGENT - TOP RECOMMENDATIONS")
        print("="*80)
        print(f"Screened {len(self.etf_universe)} {label} ({horizon_label} horizon)")
        print(f"Market Regime: {results[0]['market_regime'] if results else 'Unknown'} ({results[0]['volatility_regime'] if results else 'Unknown'} volatility)")
        print("-"*80)

        for i, etf in enumerate(results, 1):
            atr = etf.get('atr', 0)
            price = etf['current_price']
            # Use day-trade multipliers when horizon is "day"
            if self.horizon == "day":
                sl_mult = self.day_trade_config.get('stop_loss_atr_mult', 1.5)
                tp_mult = self.day_trade_config.get('take_profit_atr_mult', 2.0)
            else:
                sl_mult = self.risk_config['stop_loss_atr_mult']
                tp_mult = self.risk_config['take_profit_atr_mult']
            stop_loss = price - sl_mult * atr if atr > 0 else None
            take_profit = price + tp_mult * atr if atr > 0 else None
            is_stock = etf['symbol'] in self._stock_symbols
            is_owned_etf = etf['symbol'] in self._owned_etf_symbols
            rank_score = etf.get('_rank_score', etf['composite_score'])

            # Action model: score-based for stocks, growth-primary for ETFs
            div_pct = etf.get('dividend_yield_pct')
            if is_stock:
                if rank_score >= 0.70:
                    div_action = "Strong Buy"
                elif rank_score >= 0.50:
                    div_action = "Buy"
                elif rank_score >= 0.30:
                    div_action = "Hold"
                else:
                    div_action = "Sell"
                yield_str = f"{div_pct:.2f}%" if div_pct is not None else "N/A"
            else:
                # --- Primary: 4-week growth outlook ---
                growth = etf.get('growth_outlook')
                if growth and growth.get('growth_score', 0.0) > 0:
                    gs = growth['growth_score']
                    if gs >= 0.70:
                        growth_action = "Strong Buy"
                    elif gs >= 0.50:
                        growth_action = "Buy"
                    elif gs >= 0.30:
                        growth_action = "Hold"
                    else:
                        growth_action = "Sell"
                else:
                    growth_action = "N/A"

                # --- Supplementary: dividend-yield evaluation ---
                if div_pct is not None:
                    if div_pct >= 5.0:
                        yield_note = f"Yield: {div_pct:.2f}% (good)"
                    elif div_pct >= 4.0:
                        yield_note = f"Yield: {div_pct:.2f}% (standard)"
                    else:
                        yield_note = f"Yield: {div_pct:.2f}% (low)"
                    yield_str = f"{div_pct:.2f}%"
                else:
                    yield_note = "Yield: N/A"
                    yield_str = "N/A"

                div_action = f"{growth_action} | {yield_note}"
                if is_owned_etf:
                    div_action += " | Currently owned"

            print(f"{i}. {etf['symbol']}  [{div_action}]")
            print(f"   Dividend Yield : {yield_str}")
            if self.horizon == "day":
                print(f"   Day-Trade Score: {etf.get('day_trade_score', 0.0):.3f} (1-5 day ranking)")
            elif is_stock:
                print(f"   Short-term Score: {etf['short_term_score']:.3f} (days-scale ranking)")
            else:
                growth = etf.get('growth_outlook')
                if growth and growth.get('growth_score', 0.0) > 0:
                    print(
                        f"   4-Week Growth   : {growth['growth_score']:.3f} "
                        f"(Mom: {growth['momentum']:.3f}, Sent: {growth['sentiment']:.3f}, "
                        f"Trend: {growth['price_trend']:.3f}, Regime: {growth['regime_adj']:.3f}, "
                        f"Vol: {growth['volume']:.3f})"
                    )
                else:
                    print(f"   4-Week Growth   : N/A (insufficient data)")
            print(f"   Composite Score: {etf['composite_score']:.3f}")
            print(f"   Technical: {etf['technical_score']:.3f} | Fundamental: {etf['fundamental_score']:.3f} | Sentiment: {etf['sentiment_score']:.3f} ({etf.get('sentiment_source', '?')})")
            print(f"   Price: ${etf['current_price']:.2f} (1D: {etf['price_change_1d']:+.2f}%, 1W: {etf['price_change_1w']:+.2f}%)")
            if stop_loss and take_profit:
                print(f"   Stop-loss: ${stop_loss:.2f} | Take-profit: ${take_profit:.2f} | ATR: ${atr:.2f}")
            print(f"   Regime: {etf['market_regime']} ({etf['volatility_regime']} vol)")
            pos = etf.get('position_pct')
            if pos is not None:
                print(f"   Suggested Position: {pos*100:.1f}%")
            print()

        print("="*80)
        print("Scores: 0.0-1.0 (higher is better)")
        print("Technical: Trend, momentum, mean reversion, volume, volatility")
        print("Fundamental: Expense ratio, liquidity, AUM, yield, tracking")
        print("Sentiment: News-based sentiment analysis")
        print("Position sizing: equal-weight capped at max_position_pct, normalized to 100%")
        print("="*80)


def _emit_rotation_signals(results: List[Dict[str, Any]], stock_symbols: Set[str]) -> None:
    """
    Compare current rankings against the previous run to flag exit candidates.

    Saves the current top-10 rankings to ``output/last_rankings.csv``. On the
    next run, any symbol that was previously in the top 5 but has now dropped
    below the top 10 (or below a score threshold) is flagged as a rotation exit.
    """
    rankings_file = "output/last_rankings.csv"
    current_rankings: Dict[str, Dict[str, Any]] = {}
    for i, r in enumerate(results):
        current_rankings[r['symbol']] = {
            'rank': i + 1,
            'score': r.get('_rank_score', r['composite_score']),
            'short_term_score': r.get('short_term_score', 0.0),
            'composite_score': r['composite_score'],
        }

    # Save current rankings for next run
    os.makedirs("output", exist_ok=True)
    pd.DataFrame([
        {'symbol': s, 'rank': v['rank'], 'score': v['score'],
         'short_term_score': v['short_term_score'],
         'composite_score': v['composite_score']}
        for s, v in current_rankings.items()
    ]).to_csv(rankings_file, index=False)

    # Load previous rankings
    if not os.path.exists(rankings_file + ".prev"):
        # First run — save a copy as "prev" for next time
        shutil.copy2(rankings_file, rankings_file + ".prev")
        return

    try:
        prev = pd.read_csv(rankings_file + ".prev")
        prev_top5 = set(prev[prev['rank'] <= 5]['symbol'].values)
        prev_scores = dict(zip(prev['symbol'], prev['score']))

        exits: List[Tuple[str, str]] = []
        for sym in prev_top5:
            cur = current_rankings.get(sym)
            if cur is None:
                exits.append((sym, "dropped from universe"))
            elif cur['rank'] > 10:
                exits.append((sym, f"fell from top 5 to rank {cur['rank']}"))
            elif cur['score'] < 0.30:
                exits.append((sym, f"score dropped to {cur['score']:.3f}"))

        if exits:
            print("=" * 78)
            print("ROTATION / EXIT SIGNALS (vs previous run)")
            print("-" * 78)
            for sym, reason in exits:
                print(f"  ⚠ {sym}: {reason}")
            print("=" * 78)
            print()

        # Rotate: current becomes prev for next run
        shutil.copy2(rankings_file, rankings_file + ".prev")
    except Exception:
        pass  # Silently skip if prev file is corrupted


def main() -> None:
    """Main function to run the ETF Swing Agent."""
    parser = argparse.ArgumentParser(description="ETF Swing Trading Agent")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--mode", choices=['etf', 'stock', 'all', 'owned-etf'], default='stock',
        help="Mode of operation: 'etf' to evaluate ETFs from config, 'stock' to evaluate stocks from file, 'all' to evaluate both, 'owned-etf' to evaluate only the ETFs listed in currently_own_etf.dat (default: stock)"
    )
    parser.add_argument(
        "--horizon", choices=['swing', 'day'], default='swing',
        help="Trading horizon: 'swing' (3-20 day hold, default) or 'day' (1-5 day hold). "
             "'day' uses ultra-short indicators (RSI(2), 1-3d ROC, Bollinger squeeze, gaps) "
             "optimized for day-scale entries."
    )
    args = parser.parse_args()

    agent = ETFSwingAgent(args.config, mode=args.mode, horizon=args.horizon)
    results = agent.run_screening()
    
    # Output top results with detailed metrics (dividend yield pre-computed in evaluate_etf)
    top_n = min(len(results), 3)
    top_results = results[:top_n]
    label = agent._asset_label_plural
    horizon_label = "DAY-TRADE" if args.horizon == "day" else "SWING"
    print("\n" + "=" * 78)
    print(f"TOP {top_n} {label.upper()} RECOMMENDATIONS ({horizon_label} HORIZON)")
    print("=" * 78)
    for result in top_results:
        symbol = result['symbol']
        yield_pct = result.get('dividend_yield_pct')
        yield_str = f"{yield_pct:.2f}%" if yield_pct is not None else "N/A"
        is_stock = symbol in agent._stock_symbols
        is_owned_etf = symbol in agent._owned_etf_symbols
        rank_score = result.get('_rank_score', result['composite_score'])

        # Action model:
        #   Stocks: score-based (short-term momentum is what drives returns)
        #   ETFs:  growth outlook primary, dividend-yield supplementary
        if is_stock:
            if rank_score >= 0.70:
                action = "Strong Buy"
            elif rank_score >= 0.50:
                action = "Buy"
            elif rank_score >= 0.30:
                action = "Hold"
            else:
                action = "Sell"
        else:
            # --- Primary: 4-week growth outlook ---
            growth = result.get('growth_outlook')
            if growth and growth.get('growth_score', 0.0) > 0:
                gs = growth['growth_score']
                if gs >= 0.70:
                    growth_action = "Strong Buy"
                elif gs >= 0.50:
                    growth_action = "Buy"
                elif gs >= 0.30:
                    growth_action = "Hold"
                else:
                    growth_action = "Sell"
            else:
                growth_action = "N/A"

            # --- Supplementary: dividend-yield evaluation ---
            if yield_pct is not None:
                if yield_pct >= 5.0:
                    yield_note = f"Yield: {yield_pct:.2f}% (good)"
                elif yield_pct >= 4.0:
                    yield_note = f"Yield: {yield_pct:.2f}% (standard)"
                else:
                    yield_note = f"Yield: {yield_pct:.2f}% (low)"
            else:
                yield_note = "Yield: N/A"

            action = f"{growth_action} | {yield_note}"
            if is_owned_etf:
                action += " | Currently owned"

        print(f"{symbol}: {action}")
        print(f"   Dividend Yield : {yield_str}")
        print(
            f"   Price          : ${result['current_price']:.2f} "
            f"(1D: {result['price_change_1d']:+.2f}%, "
            f"1W: {result['price_change_1w']:+.2f}%)"
        )
        # Stop-loss / take-profit from ATR
        atr = result.get('atr', 0)
        price = result['current_price']
        if atr > 0:
            if agent.horizon == "day":
                sl_mult = agent.day_trade_config.get('stop_loss_atr_mult', 1.5)
                tp_mult = agent.day_trade_config.get('take_profit_atr_mult', 2.0)
            else:
                sl_mult = agent.risk_config['stop_loss_atr_mult']
                tp_mult = agent.risk_config['take_profit_atr_mult']
            sl = price - sl_mult * atr
            tp = price + tp_mult * atr
            print(f"   Stop-loss      : ${sl:.2f} | Take-profit: ${tp:.2f} (ATR: ${atr:.2f})")
        if agent.horizon == "day":
            print(f"   Day-Trade Score : {result.get('day_trade_score', 0.0):.3f} (1-5 day ranking)")
        elif agent.mode in ("stock", "all"):
            print(f"   Short-term Score: {result['short_term_score']:.3f} (days-scale ranking)")

        # 4-week growth outlook for ETFs
        if not is_stock:
            growth = result.get('growth_outlook')
            if growth and growth.get('growth_score', 0.0) > 0:
                print(
                    f"   4-Week Growth   : {growth['growth_score']:.3f} "
                    f"(Mom: {growth['momentum']:.3f}, Sent: {growth['sentiment']:.3f}, "
                    f"Trend: {growth['price_trend']:.3f}, Regime: {growth['regime_adj']:.3f}, "
                    f"Vol: {growth['volume']:.3f})"
                )
            else:
                print(f"   4-Week Growth   : N/A (insufficient data)")

        print(
            f"   Composite Score: {result['composite_score']:.3f} "
            f"(Tech: {result['technical_score']:.3f}, "
            f"Fund: {result['fundamental_score']:.3f}, "
            f"Sent: {result['sentiment_score']:.3f})"
        )
        # Sentiment source detail
        sent_src = result.get('sentiment_source', 'fallback')
        sent_art = result.get('sentiment_articles', 0)
        if sent_src == "news":
            print(f"   Sentiment Source: {sent_art} news articles")
        elif sent_src == "fallback":
            print(f"   Sentiment Source: price-momentum proxy (no news found)")
        else:
            print(f"   Sentiment Source: {sent_src}")
        print(
            f"   Regime         : {result['market_regime']} "
            f"({result['volatility_regime']} vol)"
        )
        print()

    # --- Exit / rotation signal ---
    # Compare current rankings against the previous run to flag symbols that
    # have dropped significantly and should be considered for exit.
    _emit_rotation_signals(results, agent._stock_symbols)

if __name__ == "__main__":
    main()