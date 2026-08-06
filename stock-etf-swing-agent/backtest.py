"""
Backtesting framework for ETF Swing Trading Agent
"""

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from datetime import datetime, timedelta
import logging
from typing import Dict, Iterator, List, Tuple, Optional, Any
import warnings
import requests
import re
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
warnings.filterwarnings('ignore', category=FutureWarning, module='ta')

# Valid ETF symbol pattern
VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9._-]{1,10}$')


def validate_etf_symbol(symbol: str) -> bool:
    """Validate ETF symbol format."""
    if not symbol or not isinstance(symbol, str):
        return False
    return bool(VALID_SYMBOL_PATTERN.match(symbol.upper()))


def _load_symbols_from_file(path: str) -> List[str]:
    """Load one-symbol-per-line from a .dat file, returning a list of symbols."""
    symbols = []
    try:
        with open(path, 'r') as f:
            for line in f:
                s = line.strip().upper()
                if s and not s.startswith('#'):
                    symbols.append(s)
    except FileNotFoundError:
        return []
    return symbols


# Shared modules
from indicators import (
    calculate_technical_indicators,
    calculate_short_term_indicators,
    calculate_short_term_score,
    calculate_4week_growth_outlook,
)
from scoring import calculate_technical_score, calculate_fundamental_score, compute_composite_relative_strength
from retry import retry_call

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETFBacktester:
    """Backtester for ETF Swing Trading Agent strategy."""
    
    def __init__(self, etf_universe: Optional[List[str]] = None, lookback_months: int = 6,
                 config_path: str = "config.yaml", mode: str = "etf",
                 stock_symbols: Optional[List[str]] = None):
        """
        Initialize the backtester.

        Mirrors the live agent's modes (etf / stock / all / owned-etf). When
        ``etf_universe`` is omitted, the universe is derived from ``mode`` and
        the data files, exactly like ``ETFSwingAgent``.

        Args:
            etf_universe: List of symbols to consider. If None, derived from
                ``mode`` (config etf_universe, list_of_stocks_to_review_for_purchase.dat, or
                currently_own_etf.dat).
            lookback_months: Months of historical data to use for evaluation
            config_path: Path to YAML configuration file
            mode: 'etf', 'stock', 'all', or 'owned-etf' (mirrors live agent)
            stock_symbols: Explicit list of which universe symbols are stocks.
                If None, inferred from ``mode`` (all symbols in 'stock' mode,
                none otherwise). Used to apply stock-specific scoring/ranking.
        """
        self.lookback_months = lookback_months
        self.mode = mode

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # --- Universe derivation (mirrors ETFSwingAgent.__init__) ---
        if etf_universe is None:
            if mode == "stock":
                etf_universe = _load_symbols_from_file('list_of_stocks_to_review_for_purchase.dat')
                if not etf_universe:
                    etf_universe = config['etf_universe']
            elif mode == "owned-etf":
                owned = _load_symbols_from_file('currently_own_etf.dat')
                if not owned:
                    raise ValueError(
                        "No symbols found in currently_own_etf.dat. "
                        "Add at least one ETF (one symbol per line) to use mode 'owned-etf'."
                    )
                etf_universe = owned
            elif mode == "all":
                etf_symbols = set(config['etf_universe'])
                stock_syms = set(_load_symbols_from_file('list_of_stocks_to_review_for_purchase.dat'))
                etf_universe = list(etf_symbols.union(stock_syms))
            else:  # 'etf'
                etf_universe = config['etf_universe']

        self.etf_universe = etf_universe

        # --- Stock symbol set (drives stock-specific scoring/ranking) ---
        if stock_symbols is not None:
            self._stock_symbols = set(s.upper() for s in stock_symbols)
        elif mode == "stock":
            self._stock_symbols = set(s.upper() for s in etf_universe)
        else:
            self._stock_symbols = set()

        # Extract weights from config
        self.technical_weights = config.get('technical_weights', {})
        self.fundamental_weights = config.get('fundamental_weights', {})
        
        # Risk management config
        self.risk_config = config.get('risk', {})
        self.stop_loss_atr_mult = self.risk_config.get('stop_loss_atr_mult', 2.0)
        self.take_profit_atr_mult = self.risk_config.get('take_profit_atr_mult', 3.0)
        self.max_correlated_positions = self.risk_config.get('max_correlated_positions', 3)
        self.correlation_threshold = 0.8  # Max pairwise correlation allowed (relaxed from 0.7)
        
        # Backtest-specific config
        bt_config = config.get('backtest', {})
        self.composite_weights = bt_config.get('composite_weights', {
            'technical': 0.50, 'fundamental': 0.30, 'sentiment': 0.20
        })
        self.benchmark_symbol = bt_config.get('benchmark_symbol', 'SPY')
        self.top_n = bt_config.get('top_n', 5)
        self.min_rank_score = float(bt_config.get('min_rank_score', 0.0))
        self.position_sizing_method = bt_config.get('position_sizing_method', 'equal')
        if self.position_sizing_method not in {'equal', 'score_weighted'}:
            raise ValueError(
                "backtest.position_sizing_method must be 'equal' or 'score_weighted'."
            )
        self.oos_tuning_profiles: Dict[str, Dict[str, Any]] = bt_config.get(
            'oos_tuning_profiles', {}
        )
        self.sentiment_mode = bt_config.get('sentiment_mode', 'off')
        if self.sentiment_mode not in {'off', 'price_proxy'}:
            raise ValueError(
                "backtest.sentiment_mode must be 'off' or 'price_proxy'. "
                "Historical live-news sentiment is not reproducible."
            )
        configured_factors = bt_config.get(
            'factor_ablation_factors',
            ['technical', 'fundamental', 'sentiment', 'relative_strength'],
        )
        valid_factors = {'technical', 'fundamental', 'sentiment', 'relative_strength'}
        invalid_factors = set(configured_factors) - valid_factors
        if invalid_factors:
            raise ValueError(
                "backtest.factor_ablation_factors contains unsupported factor(s): "
                f"{', '.join(sorted(invalid_factors))}"
            )
        self.factor_ablation_factors: List[str] = list(configured_factors)
        self._ablated_factors: set[str] = set()
        
        # Transaction cost config
        total_txn = bt_config.get('total_txn_cost_bps', 0.0)
        if total_txn > 0:
            self.txn_cost_bps: float = float(total_txn)
        else:
            self.txn_cost_bps = float(
                bt_config.get('slippage_bps', 5.0) +
                bt_config.get('bid_ask_spread_bps', 3.0) +
                bt_config.get('commission_per_share', 0.0) * 0  # per-share ignored in bps model
            )
        # Convert bps to decimal (e.g., 8 bps -> 0.0008)
        self.txn_cost: float = self.txn_cost_bps / 10000.0
        
        # Track previous holdings for turnover calculation
        self._prev_holdings: Optional[List[str]] = None

        # Cache for data to avoid redundant downloads (LRU with TTL, matching live agent)
        # Each entry: {'data': DataFrame, 'fetched_at': datetime, 'period': str}
        # Using OrderedDict for LRU eviction
        self._data_cache: OrderedDict = OrderedDict()
        self._cache_maxsize = 100  # Maximum number of cached entries
        self._cache_lock = None  # No threading in backtest, so no lock needed
        
        # Cache TTL by period (shorter periods = fresher data needed)
        self._cache_ttl = {
            '1mo': timedelta(hours=1),
            '3mo': timedelta(hours=4),
            '6mo': timedelta(hours=8),
            '1y': timedelta(hours=24),
            '2y': timedelta(hours=48),
        }
        self._default_cache_ttl = timedelta(hours=6)
        
        # Market regime config
        self.market_regime_config = config.get('market_regime', {})

        # Owned ETFs (from currently_own_etf.dat) — used to flag holdings,
        # matching the live agent's owned-ETF concept.
        self._owned_etf_symbols = set(_load_symbols_from_file('currently_own_etf.dat'))

    def _fetch_window(self, symbol: str, start_date: datetime,
                      end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch (and cache) an ETF OHLCV window between two dates with LRU/TTL."""
        cache_key = (symbol, start_date.date(), end_date.date())
        now = datetime.now()
        
        # Check cache with TTL
        if cache_key in self._data_cache:
            entry = self._data_cache[cache_key]
            # TTL based on period length
            days = (end_date - start_date).days
            if days <= 31:
                ttl = self._cache_ttl['1mo']
            elif days <= 92:
                ttl = self._cache_ttl['3mo']
            elif days <= 183:
                ttl = self._cache_ttl['6mo']
            elif days <= 365:
                ttl = self._cache_ttl['1y']
            else:
                ttl = self._cache_ttl['2y']
            
            age = now - entry['fetched_at']
            if age < ttl:
                # Move to end for LRU (most recently used)
                self._data_cache.move_to_end(cache_key)
                return entry['data']  # type: ignore[no-any-return]
            else:
                # Remove expired entry
                del self._data_cache[cache_key]

        try:
            ticker = yf.Ticker(symbol)
            data = retry_call(ticker.history, start=start_date, end=end_date, auto_adjust=True)
            if data.empty:
                logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
                self._data_cache[cache_key] = {'data': None, 'fetched_at': now}
                return None
            
            # LRU eviction if cache is full
            while len(self._data_cache) >= self._cache_maxsize:
                oldest_key = next(iter(self._data_cache))
                del self._data_cache[oldest_key]
            
            self._data_cache[cache_key] = {'data': data, 'fetched_at': now}
            return data  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} after retries: {e}")
            return None

    def fetch_historical_data(self, symbol: str, end_date: datetime, 
                            period_months: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for an ETF up to a specific end date.
        
        Args:
            symbol: ETF symbol
            end_date: End date for historical data (inclusive)
            period_months: Number of months of lookback data (defaults to self.lookback_months)
            
        Returns:
            DataFrame with historical OHLCV data
        """
        if period_months is None:
            period_months = self.lookback_months
            
        start_date = end_date - timedelta(days=period_months*30)
        return self._fetch_window(symbol, start_date, end_date)
    
    def fetch_period_data(self, symbol: str, start_date: datetime,
                          end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for an ETF between two specific dates.
        
        Unlike fetch_historical_data (which always looks back lookback_months),
        this fetches exactly the window between start_date and end_date.
        Used for computing period returns without look-ahead bias.
        
        Args:
            symbol: ETF symbol
            start_date: Start of the holding period
            end_date: End of the holding period
            
        Returns:
            DataFrame with historical OHLCV data for the exact period
        """
        return self._fetch_window(symbol, start_date, end_date)
    
    def get_sentiment_score(self, symbol: str, end_date: datetime) -> float:
        """
        Get the configured reproducible backtest sentiment score.

        ``off`` returns the neutral 0.5 score and is the default because the
        live agent's current-news TextBlob score is not historically
        reproducible. ``price_proxy`` uses trailing price movement as an
        explicitly separate, reproducible experimental factor.
        """
        if self.sentiment_mode == 'off':
            return 0.5

        # Explicit historical price-momentum proxy; this is not live-news sentiment.
        data = self.fetch_historical_data(symbol, end_date, period_months=1)
        if data is None or len(data) < 5:
            return 0.5
        
        # Simple sentiment based on recent price action (1-month momentum)
        recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) if len(data) >= 5 else 0
        
        # Convert return to sentiment score (0-1)
        # -5% return = 0.0, +5% return = 1.0
        sentiment = 0.5 + recent_return * 10
        return float(min(max(sentiment, 0), 1))

    @staticmethod
    def _filter_holding_window(
        data: pd.DataFrame,
        signal_date: datetime,
        exit_date: datetime,
    ) -> pd.DataFrame:
        """Return bars strictly after the signal close through the exit date."""
        signal_ts = pd.Timestamp(signal_date).normalize()
        exit_ts = pd.Timestamp(exit_date).normalize()
        normalized_index = pd.DatetimeIndex(data.index).normalize()
        return data.loc[(normalized_index > signal_ts) & (normalized_index <= exit_ts)]

    def _get_next_session_entry(
        self,
        symbol: str,
        signal_date: datetime,
        exit_date: datetime,
    ) -> Optional[Tuple[float, datetime]]:
        """Return the first tradable next-session open after a signal close."""
        data = self.fetch_period_data(symbol, signal_date, exit_date + timedelta(days=1))
        if data is None or data.empty:
            return None

        holding_data = self._filter_holding_window(data, signal_date, exit_date)
        if holding_data.empty:
            return None

        entry_row = holding_data.iloc[0]
        entry_price = float(entry_row['Open'])
        if not np.isfinite(entry_price) or entry_price <= 0:
            return None
        return entry_price, holding_data.index[0].to_pydatetime()
    
    def _get_market_regime(self, end_date: datetime) -> Dict[str, Any]:
        """
        Determine market regime based on SPY price vs 200-day SMA and VIX.
        Matches the live agent's market regime detection logic.
        
        Returns: dict with 'regime' and 'volatility' keys
        """
        try:
            spy_data = self.fetch_historical_data("SPY", end_date, period_months=12)
            if spy_data is None or len(spy_data) < 200:
                return {"regime": "sideways", "volatility": "unknown"}
            
            close = spy_data['Close']
            sma_200 = close.rolling(200).mean().iloc[-1]
            price = close.iloc[-1]
            price_vs_sma200 = price / sma_200 - 1
            
            # Trend determination (matching live agent)
            trend_threshold = self.market_regime_config.get('trend_threshold', 0.02)
            if price_vs_sma200 > trend_threshold:
                trend = "bull"
            elif price_vs_sma200 < -trend_threshold:
                trend = "bear"
            else:
                trend = "sideways"
            
            # Volatility regime (using VIX if available)
            try:
                vix_data = retry_call(yf.Ticker("^VIX").history, period="1mo")
                if not vix_data.empty:
                    vix_level = vix_data['Close'].iloc[-1]
                    vix_high = self.market_regime_config.get('vix_high_threshold', 30)
                    vix_low = self.market_regime_config.get('vix_low_threshold', 15)
                    if vix_level > vix_high:
                        vol_regime = "high"
                    elif vix_level < vix_low:
                        vol_regime = "low"
                    else:
                        vol_regime = "moderate"
                else:
                    vol_regime = "unknown"
            except Exception:
                vol_regime = "unknown"
            
            return {"regime": trend, "volatility": vol_regime, "price_vs_sma200": price_vs_sma200}
        except Exception:
            return {"regime": "sideways", "volatility": "unknown"}
    
    def evaluate_etf(self, symbol: str, end_date: datetime) -> Dict[str, Any]:
        """Evaluate a single ETF as of a specific date.

        Mirrors the live agent's ETF evaluation: composite score for context,
        plus the 4-week growth outlook (primary ranking criteria), dividend
        yield, and an owned-ETF flag (from currently_own_etf.dat).
        """
        # Validate symbol format
        if not validate_etf_symbol(symbol):
            return {"symbol": symbol, "error": "Invalid symbol format"}
        
        # Fetch data up to end_date
        data = self.fetch_historical_data(symbol, end_date)
        if data is None:
            return {"symbol": symbol, "error": "No data available"}
        
        # Calculate components
        technical_indicators = calculate_technical_indicators(data)
        technical_score = calculate_technical_score(technical_indicators, self.technical_weights)
        if 'technical' in self._ablated_factors:
            technical_score = 0.5
        
        # Fetch benchmark data for tracking error calculation
        bench_data = self.fetch_historical_data(self.benchmark_symbol, end_date)
        fundamental_score = calculate_fundamental_score(
            symbol, self.fundamental_weights, price_data=data, benchmark_data=bench_data
        )
        sentiment_score = self.get_sentiment_score(symbol, end_date)
        if 'fundamental' in self._ablated_factors:
            fundamental_score = 0.5
        if 'sentiment' in self._ablated_factors:
            sentiment_score = 0.5
        
        # Composite score using config weights
        composite_score = (
            technical_score * self.composite_weights['technical'] +
            fundamental_score * self.composite_weights['fundamental'] +
            sentiment_score * self.composite_weights['sentiment']
        )
        
        # Current price info (as of end_date)
        current_price = data['Close'].iloc[-1]
        atr = technical_indicators.get('atr_14', 0.0)
        price_change_1w = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100 if len(data) >= 5 else 0.0
        
        # --- 4-week growth outlook (PRIMARY ranking criteria, matches live agent) ---
        growth_outlook = None
        short_data = self.fetch_historical_data(symbol, end_date, period_months=6)
        if short_data is not None and len(short_data) >= 20:
            short_ind = calculate_short_term_indicators(short_data)
            spy_short = self.fetch_historical_data(self.benchmark_symbol, end_date, period_months=6)
            spy_ind = calculate_short_term_indicators(spy_short) if (spy_short is not None and len(spy_short) >= 20) else None
            regime = self._get_market_regime(end_date)
            growth_outlook = calculate_4week_growth_outlook(
                short_ind,
                spy_indicators=spy_ind,
                sentiment_score=sentiment_score,
                price_change_1w=price_change_1w,
                market_regime=regime.get('regime', 'bull'),
            )
        
        # --- Dividend yield (computed from trailingAnnualDividendRate / price) ---
        dividend_yield_pct = None
        try:
            info = retry_call(yf.Ticker(symbol).info.get, None)
            if info:
                rate = info.get('trailingAnnualDividendRate')
                price = info.get('regularMarketPrice') or info.get('currentPrice')
                if rate and price:
                    dividend_yield_pct = (rate / price) * 100
                if dividend_yield_pct is None:
                    raw = info.get('dividendYield', 0) or 0
                    # Yahoo's dividendYield is already a percentage, so use as-is
                    dividend_yield_pct = raw
        except Exception:
            dividend_yield_pct = None
        
        # --- Owned-ETF flag (from currently_own_etf.dat) ---
        is_owned = symbol in self._owned_etf_symbols
        
        return {
            "symbol": symbol,
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "sentiment_score": sentiment_score,
            "composite_score": composite_score,
            "growth_outlook": growth_outlook,
            "dividend_yield_pct": dividend_yield_pct,
            "is_owned_etf": is_owned,
            "current_price": current_price,
            "price_change_1w": price_change_1w,
            "atr": atr,
            "technical_indicators": technical_indicators,
            "data_points": len(data)
        }

    def evaluate_stock(self, symbol: str, end_date: datetime) -> Dict[str, Any]:
        """Evaluate a single stock as of a specific date.

        Mirrors the live agent's stock path: stock-specific technical weights
        (reduced mean-reversion, boosted momentum/trend), a fundamental score
        using only liquidity + 1y Sharpe (ETF fields are meaningless for
        stocks), a composite blend of 80% technical / 10% fundamental /
        10% sentiment, and a short-term (days-scale) score that drives ranking.
        """
        # Validate symbol format
        if not validate_etf_symbol(symbol):
            return {"symbol": symbol, "error": "Invalid symbol format"}

        # Fetch data up to end_date
        data = self.fetch_historical_data(symbol, end_date)
        if data is None:
            return {"symbol": symbol, "error": "No data available"}

        # --- Technical score with stock-specific weights ---
        technical_indicators = calculate_technical_indicators(data)
        stock_tech_weights = dict(self.technical_weights)
        stock_tech_weights['mean_reversion_score'] = 0.05
        stock_tech_weights['momentum_score'] = 0.35
        stock_tech_weights['trend_score'] = 0.35
        technical_score = calculate_technical_score(technical_indicators, stock_tech_weights)
        if 'technical' in self._ablated_factors:
            technical_score = 0.5

        # Fetch benchmark data for tracking error / relative strength
        bench_data = self.fetch_historical_data(self.benchmark_symbol, end_date)
        fundamental_score = calculate_fundamental_score(
            symbol, self.fundamental_weights, price_data=data,
            benchmark_data=bench_data, is_stock=True
        )
        sentiment_score = self.get_sentiment_score(symbol, end_date)
        if 'fundamental' in self._ablated_factors:
            fundamental_score = 0.5
        if 'sentiment' in self._ablated_factors:
            sentiment_score = 0.5

        # Composite: stocks use 80/10/10 (ETF-centric fields are meaningless)
        composite_score = (
            technical_score * 0.80 +
            fundamental_score * 0.10 +
            sentiment_score * 0.10
        )

        # Current price info (as of end_date)
        current_price = data['Close'].iloc[-1]
        atr = technical_indicators.get('atr_14', 0.0)
        price_change_1w = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100 if len(data) >= 5 else 0.0

        # --- Short-term (days-scale) score — PRIMARY ranking criteria for stocks ---
        short_term_score = 0.0
        short_data = self.fetch_historical_data(symbol, end_date, period_months=6)
        if short_data is not None and len(short_data) >= 20:
            short_ind = calculate_short_term_indicators(short_data)
            spy_short = self.fetch_historical_data(self.benchmark_symbol, end_date, period_months=6)
            spy_ind = calculate_short_term_indicators(spy_short) if (spy_short is not None and len(spy_short) >= 20) else None
            short_term_score = calculate_short_term_score(short_ind, spy_indicators=spy_ind)

        # --- Dividend yield (computed from trailingAnnualDividendRate / price) ---
        dividend_yield_pct = None
        try:
            info = retry_call(yf.Ticker(symbol).info.get, None)
            if info:
                rate = info.get('trailingAnnualDividendRate')
                price = info.get('regularMarketPrice') or info.get('currentPrice')
                if rate and price:
                    dividend_yield_pct = (rate / price) * 100
                if dividend_yield_pct is None:
                    raw = info.get('dividendYield', 0) or 0
                    dividend_yield_pct = raw * 100 if 0 < raw <= 1 else raw
        except Exception:
            dividend_yield_pct = None

        # Stocks are never "owned ETFs"
        is_owned = False

        return {
            "symbol": symbol,
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "sentiment_score": sentiment_score,
            "composite_score": composite_score,
            "short_term_score": short_term_score,
            "growth_outlook": None,
            "dividend_yield_pct": dividend_yield_pct,
            "is_owned_etf": is_owned,
            "current_price": current_price,
            "price_change_1w": price_change_1w,
            "atr": atr,
            "data_points": len(data)
        }

    def evaluate_symbol(self, symbol: str, end_date: datetime) -> Dict[str, Any]:
        """Evaluate a symbol, dispatching to stock or ETF logic.

        Mirrors the live agent: a symbol in ``self._stock_symbols`` is scored
        as a stock (ranked by short-term score); otherwise as an ETF (ranked by
        growth outlook).
        """
        if symbol in self._stock_symbols:
            return self.evaluate_stock(symbol, end_date)
        return self.evaluate_etf(symbol, end_date)
    
    def run_backtest(self, start_date: datetime, end_date: datetime, 
                    rebalancing_freq: str = 'M') -> Dict[str, Any]:
        """
        Run the backtest over a date range.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            rebalancing_freq: Rebalancing frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
            
        Returns:
            Dictionary with backtest results including returns, metrics, and equity curve
        """
        logger.info(f"Running backtest from {start_date.date()} to {end_date.date()} with {rebalancing_freq} rebalancing")
        logger.info(f"Transaction cost: {self.txn_cost_bps:.1f} bps ({self.txn_cost:.4%}) per one-way trade")
        
        # Reset holdings tracker for fresh backtest
        self._prev_holdings = None
        
        # Generate rebalancing dates
        if rebalancing_freq == 'D':
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        elif rebalancing_freq == 'W':
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='W')  # Weekly
        else:  # Monthly
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='BME')  # Business month end
        
        # Ensure we have at least two dates to calculate returns
        if len(rebalancing_dates) < 2:
            logger.error("Not enough rebalancing dates for backtest")
            return {"error": "Insufficient date range"}
        
        # Initialize returns series
        strategy_returns = []
        benchmark_returns = []
        portfolio_history = []
        total_txn_costs = []  # Track transaction costs per period
        
        # Track holdings for minimum holding period logic
        self._holding_periods: Dict[str, int] = {}  # symbol -> months held
        
        # Loop through rebalancing periods
        for i in range(len(rebalancing_dates) - 1):
            current_date = rebalancing_dates[i]
            next_date = rebalancing_dates[i + 1]
            period_regime = self._get_market_regime(current_date).get('regime', 'unknown')
            
            logger.info(f"Rebalancing on {current_date.date()}")
            
            # Evaluate all symbols as of current_date (stocks + ETFs)
            etf_scores = []
            for symbol in self.etf_universe:
                try:
                    result = self.evaluate_symbol(symbol, current_date)
                    if "error" not in result:
                        etf_scores.append(result)
                    else:
                        logger.warning(f"Skipping {symbol} on {current_date.date()}: {result['error']}")
                except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as e:
                    logger.error(f"Error evaluating {symbol} on {current_date.date()}: {e}")

            # Rank by asset-appropriate score (stocks: short_term_score;
            # ETFs: growth_score primary, composite fallback). Mirrors the
            # live agent's run_screening() ranking, including the 20%
            # relative-strength percentile blend.
            self._apply_relative_strength(etf_scores)
            etf_scores.sort(key=lambda x: x['_rank_score'], reverse=True)

            eligible_scores = [
                score for score in etf_scores
                if score['_rank_score'] >= self.min_rank_score
            ]
            if not eligible_scores and etf_scores:
                logger.info(
                    "No symbols met min_rank_score=%.3f; retaining the top candidate.",
                    self.min_rank_score,
                )
                eligible_scores = etf_scores[:1]

            # Log diagnostic info: how many symbols scored and the threshold
            etf_threshold = self.risk_config.get('min_score_threshold', 0.55)
            stock_threshold = 0.35
            logger.info(
                f"  {len(etf_scores)} symbols scored "
                f"(etf_thresh={etf_threshold:.2f}, stock_thresh={stock_threshold:.2f})"
            )

            filtered_scores = self._apply_correlation_filter(eligible_scores, current_date)
            logger.info(f"  After correlation filter: {len(filtered_scores)} symbols (removed {len(eligible_scores) - len(filtered_scores)})")

            # Apply minimum holding period logic to reduce turnover
            top_etfs = self._apply_holding_period_filter(filtered_scores, current_date)
            new_holdings = [etf['symbol'] for etf in top_etfs]

            # Apply position sizing (matching live agent)
            self._apply_position_sizing(top_etfs)

            # Log selected symbols with their ranking scores
            if top_etfs:
                selected_str = ', '.join(
                    f'{e["symbol"]}(rank={e.get("_rank_score", self._rank_score_for(e)):.3f})'
                    for e in top_etfs
                )
                logger.info(f"  Selected: {selected_str}")
            
            # Calculate turnover and transaction costs
            period_txn_cost = self._calculate_turnover_cost(new_holdings)
            total_txn_costs.append(period_txn_cost)
            self._prev_holdings = new_holdings
            
            if not top_etfs:
                logger.warning(f"No valid ETFs found for {current_date.date()}, skipping period")
                period_return = 0.0
                benchmark_period_return = 0.0
                gross_return = 0.0
                stop_loss_hits = 0
                take_profit_hits = 0
            else:
                # Signals use the rebalance close. Trades enter at the first
                # next-session open; no same-close execution is permitted.
                period_returns = []
                stop_loss_hits = 0
                take_profit_hits = 0
                for etf in top_etfs:
                    symbol = etf['symbol']
                    atr = etf.get('atr', 0.0)
                    entry = self._get_next_session_entry(symbol, current_date, next_date)
                    if entry is None:
                        logger.warning(f"No next-session entry available for {symbol} on {current_date.date()}")
                        continue
                    entry_price, _entry_date = entry
                    
                    etf_return = self._simulate_stop_loss_take_profit(
                        symbol, entry_price, atr, current_date, next_date
                    )
                    
                    # Track which level was hit (for reporting)
                    if atr > 0:
                        target_return = self.take_profit_atr_mult * atr / entry_price
                        stop_return = -self.stop_loss_atr_mult * atr / entry_price
                        if etf_return <= stop_return * 0.99:
                            stop_loss_hits += 1
                        elif etf_return >= target_return * 0.99:
                            take_profit_hits += 1
                    
                    period_returns.append(etf_return)
                
                # Apply selected portfolio weights to realized returns.
                if period_returns:
                    executed_weights = [
                        float(etf.get('position_pct', 0.0))
                        for etf in top_etfs[:len(period_returns)]
                    ]
                    if len(executed_weights) == len(period_returns) and sum(executed_weights) > 0:
                        gross_return = float(np.average(period_returns, weights=executed_weights))
                    else:
                        gross_return = float(np.mean(period_returns))
                else:
                    gross_return = 0.0
                
                # Apply transaction costs to get net return
                period_return = gross_return - period_txn_cost
                
                # Benchmark uses the identical next-session-open to exit-close window.
                spy_data = self.fetch_period_data(
                    self.benchmark_symbol, current_date, next_date + timedelta(days=1)
                )
                if spy_data is None or spy_data.empty:
                    benchmark_period_return = 0.0
                else:
                    holding_spy = self._filter_holding_window(spy_data, current_date, next_date)
                    if holding_spy.empty:
                        benchmark_period_return = 0.0
                    else:
                        spy_start = holding_spy['Open'].iloc[0]
                        spy_end = holding_spy['Close'].iloc[-1]
                        benchmark_period_return = (spy_end / spy_start) - 1
            
            strategy_returns.append(period_return)
            benchmark_returns.append(benchmark_period_return)
            
            # Record holdings for this period
            portfolio_history.append({
                'rebalancing_date': current_date,
                'market_regime': period_regime,
                'holdings': new_holdings,
                'execution_assumption': 'next_session_open_to_exit_close',
                'sentiment_mode': self.sentiment_mode,
                'scores': [etf['composite_score'] for etf in top_etfs],
                'growth_scores': [
                    etf.get('growth_outlook', {}).get('growth_score', 0.0)
                    for etf in top_etfs
                ],
                'short_term_scores': [
                    etf.get('short_term_score', 0.0) for etf in top_etfs
                ],
                'gross_return': gross_return,
                'txn_cost': period_txn_cost,
                'period_return': period_return,
                'benchmark_return': benchmark_period_return,
                'stop_loss_hits': stop_loss_hits,
                'take_profit_hits': take_profit_hits,
            })
        
        # Calculate cumulative returns
        strategy_cumulative = np.prod([1 + r for r in strategy_returns]) - 1
        benchmark_cumulative = np.prod([1 + r for r in benchmark_returns]) - 1
        
        # Calculate annualized metrics
        trading_days_per_year = 252
        periods_per_year = trading_days_per_year / np.mean([(next_date - current_date).days 
                                                           for current_date, next_date in zip(rebalancing_dates[:-1], rebalancing_dates[1:])])
        
        # Annualized return
        strategy_annual_return = (1 + strategy_cumulative) ** (periods_per_year / len(strategy_returns)) - 1 if strategy_returns else 0
        benchmark_annual_return = (1 + benchmark_cumulative) ** (periods_per_year / len(benchmark_returns)) - 1 if benchmark_returns else 0
        
        # Annualized volatility
        strategy_annual_vol = np.std(strategy_returns) * np.sqrt(periods_per_year) if len(strategy_returns) > 1 else 0
        benchmark_annual_vol = np.std(benchmark_returns) * np.sqrt(periods_per_year) if len(benchmark_returns) > 1 else 0
        
        # Sharpe ratio (assuming risk-free rate = 0%)
        strategy_sharpe = strategy_annual_return / strategy_annual_vol if strategy_annual_vol > 0 else 0
        benchmark_sharpe = benchmark_annual_return / benchmark_annual_vol if benchmark_annual_vol > 0 else 0
        
        # --- Additional Performance Metrics ---
        
        # Sortino ratio: uses downside deviation (only returns below 0)
        downside_returns = [r for r in strategy_returns if r < 0]
        if downside_returns:
            downside_deviation = np.std(downside_returns) * np.sqrt(periods_per_year)
            strategy_sortino = strategy_annual_return / downside_deviation if downside_deviation > 0 else 0
        else:
            strategy_sortino = float('inf') if strategy_annual_return > 0 else 0
        
        benchmark_downside = [r for r in benchmark_returns if r < 0]
        if benchmark_downside:
            bench_downside_dev = np.std(benchmark_downside) * np.sqrt(periods_per_year)
            benchmark_sortino = benchmark_annual_return / bench_downside_dev if bench_downside_dev > 0 else 0
        else:
            benchmark_sortino = float('inf') if benchmark_annual_return > 0 else 0
        
        # Win rate and average win/loss ratio
        winning_periods = [r for r in strategy_returns if r > 0]
        losing_periods = [r for r in strategy_returns if r < 0]
        win_rate = len(winning_periods) / len(strategy_returns) if strategy_returns else 0
        avg_win = float(np.mean(winning_periods)) if winning_periods else 0.0
        avg_loss = float(abs(np.mean(losing_periods))) if losing_periods else 0.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else (float('inf') if avg_win > 0 else 0)
        
        # Information ratio: excess return over benchmark / tracking error
        excess_returns = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
        tracking_error = np.std(excess_returns) * np.sqrt(periods_per_year) if len(excess_returns) > 1 else 0
        mean_excess = np.mean(excess_returns) * periods_per_year if excess_returns else 0
        information_ratio = mean_excess / tracking_error if tracking_error > 0 else 0
        
        # Maximum drawdown
        strategy_cumulative_returns = np.cumprod([1 + r for r in strategy_returns])
        strategy_running_max = np.maximum.accumulate(strategy_cumulative_returns)
        strategy_drawdown = (strategy_cumulative_returns - strategy_running_max) / strategy_running_max
        strategy_max_drawdown = np.min(strategy_drawdown) if len(strategy_drawdown) > 0 else 0
        
        benchmark_cumulative_returns = np.cumprod([1 + r for r in benchmark_returns])
        benchmark_running_max = np.maximum.accumulate(benchmark_cumulative_returns)
        benchmark_drawdown = (benchmark_cumulative_returns - benchmark_running_max) / benchmark_running_max
        benchmark_max_drawdown = np.min(benchmark_drawdown) if len(benchmark_drawdown) > 0 else 0
        
        # Calmar ratio: annualized return / max drawdown (must be after drawdown calc)
        strategy_calmar = strategy_annual_return / abs(strategy_max_drawdown) if strategy_max_drawdown < 0 else (float('inf') if strategy_annual_return > 0 else 0)
        benchmark_calmar = benchmark_annual_return / abs(benchmark_max_drawdown) if benchmark_max_drawdown < 0 else (float('inf') if benchmark_annual_return > 0 else 0)
        
        # Prepare results
        total_txn_cost_sum = sum(total_txn_costs)
        total_stop_loss_hits = sum(int(p.get('stop_loss_hits', 0)) for p in portfolio_history)  # type: ignore[call-overload]
        total_take_profit_hits = sum(int(p.get('take_profit_hits', 0)) for p in portfolio_history)  # type: ignore[call-overload]
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'rebalancing_freq': rebalancing_freq,
            'num_periods': len(strategy_returns),
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'strategy_cumulative_return': strategy_cumulative,
            'benchmark_cumulative_return': benchmark_cumulative,
            'strategy_annual_return': strategy_annual_return,
            'benchmark_annual_return': benchmark_annual_return,
            'strategy_annual_volatility': strategy_annual_vol,
            'benchmark_annual_volatility': benchmark_annual_vol,
            'strategy_sharpe_ratio': strategy_sharpe,
            'benchmark_sharpe_ratio': benchmark_sharpe,
            'strategy_max_drawdown': strategy_max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'portfolio_history': portfolio_history,
            'txn_cost_bps': self.txn_cost_bps,
            'total_txn_costs': total_txn_cost_sum,
            'avg_turnover_pct': self._calc_avg_turnover(portfolio_history),
            'stop_loss_hits': total_stop_loss_hits,
            'take_profit_hits': total_take_profit_hits,
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            # Additional performance metrics
            'strategy_sortino_ratio': strategy_sortino,
            'benchmark_sortino_ratio': benchmark_sortino,
            'strategy_calmar_ratio': strategy_calmar,
            'benchmark_calmar_ratio': benchmark_calmar,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'information_ratio': information_ratio,
        }
        
        # Run naive baseline comparison for context
        baseline_results = self._run_naive_baseline(start_date, end_date, rebalancing_freq)
        if "error" not in baseline_results:
            results.update(baseline_results)
        
        return results
    
    def _calculate_turnover_cost(self, new_holdings: List[str]) -> float:
        """
        Calculate transaction cost based on turnover from previous holdings.
        
        Turnover is the fraction of the portfolio that changed. Cost is applied
        to both sells (exiting old positions) and buys (entering new positions).
        
        Args:
            new_holdings: List of ETF symbols in the new portfolio
            
        Returns:
            Transaction cost as a decimal fraction of portfolio value
        """
        if self._prev_holdings is None:
            # First period: cost of entering all positions (one-way)
            return self.txn_cost
        
        prev_set = set(self._prev_holdings)
        new_set = set(new_holdings)
        
        if not prev_set:
            return self.txn_cost
        
        # Number of positions that changed
        sold = len(prev_set - new_set)
        bought = len(new_set - prev_set)
        total_changes = sold + bought
        
        # Turnover as fraction of portfolio (each position is 1/N of portfolio)
        n = max(len(self._prev_holdings), len(new_holdings))
        if n == 0:
            return 0.0
        
        # Each changed position costs txn_cost * (1/n) of portfolio
        # Both selling old and buying new incur costs
        turnover_fraction = total_changes / n
        return float(turnover_fraction * self.txn_cost)
    
    def _calc_avg_turnover(self, portfolio_history: List[Dict]) -> float:
        """Calculate average turnover percentage across all periods."""
        if len(portfolio_history) < 2:
            return 0.0
        
        turnovers = []
        for i in range(1, len(portfolio_history)):
            prev = set(portfolio_history[i-1]['holdings'])
            curr = set(portfolio_history[i]['holdings'])
            if not prev:
                continue
            changes = len(prev - curr) + len(curr - prev)
            n = max(len(portfolio_history[i-1]['holdings']), len(portfolio_history[i]['holdings']))
            if n > 0:
                turnovers.append(changes / n)
        
        return float(np.mean(turnovers)) * 100 if turnovers else 0.0
    
    def _rank_score_for(self, result: Dict[str, Any]) -> float:
        """Compute the ranking score for a single evaluated symbol.

        Mirrors the live agent's ranking: stocks rank by ``short_term_score``
        (days-scale), ETFs rank by ``growth_score`` (primary) with a
        ``composite_score`` fallback when growth is unavailable. Both are on a
        0–1 scale so they are comparable in 'all' mode.

        Note: The live agent blends a 20% relative-strength percentile into
        the rank score. The backtester applies that blend batch-wide in
        ``_apply_relative_strength()`` before sorting (since percentiles are
        universe-relative and can't be computed per-symbol).
        """
        symbol = result.get('symbol')
        if symbol in self._stock_symbols:
            return float(result.get('short_term_score', 0.0))  # type: ignore[no-any-return]
        growth = result.get('growth_outlook')
        if growth and growth.get('growth_score', 0.0) > 0:
            return float(growth['growth_score'])  # type: ignore[no-any-return]
        return float(result.get('composite_score', 0.0))  # type: ignore[no-any-return]

    def _apply_relative_strength(self, results: List[Dict[str, Any]]) -> None:
        """Blend relative-strength percentiles into each result's rank score.

        Mirrors the live agent's run_screening(): computes percentile ranks
        for key technical indicators across the evaluated universe, then
        blends the composite relative strength into a per-symbol
        ``_relative_strength`` and ``_rank_score`` at 20% weight.

        Mutates ``results`` in place. Call before sorting.
        If technical_indicators are not available (e.g., test data), falls
        back to the base rank score with neutral relative strength.
        """
        if 'relative_strength' in self._ablated_factors:
            for r in results:
                r['_relative_strength'] = 0.5
                r['_rank_score'] = self._rank_score_for(r)
            return

        if len(results) < 3:
            for r in results:
                r['_relative_strength'] = 0.5
                r['_rank_score'] = self._rank_score_for(r)
            return

        indicator_sets: Dict[str, Dict[str, float]] = {}
        for key in ('roc_10', 'obv_trend', 'atr_trend_ratio', 'volume_ratio', 'adx'):
            vals = {}
            for r in results:
                tech = r.get('technical_indicators', {})
                if tech and key in tech and tech[key] is not None:
                    vals[r['symbol']] = tech[key]
            if vals:
                indicator_sets[key] = vals

        rs_percentiles = compute_composite_relative_strength(indicator_sets)

        rs_weight = 0.20
        for r in results:
            rs = rs_percentiles.get(r['symbol'], 0.5)
            r['_relative_strength'] = rs
            base = self._rank_score_for(r)
            r['_rank_score'] = base * (1 - rs_weight) + rs * rs_weight

    def _apply_correlation_filter(self, etf_scores: List[Dict], end_date: datetime) -> List[Dict]:
        """
        Filter ETFs to limit highly correlated positions.
        
        Uses pairwise price correlation over the lookback period. If more than
        max_correlated_positions ETFs are correlated above the threshold,
        lower-scoring ones are removed.
        
        Args:
            etf_scores: List of ETF evaluation results, sorted by composite_score desc
            end_date: Reference date for fetching correlation data
            
        Returns:
            Filtered list respecting correlation limits
        """
        if len(etf_scores) <= 1:
            return etf_scores
        
        # Build correlation matrix from price returns
        symbols = [e['symbol'] for e in etf_scores]
        price_series = {}
        
        for symbol in symbols:
            data = self.fetch_historical_data(symbol, end_date, period_months=self.lookback_months)
            if data is not None and len(data) >= 20:
                price_series[symbol] = data['Close'].pct_change(fill_method=None).dropna()
        
        if len(price_series) < 2:
            return etf_scores
        
        returns_df = pd.DataFrame(price_series)
        corr_matrix = returns_df.corr()
        
        # Greedy selection: start with highest score, add if not too correlated
        selected = []
        for etf in etf_scores:
            symbol = etf['symbol']
            if symbol not in corr_matrix.columns:
                selected.append(etf)
                continue
            
            # Count how many already-selected ETFs this one is highly correlated with
            high_corr_count = 0
            for sel in selected:
                sel_sym = sel['symbol']
                if sel_sym in corr_matrix.columns and symbol in corr_matrix.columns:
                    corr_val = abs(corr_matrix.loc[symbol, sel_sym])
                    if corr_val > self.correlation_threshold:
                        high_corr_count += 1
            
            if high_corr_count < self.max_correlated_positions:
                selected.append(etf)
        
        return selected
    
    def _apply_holding_period_filter(self, etf_scores: List[Dict], current_date: datetime) -> List[Dict]:
        """
        Apply minimum holding period logic to reduce turnover.
        
        If an ETF is already held and has been held for less than 2 months,
        give it a score boost to reduce unnecessary churn. Only replace if
        a new ETF significantly outperforms (score difference > 0.15).
        
        Ranking uses the asset-appropriate score (stocks: short_term_score;
        ETFs: growth score primary, composite fallback), matching the live
        agent's run_screening() ranking.
        
        Args:
            etf_scores: List of symbol evaluation results, sorted by rank score desc
            current_date: Current rebalancing date
            
        Returns:
            Filtered list with holding period logic applied
        """
        # Always rank by the asset-appropriate score before selecting, so the
        # function is self-contained and matches the live agent's ranking.
        self._apply_relative_strength(etf_scores)
        etf_scores = sorted(etf_scores, key=lambda x: x['_rank_score'], reverse=True)

        if not self._prev_holdings:
            return etf_scores[:self.top_n]
        
        # Boost scores for currently held symbols that haven't been held long
        min_holding_months = 2
        score_boost = 0.15  # Minimum score difference to trigger replacement
        
        for etf in etf_scores:
            symbol = etf['symbol']
            if symbol in self._prev_holdings:
                months_held = self._holding_periods.get(symbol, 0)
                if months_held < min_holding_months:
                    etf['composite_score'] += score_boost
        
        # Re-sort after boosting using the asset-appropriate rank key
        self._apply_relative_strength(etf_scores)
        etf_scores.sort(key=lambda x: x['_rank_score'], reverse=True)
        
        # Update holding periods
        for etf in etf_scores[:self.top_n]:
            symbol = etf['symbol']
            self._holding_periods[symbol] = self._holding_periods.get(symbol, 0) + 1
        
        # Clear holding periods for ETFs no longer held
        for symbol in list(self._holding_periods.keys()):
            if symbol not in [e['symbol'] for e in etf_scores[:self.top_n]]:
                del self._holding_periods[symbol]
        
        return etf_scores[:self.top_n]
    
    def _apply_position_sizing(self, results: List[Dict]) -> None:
        """
        Assign position weights using the configured sizing method.
        
        Args:
            results: List of selected ETF results to mutate in place
        """
        max_pct = self.risk_config.get('max_position_pct', 0.20)
        n = len(results)
        if n == 0:
            return

        if self.position_sizing_method == 'score_weighted':
            raw_weights = [max(float(r.get('_rank_score', 0.0)), 0.0) for r in results]
            total = sum(raw_weights)
            if total > 0:
                normalized = [weight / total for weight in raw_weights]
                # Cap exposures and redistribute any remaining budget among
                # positions still below the cap.
                weights = [min(weight, max_pct) for weight in normalized]
                remaining = 1.0 - sum(weights)
                while remaining > 1e-8:
                    eligible = [i for i, weight in enumerate(weights) if weight < max_pct - 1e-8]
                    if not eligible:
                        break
                    allocation = remaining / len(eligible)
                    added = 0.0
                    for i in eligible:
                        increment = min(allocation, max_pct - weights[i])
                        weights[i] += increment
                        added += increment
                    remaining -= added
                for result, weight in zip(results, weights):
                    result['position_pct'] = round(weight, 4)
                return

        # Equal weight capped at max_position_pct, then normalized to 100%.
        raw_weight = min(1.0 / n, max_pct)
        total = raw_weight * n
        for result in results:
            result['position_pct'] = round(raw_weight / total, 4)
    
    def _simulate_stop_loss_take_profit(
        self, symbol: str, entry_price: float, atr: float,
        current_date: datetime, next_date: datetime
    ) -> float:
        """
        Simulate intra-period return with stop-loss and take-profit levels.
        
        Uses daily data within the holding period to check if stop-loss or
        take-profit was triggered. If triggered, returns the capped return
        instead of the full period return.
        
        Market-regime aware: In bull markets, uses wider stops (2.5x ATR)
        to avoid being stopped out by normal volatility.
        
        Args:
            symbol: ETF symbol
            entry_price: Entry price at current_date
            atr: Average True Range at entry
            current_date: Start of holding period
            next_date: End of holding period
            
        Returns:
            Realized return (may be capped by stop/take-profit)
        """
        if atr <= 0:
            # No ATR available, fall back to simple period return
            price_data = self.fetch_period_data(symbol, current_date, next_date + timedelta(days=1))
            if price_data is None or price_data.empty:
                return 0.0
            price_data = self._filter_holding_window(price_data, current_date, next_date)
            if price_data.empty:
                return 0.0
            return float((price_data['Close'].iloc[-1] / entry_price) - 1)
        
        # Market-regime aware stop-loss: wider stops in bull markets
        regime = self._get_market_regime(current_date)
        stop_mult = self.stop_loss_atr_mult
        if regime == "bull":
            stop_mult = max(self.stop_loss_atr_mult, 3.5)  # Wider stops in bull markets (3.5x ATR)
        elif regime == "bear":
            stop_mult = min(self.stop_loss_atr_mult, 1.5)  # Tighter stops in bear markets (1.5x ATR)
        
        stop_loss = entry_price - stop_mult * atr
        take_profit = entry_price + self.take_profit_atr_mult * atr
        
        # Fetch only the post-entry holding window. The signal-date bar is
        # excluded because its close was used to form the signal.
        price_data = self.fetch_period_data(symbol, current_date, next_date + timedelta(days=1))
        if price_data is None or len(price_data) < 2:
            return 0.0
        price_data = self._filter_holding_window(price_data, current_date, next_date)
        if price_data.empty:
            return 0.0
        
        # Walk through daily prices to check for stop/take-profit triggers
        for _, row in price_data.iterrows():
            low_price = row['Low']
            high_price = row['High']
            
            # Check stop-loss first (risk management priority)
            if low_price <= stop_loss:
                return float((stop_loss / entry_price) - 1)
            
            # Check take-profit
            if high_price >= take_profit:
                return float((take_profit / entry_price) - 1)
        
        # Neither triggered — use full period return
        end_price = price_data['Close'].iloc[-1]
        return float((end_price / entry_price) - 1)
    
    @staticmethod
    def _fmt_ratio(value: float) -> str:
        """Format a ratio, handling infinity gracefully."""
        if value == float('inf'):
            return '∞'
        if value == float('-inf'):
            return '-∞'
        return f'{value:.2f}'
    
    def print_backtest_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of backtest results."""
        if "error" in results:
            print(f"Backtest failed: {results['error']}")
            return
        
        print("\n" + "="*60)
        print("ETF SWING TRADING AGENT - BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
        print(f"Rebalancing: {results['rebalancing_freq']}")
        print(f"Number of periods: {results['num_periods']}")
        print("-"*60)
        print("STRATEGY PERFORMANCE:")
        print(f"  Cumulative Return: {results['strategy_cumulative_return']:.2%}")
        print(f"  Annualized Return: {results['strategy_annual_return']:.2%}")
        print(f"  Annualized Volatility: {results['strategy_annual_volatility']:.2%}")
        print(f"  Sharpe Ratio: {results['strategy_sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {self._fmt_ratio(results.get('strategy_sortino_ratio', 0))}")
        print(f"  Calmar Ratio:  {self._fmt_ratio(results.get('strategy_calmar_ratio', 0))}")
        print(f"  Max Drawdown: {results['strategy_max_drawdown']:.2%}")
        print(f"  Win Rate: {results.get('win_rate', 0):.1%}  |  Avg Win: {results.get('avg_win', 0):.2%}  |  Avg Loss: {results.get('avg_loss', 0):.2%}")
        print(f"  Win/Loss Ratio: {self._fmt_ratio(results.get('win_loss_ratio', 0))}")
        print("-"*60)
        print("BENCHMARK (SPY) PERFORMANCE:")
        print(f"  Cumulative Return: {results['benchmark_cumulative_return']:.2%}")
        print(f"  Annualized Return: {results['benchmark_annual_return']:.2%}")
        print(f"  Annualized Volatility: {results['benchmark_annual_volatility']:.2%}")
        print(f"  Sharpe Ratio: {results['benchmark_sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {self._fmt_ratio(results.get('benchmark_sortino_ratio', 0))}")
        print(f"  Calmar Ratio:  {self._fmt_ratio(results.get('benchmark_calmar_ratio', 0))}")
        print(f"  Max Drawdown: {results['benchmark_max_drawdown']:.2%}")
        print("-"*60)
        print("OUTPERFORMANCE:")
        print(f"  Excess Return: {results['strategy_annual_return'] - results['benchmark_annual_return']:.2%}")
        print(f"  Information Ratio: {self._fmt_ratio(results.get('information_ratio', 0))}")
        print(f"  Excess Sharpe: {results['strategy_sharpe_ratio'] - results['benchmark_sharpe_ratio']:.2f}")
        print("-"*60)
        print("TRANSACTION COSTS:")
        print(f"  Cost per one-way trade: {results.get('txn_cost_bps', 0):.1f} bps")
        print(f"  Total txn costs: {results.get('total_txn_costs', 0):.4%}")
        print(f"  Avg turnover per rebalance: {results.get('avg_turnover_pct', 0):.1f}%")
        print("-"*60)
        print("RISK MANAGEMENT:")
        print(f"  Stop-loss: {results.get('stop_loss_atr_mult', 0):.1f}x ATR  |  Take-profit: {results.get('take_profit_atr_mult', 0):.1f}x ATR")
        print(f"  Stop-loss hits: {results.get('stop_loss_hits', 0)}  |  Take-profit hits: {results.get('take_profit_hits', 0)}")
        print("="*60)
        
        # Print naive baseline comparison if available
        if 'naive_baseline_return' in results:
            print("\nNAIVE BASELINE (Equal-weight top 5, no screening):")
            print(f"  Cumulative Return: {results['naive_baseline_return']:.2%}")
            print(f"  Annualized Return: {results['naive_baseline_annual_return']:.2%}")
            print(f"  Sharpe Ratio: {results['naive_baseline_sharpe']:.2f}")
            print(f"  Strategy vs Baseline: {results['strategy_cumulative_return'] - results['naive_baseline_return']:.2%}")
            print("="*60)

    def _run_naive_baseline(self, start_date: datetime, end_date: datetime,
                            rebalancing_freq: str = 'M') -> Dict[str, Any]:
        """
        Run a naive equal-weight baseline (top 5 ETFs by score, no screening).
        
        This provides a comparison to measure if the scoring/screening adds value.
        Uses the same universe but selects top 5 by composite score without
        correlation filtering or stop-loss/take-profit.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            rebalancing_freq: Rebalancing frequency
            
        Returns:
            Dictionary with baseline performance metrics
        """
        logger.info("Running naive baseline comparison...")
        
        # Generate rebalancing dates
        if rebalancing_freq == 'D':
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        elif rebalancing_freq == 'W':
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='W')
        else:
            rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq='BME')
        
        if len(rebalancing_dates) < 2:
            return {"error": "Insufficient date range"}
        
        # Get top 5 symbols by rank score across all periods
        all_scores = []
        for symbol in self.etf_universe:
            if validate_etf_symbol(symbol):
                try:
                    result = self.evaluate_symbol(symbol, end_date)
                    if "error" not in result:
                        all_scores.append(result)
                except Exception:
                    pass
        
        if len(all_scores) < 5:
            return {"error": "Not enough valid symbols for baseline"}
        
        self._apply_relative_strength(all_scores)
        all_scores.sort(key=lambda x: x['_rank_score'], reverse=True)
        baseline_symbols = [e['symbol'] for e in all_scores[:5]]
        
        # Calculate baseline returns (simple equal-weight, no stop-loss/take-profit)
        baseline_returns = []
        for i in range(len(rebalancing_dates) - 1):
            current_date = rebalancing_dates[i]
            next_date = rebalancing_dates[i + 1]
            
            period_returns = []
            for symbol in baseline_symbols:
                price_data = self.fetch_period_data(symbol, current_date, next_date)
                if price_data is not None and len(price_data) >= 2:
                    ret = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[0]) - 1
                    period_returns.append(ret)
            
            if period_returns:
                baseline_returns.append(np.mean(period_returns))
        
        if not baseline_returns:
            return {"error": "No baseline returns calculated"}
        
        baseline_cumulative = np.prod([1 + r for r in baseline_returns]) - 1
        periods_per_year = 12  # Monthly rebalancing
        baseline_annual = (1 + baseline_cumulative) ** (periods_per_year / len(baseline_returns)) - 1
        baseline_vol = np.std(baseline_returns) * np.sqrt(periods_per_year)
        baseline_sharpe = baseline_annual / baseline_vol if baseline_vol > 0 else 0
        
        return {
            'naive_baseline_return': baseline_cumulative,
            'naive_baseline_annual_return': baseline_annual,
            'naive_baseline_sharpe': baseline_sharpe,
            'naive_baseline_symbols': baseline_symbols,
        }

    @staticmethod
    def _summarize_regime_performance(
        windows: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate realized period returns by the signal-date market regime."""
        observations: Dict[str, List[Tuple[float, float]]] = {}
        for window in windows:
            for period in window.get('portfolio_history', []):
                regime = str(period.get('market_regime', 'unknown'))
                observations.setdefault(regime, []).append((
                    float(period.get('period_return', 0.0)),
                    float(period.get('benchmark_return', 0.0)),
                ))

        summary: Dict[str, Dict[str, float]] = {}
        for regime, values in observations.items():
            strategy_returns = [value[0] for value in values]
            benchmark_returns = [value[1] for value in values]
            summary[regime] = {
                'periods': float(len(values)),
                'mean_strategy_return': float(np.mean(strategy_returns)),
                'mean_benchmark_return': float(np.mean(benchmark_returns)),
                'mean_excess_return': float(np.mean([
                    strategy - benchmark
                    for strategy, benchmark in values
                ])),
                'win_rate': float(np.mean([ret > 0 for ret in strategy_returns])),
            }
        return summary

    def run_factor_ablation(
        self,
        start_date: datetime,
        end_date: datetime,
        window_months: int = 12,
        step_months: int = 3,
        rebalancing_freq: str = 'M',
        factors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a walk-forward baseline and one neutralized-factor scenario per module.

        Each ablation replaces a module with its neutral value while preserving
        all execution, universe, and risk-management rules. Results should be
        compared out of sample; this report does not optimize weights.
        """
        requested = factors if factors is not None else self.factor_ablation_factors
        invalid = set(requested) - set(self.factor_ablation_factors)
        if invalid:
            return {'error': f"Unsupported ablation factor(s): {', '.join(sorted(invalid))}"}

        previous = set(self._ablated_factors)
        scenarios: Dict[str, Dict[str, Any]] = {}
        try:
            self._ablated_factors = set()
            scenarios['baseline'] = self.run_walk_forward(
                start_date, end_date, window_months, step_months, rebalancing_freq
            )
            for factor in requested:
                self._ablated_factors = {factor}
                scenarios[f'without_{factor}'] = self.run_walk_forward(
                    start_date, end_date, window_months, step_months, rebalancing_freq
                )
        finally:
            self._ablated_factors = previous

        baseline = scenarios['baseline']
        if 'error' in baseline:
            return {'error': f"Baseline ablation run failed: {baseline['error']}"}

        comparison: Dict[str, Dict[str, float]] = {}
        baseline_return = float(baseline['mean_annual_return'])
        baseline_sharpe = float(baseline['mean_sharpe'])
        for name, result in scenarios.items():
            if name == 'baseline' or 'error' in result:
                continue
            comparison[name] = {
                'mean_annual_return': float(result['mean_annual_return']),
                'mean_sharpe': float(result['mean_sharpe']),
                'return_delta_vs_baseline': float(result['mean_annual_return']) - baseline_return,
                'sharpe_delta_vs_baseline': float(result['mean_sharpe']) - baseline_sharpe,
            }

        return {
            'baseline': baseline,
            'scenarios': scenarios,
            'comparison': comparison,
            'factors': requested,
        }

    @contextmanager
    def _tuning_profile(self, profile: Dict[str, Any]) -> Iterator[None]:
        """Temporarily apply a validated candidate profile to the backtester."""
        allowed = {
            'composite_weights', 'top_n', 'min_rank_score',
            'stop_loss_atr_mult', 'take_profit_atr_mult',
            'position_sizing_method',
        }
        unsupported = set(profile) - allowed
        if unsupported:
            raise ValueError(
                f"Unsupported tuning profile setting(s): {', '.join(sorted(unsupported))}"
            )

        previous = {
            'composite_weights': deepcopy(self.composite_weights),
            'top_n': self.top_n,
            'min_rank_score': self.min_rank_score,
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            'position_sizing_method': self.position_sizing_method,
        }
        try:
            if 'composite_weights' in profile:
                weights = profile['composite_weights']
                if not isinstance(weights, dict) or set(weights) != {
                    'technical', 'fundamental', 'sentiment'
                }:
                    raise ValueError("composite_weights must define technical, fundamental, and sentiment")
                if not np.isclose(sum(float(weight) for weight in weights.values()), 1.0):
                    raise ValueError("tuning profile composite_weights must sum to 1.0")
                self.composite_weights = {key: float(value) for key, value in weights.items()}
            if 'top_n' in profile:
                self.top_n = max(1, int(profile['top_n']))
            if 'min_rank_score' in profile:
                self.min_rank_score = float(profile['min_rank_score'])
            if 'stop_loss_atr_mult' in profile:
                self.stop_loss_atr_mult = float(profile['stop_loss_atr_mult'])
            if 'take_profit_atr_mult' in profile:
                self.take_profit_atr_mult = float(profile['take_profit_atr_mult'])
            if 'position_sizing_method' in profile:
                method = str(profile['position_sizing_method'])
                if method not in {'equal', 'score_weighted'}:
                    raise ValueError("position_sizing_method must be 'equal' or 'score_weighted'")
                self.position_sizing_method = method
            yield
        finally:
            self.composite_weights = previous['composite_weights']
            self.top_n = previous['top_n']
            self.min_rank_score = previous['min_rank_score']
            self.stop_loss_atr_mult = previous['stop_loss_atr_mult']
            self.take_profit_atr_mult = previous['take_profit_atr_mult']
            self.position_sizing_method = previous['position_sizing_method']

    def run_out_of_sample_tuning(
        self,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        rebalancing_freq: str = 'M',
        profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Select a frozen profile on training data, then evaluate it once out of sample.

        Candidate profiles are never fitted on the test range. Selection uses
        training Sharpe ratio, then annual return as a deterministic tie-break.
        The returned ``test`` result is the only performance result that should
        be used to judge the chosen profile.
        """
        if train_end >= test_start:
            return {'error': 'Training period must end before the test period begins'}

        candidates = profiles if profiles is not None else self.oos_tuning_profiles
        if not candidates:
            return {'error': 'No out-of-sample tuning profiles configured'}

        training_results: Dict[str, Dict[str, Any]] = {}
        for name, profile in candidates.items():
            try:
                with self._tuning_profile(profile):
                    training_results[name] = self.run_backtest(
                        train_start, train_end, rebalancing_freq
                    )
            except ValueError as error:
                training_results[name] = {'error': str(error)}

        viable = [
            (name, result) for name, result in training_results.items()
            if 'error' not in result
        ]
        if not viable:
            return {'error': 'All training profile runs failed', 'training': training_results}

        selected_name, selected_train = max(
            viable,
            key=lambda item: (
                float(item[1].get('strategy_sharpe_ratio', float('-inf'))),
                float(item[1].get('strategy_annual_return', float('-inf'))),
            ),
        )
        selected_profile = candidates[selected_name]
        with self._tuning_profile(selected_profile):
            test_result = self.run_backtest(test_start, test_end, rebalancing_freq)

        return {
            'selected_profile': selected_name,
            'selected_settings': deepcopy(selected_profile),
            'selection_metric': 'training_sharpe_then_annual_return',
            'training': training_results,
            'selected_training_result': selected_train,
            'test': test_result,
            'train_period': {'start': train_start, 'end': train_end},
            'test_period': {'start': test_start, 'end': test_end},
        }

    def run_walk_forward(
        self,
        start_date: datetime,
        end_date: datetime,
        window_months: int = 12,
        step_months: int = 3,
        rebalancing_freq: str = 'M',
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation to test strategy robustness.

        Splits the full period into overlapping windows. Each window is an
        independent backtest. Results are aggregated to show consistency
        across different market regimes.

        Args:
            start_date: Overall start date
            end_date: Overall end date
            window_months: Length of each walk-forward window
            step_months: How far to advance each window (overlap = window - step)
            rebalancing_freq: Rebalancing frequency within each window

        Returns:
            Dictionary with aggregated walk-forward results
        """
        logger.info(f"Running walk-forward validation: {window_months}m windows, {step_months}m steps")

        # Generate window boundaries
        windows = []
        current_start = start_date
        while current_start + timedelta(days=window_months * 30) <= end_date:
            window_end = current_start + timedelta(days=window_months * 30)
            windows.append((current_start, window_end))
            current_start += timedelta(days=step_months * 30)

        if len(windows) < 2:
            logger.error(f"Not enough windows for walk-forward (got {len(windows)})")
            return {"error": "Insufficient date range for walk-forward"}

        logger.info(f"Generated {len(windows)} walk-forward windows")

        # Run backtest on each window
        window_results = []
        for i, (w_start, w_end) in enumerate(windows):
            logger.info(f"Walk-forward window {i+1}/{len(windows)}: {w_start.date()} → {w_end.date()}")
            result = self.run_backtest(w_start, w_end, rebalancing_freq)
            if "error" not in result:
                result['window_start'] = w_start
                result['window_end'] = w_end
                result['window_index'] = i
                window_results.append(result)

        if not window_results:
            return {"error": "All walk-forward windows failed"}

        # --- Aggregate metrics across windows ---
        annual_returns = [w['strategy_annual_return'] for w in window_results]
        bench_returns = [w['benchmark_annual_return'] for w in window_results]
        sharpes = [w['strategy_sharpe_ratio'] for w in window_results]
        sortinos = [w.get('strategy_sortino_ratio', 0) for w in window_results]
        max_dds = [w['strategy_max_drawdown'] for w in window_results]
        info_ratios = [w.get('information_ratio', 0) for w in window_results]
        win_rates = [w.get('win_rate', 0) for w in window_results]

        # Count windows where strategy beat benchmark
        beat_benchmark = sum(1 for s, b in zip(annual_returns, bench_returns) if s > b)

        # Filter out inf values for mean calculation
        def safe_mean(values: List[float]) -> float:
            finite = [v for v in values if v != float('inf') and v != float('-inf')]
            return float(np.mean(finite)) if finite else 0.0

        def safe_std(values: List[float]) -> float:
            finite = [v for v in values if v != float('inf') and v != float('-inf')]
            return float(np.std(finite)) if len(finite) > 1 else 0.0

        aggregated = {
            'num_windows': len(window_results),
            'window_months': window_months,
            'step_months': step_months,
            'windows': window_results,

            # Strategy metrics (mean ± std across windows)
            'mean_annual_return': safe_mean(annual_returns),
            'std_annual_return': safe_std(annual_returns),
            'min_annual_return': min(annual_returns),
            'max_annual_return': max(annual_returns),

            'mean_sharpe': safe_mean(sharpes),
            'std_sharpe': safe_std(sharpes),
            'min_sharpe': min(sharpes),
            'max_sharpe': max(sharpes),

            'mean_sortino': safe_mean(sortinos),
            'mean_max_drawdown': safe_mean(max_dds),
            'worst_drawdown': min(max_dds),

            'mean_information_ratio': safe_mean(info_ratios),
            'mean_win_rate': safe_mean(win_rates),

            # Benchmark comparison
            'mean_benchmark_return': safe_mean(bench_returns),
            'beat_benchmark_pct': beat_benchmark / len(window_results),
            'mean_excess_return': safe_mean([s - b for s, b in zip(annual_returns, bench_returns)]),

            # Consistency score: % of windows with positive return
            'pct_profitable_windows': sum(1 for r in annual_returns if r > 0) / len(window_results),
        }
        aggregated['regime_performance'] = self._summarize_regime_performance(window_results)

        return aggregated

    def print_walk_forward_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of walk-forward validation results."""
        if "error" in results:
            print(f"Walk-forward failed: {results['error']}")
            return

        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("=" * 70)
        print(f"Windows: {results['num_windows']} × {results['window_months']}mo (step: {results['step_months']}mo)")
        print("-" * 70)
        print("STRATEGY PERFORMANCE (across windows):")
        print(f"  Annual Return:  mean={results['mean_annual_return']:.2%}  "
              f"std={results['std_annual_return']:.2%}  "
              f"range=[{results['min_annual_return']:.2%}, {results['max_annual_return']:.2%}]")
        print(f"  Sharpe Ratio:   mean={results['mean_sharpe']:.2f}  "
              f"std={results['std_sharpe']:.2f}  "
              f"range=[{results['min_sharpe']:.2f}, {results['max_sharpe']:.2f}]")
        print(f"  Sortino Ratio:  mean={self._fmt_ratio(results['mean_sortino'])}")
        print(f"  Max Drawdown:   mean={results['mean_max_drawdown']:.2%}  "
              f"worst={results['worst_drawdown']:.2%}")
        print(f"  Win Rate:       mean={results['mean_win_rate']:.1%}")
        print(f"  Info Ratio:     mean={self._fmt_ratio(results['mean_information_ratio'])}")
        print("-" * 70)
        print("CONSISTENCY:")
        print(f"  Profitable windows: {results['pct_profitable_windows']:.0%}")
        print(f"  Beat benchmark:     {results['beat_benchmark_pct']:.0%} ({int(results['beat_benchmark_pct'] * results['num_windows'])}/{results['num_windows']})")
        print(f"  Mean excess return: {results['mean_excess_return']:.2%}")
        print(f"  Benchmark mean ret: {results['mean_benchmark_return']:.2%}")
        print("-" * 70)
        print("REGIME-SEGMENTED PERIOD RETURNS:")
        regime_performance = results.get('regime_performance', {})
        if regime_performance:
            for regime, metrics in sorted(regime_performance.items()):
                periods = int(metrics['periods'])
                print(
                    f"  {regime:<9} periods={periods:<3} "
                    f"strategy={metrics['mean_strategy_return']:.2%}  "
                    f"benchmark={metrics['mean_benchmark_return']:.2%}  "
                    f"excess={metrics['mean_excess_return']:.2%}  "
                    f"win-rate={metrics['win_rate']:.0%}"
                )
        else:
            print("  No regime observations available.")
        print("-" * 70)
        print("WINDOW DETAILS:")
        for w in results['windows']:
            beat = "✓" if w['strategy_annual_return'] > w['benchmark_annual_return'] else "✗"
            print(f"  {w['window_start'].date()} → {w['window_end'].date()}  "
                  f"strat={w['strategy_annual_return']:.2%}  "
                  f"bench={w['benchmark_annual_return']:.2%}  "
                  f"sharpe={w['strategy_sharpe_ratio']:.2f}  {beat}")
        print("=" * 70)


def run_sample_backtest() -> Dict[str, Any]:
    """Run a sample backtest for demonstration."""
    # Define backtest period (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Define ETF universe (subset for faster backtesting)
    etf_universe = [
        'SPY', 'QQQ', 'VTI', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 
        'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC', 'VOO', 'VO',
        'VT', 'VEA', 'VEU', 'VWO', 'BND', 'BNDX', 'VNQ', 'VNQI',
        'GLD', 'SLV', 'USO', 'UNG', 'DBC', 'GSG'
    ]
    
    # Create backtester
    backtester = ETFBacktester(etf_universe=etf_universe, lookback_months=6)
    
    # Run backtest with monthly rebalancing
    results = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        rebalancing_freq='M'
    )
    
    # Print results
    backtester.print_backtest_summary(results)
    
    # Also run walk-forward validation
    print("\n\n")
    wf_results = backtester.run_walk_forward(
        start_date=start_date,
        end_date=end_date,
        window_months=12,
        step_months=6,
        rebalancing_freq='M',
    )
    backtester.print_walk_forward_summary(wf_results)
    
    return results

if __name__ == "__main__":
    # Run sample backtest when script is executed directly
    run_sample_backtest()