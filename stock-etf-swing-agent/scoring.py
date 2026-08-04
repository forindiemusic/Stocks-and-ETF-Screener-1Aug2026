"""
Shared scoring functions for ETF Swing Trading Agent.
Used by both the live agent (etf_and_stock_agent.py) and the backtester (backtest.py).
"""

import numpy as np
import yfinance as yf
import pandas as pd
import logging
import re
from typing import Dict, Optional, Any

from retry import retry_call

logger = logging.getLogger(__name__)

# Valid ETF symbol pattern
VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9._-]{1,10}$')


def validate_etf_symbol(symbol: str) -> bool:
    """Validate ETF symbol format."""
    if not symbol or not isinstance(symbol, str):
        return False
    return bool(VALID_SYMBOL_PATTERN.match(symbol.upper()))


def calculate_technical_score(
    indicators: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate technical score based on weighted indicators.

    Args:
        indicators: Dictionary of technical indicators from calculate_technical_indicators()
        weights: Dictionary with keys trend_score, momentum_score, mean_reversion_score,
                 volume_score, volatility_score. Defaults to standard weights if None.

    Returns:
        Technical score between 0.0 and 1.0.
    """
    if weights is None:
        weights = {
            'trend_score': 0.30,
            'momentum_score': 0.25,
            'mean_reversion_score': 0.20,
            'volume_score': 0.15,
            'volatility_score': 0.10,
        }

    if not indicators:
        return 0.0

    score_components: Dict[str, float] = {}

    # Trend Score
    trend_signals = []
    if indicators.get('price_vs_sma50', 0) > 0:
        trend_signals.append(min(indicators['price_vs_sma50'] * 5, 1.0))
    if indicators.get('price_vs_sma200', 0) > 0:
        trend_signals.append(min(indicators['price_vs_sma200'] * 3, 1.0))
    if indicators.get('sma50_vs_sma200', 0) > 0:
        trend_signals.append(min(indicators['sma50_vs_sma200'] * 10, 1.0))
    if indicators.get('macd_histogram', 0) > 0:
        trend_signals.append(0.5)
    # ADX: strong trend (>25) is good for swing trading, very strong (>40) is ideal
    adx = indicators.get('adx', 20)
    if adx >= 25:
        trend_signals.append(min((adx - 20) / 30, 1.0))  # 25→0.17, 40→0.67, 50→1.0
    score_components['trend'] = float(np.mean(trend_signals) if trend_signals else 0.0)

    # Momentum Score
    momentum_signals = []
    rsi = indicators.get('rsi', 50)
    if 30 <= rsi <= 70:
        momentum_signals.append(1.0 - abs(rsi - 50) / 50)
    elif rsi > 70:
        momentum_signals.append(max(0, (100 - rsi) / 30))
    else:
        momentum_signals.append(max(0, (rsi - 20) / 10))

    roc_10 = indicators.get('roc_10', 0)
    momentum_signals.append(min(max(roc_10 / 10, -1), 1))
    score_components['momentum'] = float(np.mean([max(0, x) for x in momentum_signals]) if momentum_signals else 0.0)

    # Mean Reversion Score
    mean_rev_signals = []
    bb_pos = indicators.get('bb_position', 0.5)
    mean_rev_signals.append(1.0 - abs(bb_pos - 0.5) * 2)

    price_vs_sma20 = (indicators.get('price', 0) / indicators.get('sma_20', 1) - 1) if indicators.get('sma_20', 0) > 0 else 0
    mean_rev_signals.append(max(0, 1 - abs(price_vs_sma20) * 20))
    score_components['mean_reversion'] = float(np.mean([max(0, x) for x in mean_rev_signals]) if mean_rev_signals else 0.0)

    # Volume Score
    volume_signals = []
    vol_ratio = indicators.get('volume_ratio', 1)
    volume_signals.append(min(vol_ratio / 2, 1.0))
    score_components['volume'] = float(np.mean([max(0, x) for x in volume_signals]) if volume_signals else 0.0)

    # Volatility Score
    vol_signals = []
    vol_20 = indicators.get('volatility_20', 0.2)
    if 0.15 <= vol_20 <= 0.35:
        vol_signals.append(1.0)
    elif vol_20 < 0.15:
        vol_signals.append(vol_20 / 0.15)
    else:
        vol_signals.append(max(0, 0.5 - (vol_20 - 0.35) / 0.35))
    score_components['volatility'] = float(np.mean([max(0, x) for x in vol_signals]) if vol_signals else 0.0)

    # Calculate weighted score
    total_score = (
        score_components['trend'] * weights['trend_score'] +
        score_components['momentum'] * weights['momentum_score'] +
        score_components['mean_reversion'] * weights['mean_reversion_score'] +
        score_components['volume'] * weights['volume_score'] +
        score_components['volatility'] * weights['volatility_score']
    )

    return min(max(total_score, 0), 1)


def calculate_fundamental_score(
    symbol: str,
    weights: Optional[Dict[str, float]] = None,
    price_data: Optional[pd.DataFrame] = None,
    benchmark_data: Optional[pd.DataFrame] = None,
    is_stock: bool = False,
) -> float:
    """
    Calculate fundamental score based on ETF or stock characteristics.

    Args:
        symbol: Ticker symbol
        weights: Dictionary with keys for ETF or stock fundamental factors.
        price_data: Optional DataFrame with historical OHLCV data.
        benchmark_data: Optional DataFrame with benchmark (SPY) OHLCV data.
        is_stock: If True, use stock-relevant fundamentals (liquidity + Sharpe only)
                  instead of ETF-specific fields (expense ratio, tracking error, AUM).

    Returns:
        Fundamental score between 0.0 and 1.0.
    """
    if weights is None:
        if is_stock:
            weights = {
                'liquidity': 0.50,
                'sharpe_1y': 0.50,
            }
        else:
            weights = {
                'expense_ratio': 0.20,
                'liquidity': 0.20,
                'tracking_error': 0.15,
                'aum': 0.15,
                'yield': 0.15,
                'sharpe_1y': 0.15,
            }

    # Validate symbol format
    if not validate_etf_symbol(symbol):
        logger.warning(f"Invalid ETF symbol format: {symbol}")
        return 0.5

    try:
        ticker = yf.Ticker(symbol)
        info = retry_call(lambda: ticker.info)

        if not info:
            return 0.5

        score_components: Dict[str, float] = {}

        if is_stock:
            # Stock-specific: liquidity + risk-adjusted return only.
            # ETF fields (expense ratio, tracking error, AUM, yield) are
            # meaningless for individual stocks.

            # Liquidity (average volume)
            avg_volume = info.get('averageVolume', 0)
            if avg_volume > 0:
                score_components['liquidity'] = min(np.log10(avg_volume) / 6, 1.0)
            else:
                score_components['liquidity'] = 0.0

            # Sharpe ratio (1-year): computed from actual price data
            score_components['sharpe_1y'] = _compute_sharpe_1y(price_data)

            total_score = (
                score_components['liquidity'] * weights['liquidity'] +
                score_components['sharpe_1y'] * weights['sharpe_1y']
            )
            return min(max(total_score, 0), 1)

        # --- ETF-specific scoring below ---
        # Expense ratio (lower is better)
        expense_ratio = info.get('annualReportExpenseRatio', 0.005)
        score_components['expense_ratio'] = max(0, 1 - expense_ratio * 100)

        # Liquidity (average volume)
        avg_volume = info.get('averageVolume', 0)
        if avg_volume > 0:
            volume_score = min(np.log10(avg_volume) / 6, 1.0)
            score_components['liquidity'] = volume_score
        else:
            score_components['liquidity'] = 0.0

        # Tracking error: computed from actual price data if available
        score_components['tracking_error'] = _compute_tracking_error(
            price_data, benchmark_data
        )

        # AUM (Assets Under Management)
        aum = info.get('totalAssets', 0)
        if aum > 0:
            aum_score = min(np.log10(aum) / 9, 1.0)
            score_components['aum'] = aum_score
        else:
            score_components['aum'] = 0.0

        # Dividend yield
        yield_val = info.get('yield', 0)
        if 0.02 <= yield_val <= 0.04:
            score_components['yield'] = 1.0
        elif yield_val < 0.02:
            score_components['yield'] = yield_val / 0.02
        else:
            score_components['yield'] = max(0, 1 - (yield_val - 0.04) / 0.06)

        # Sharpe ratio (1-year): computed from actual price data if available
        score_components['sharpe_1y'] = _compute_sharpe_1y(price_data)

        # Calculate weighted score
        total_score = (
            score_components['expense_ratio'] * weights['expense_ratio'] +
            score_components['liquidity'] * weights['liquidity'] +
            score_components['tracking_error'] * weights['tracking_error'] +
            score_components['aum'] * weights['aum'] +
            score_components['yield'] * weights['yield'] +
            score_components['sharpe_1y'] * weights['sharpe_1y']
        )

        return min(max(total_score, 0), 1)

    except Exception as e:
        logger.error(f"Error calculating fundamental score for {symbol}: {e}")
        return 0.5


def _compute_tracking_error(
    price_data: Optional[pd.DataFrame] = None,
    benchmark_data: Optional[pd.DataFrame] = None,
) -> float:
    """
    Compute a tracking error score from actual price data.

    Tracking error is the standard deviation of daily return differences
    between the ETF and its benchmark (SPY), annualized. Lower is better.

    Score mapping:
        < 2% annual TE  → 1.0  (excellent tracking)
        2-5%            → 0.8  (good)
        5-10%           → 0.5  (moderate)
        10-20%          → 0.3  (high)
        > 20%           → 0.1  (very high / not tracking benchmark)

    Args:
        price_data: DataFrame with ETF OHLCV data (must have 'Close')
        benchmark_data: DataFrame with benchmark OHLCV data (must have 'Close')

    Returns:
        Score between 0.0 and 1.0. Returns 0.8 (neutral) if data unavailable.
    """
    if price_data is None or benchmark_data is None:
        return 0.8  # Neutral fallback

    if len(price_data) < 20 or len(benchmark_data) < 20:
        return 0.8

    try:
        etf_returns = price_data['Close'].pct_change(fill_method=None).dropna()
        bench_returns = benchmark_data['Close'].pct_change(fill_method=None).dropna()

        # Align on common dates
        common_idx = etf_returns.index.intersection(bench_returns.index)
        if len(common_idx) < 20:
            return 0.8

        etf_aligned = etf_returns.loc[common_idx]
        bench_aligned = bench_returns.loc[common_idx]

        # Daily tracking error
        daily_te = np.std(etf_aligned - bench_aligned)
        annual_te = daily_te * np.sqrt(252)

        # Map to score (lower TE = higher score)
        if annual_te < 0.02:
            return 1.0
        elif annual_te < 0.05:
            return 0.8
        elif annual_te < 0.10:
            return 0.5
        elif annual_te < 0.20:
            return 0.3
        else:
            return 0.1
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.debug(f"Tracking error computation error: {e}")
        return 0.8


def _compute_sharpe_1y(price_data: Optional[pd.DataFrame] = None) -> float:
    """
    Compute a 1-year Sharpe ratio score from actual price data.

    Uses daily returns over the available period (up to ~1 year), annualized.
    Assumes risk-free rate = 0%.

    Score mapping:
        Sharpe > 2.0  → 1.0  (excellent)
        1.0-2.0       → 0.8  (good)
        0.5-1.0       → 0.6  (moderate)
        0.0-0.5       → 0.4  (below average)
        < 0.0         → 0.2  (negative returns)

    Args:
        price_data: DataFrame with ETF OHLCV data (must have 'Close')

    Returns:
        Score between 0.0 and 1.0. Returns 0.6 (neutral) if data unavailable.
    """
    if price_data is None or len(price_data) < 20:
        return 0.6  # Neutral fallback

    try:
        daily_returns = price_data['Close'].pct_change(fill_method=None).dropna()
        if len(daily_returns) < 20:
            return 0.6

        ann_return = daily_returns.mean() * 252
        ann_vol = daily_returns.std() * np.sqrt(252)

        if ann_vol == 0:
            return 0.6

        sharpe = ann_return / ann_vol

        if sharpe > 2.0:
            return 1.0
        elif sharpe > 1.0:
            return 0.8
        elif sharpe > 0.5:
            return 0.6
        elif sharpe > 0.0:
            return 0.4
        else:
            return 0.2
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
        logger.debug(f"Sharpe computation error: {e}")
        return 0.6