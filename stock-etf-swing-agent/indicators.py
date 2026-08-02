"""
Shared technical indicator calculations for ETF Swing Trading Agent.
Used by both the live agent (etf_and_stock_agent.py) and the backtester (backtest.py).

Uses TA-Lib for industry-standard indicator calculations (RSI, MACD, ATR, ADX)
with Wilder's smoothing where appropriate. Simple moving averages, Bollinger Bands,
and volume metrics remain in pandas for simplicity.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional


def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate technical indicators for scoring."""
    if data is None or len(data) < 50:
        return {}

    close = data['Close'].values.astype(np.float64)
    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    volume = data['Volume'].values.astype(np.float64)

    indicators: Dict[str, float] = {}

    # --- Moving Averages (pandas — simple and correct) ---
    close_series = data['Close']
    indicators['sma_20'] = close_series.rolling(20).mean().iloc[-1]
    indicators['sma_50'] = close_series.rolling(50).mean().iloc[-1]
    indicators['sma_200'] = close_series.rolling(200).mean().iloc[-1]

    # Current price
    indicators['price'] = close_series.iloc[-1]

    # Trend indicators
    indicators['price_vs_sma50'] = (indicators['price'] / indicators['sma_50'] - 1) if indicators['sma_50'] > 0 else 0
    indicators['price_vs_sma200'] = (indicators['price'] / indicators['sma_200'] - 1) if indicators['sma_200'] > 0 else 0
    indicators['sma50_vs_sma200'] = (indicators['sma_50'] / indicators['sma_200'] - 1) if indicators['sma_200'] > 0 else 0

    # --- MACD (TA-Lib — proper EMA-of-MACD signal line) ---
    macd_line, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)  # type: ignore[arg-type]
    indicators['macd'] = float(macd_line[-1])
    indicators['macd_signal'] = float(macd_signal[-1])
    indicators['macd_histogram'] = float(macd_hist[-1])

    # --- RSI (TA-Lib — uses Wilder's smoothing, not SMA) ---
    rsi = talib.RSI(close, timeperiod=14)  # type: ignore[arg-type]
    indicators['rsi'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0

    # --- ADX (TA-Lib — trend strength, 0-100) ---
    adx = talib.ADX(high, low, close, timeperiod=14)  # type: ignore[arg-type]
    indicators['adx'] = float(adx[-1]) if not np.isnan(adx[-1]) else 20.0

    # --- Bollinger Bands (pandas — simple and correct) ---
    sma_20 = close_series.rolling(20).mean()
    std_20 = close_series.rolling(20).std()
    indicators['bb_upper'] = (sma_20 + 2 * std_20).iloc[-1]
    indicators['bb_lower'] = (sma_20 - 2 * std_20).iloc[-1]
    indicators['bb_position'] = (indicators['price'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']) if indicators['bb_upper'] != indicators['bb_lower'] else 0.5

    # --- Momentum / Rate of Change (pandas) ---
    indicators['roc_10'] = (close_series.iloc[-1] / close_series.iloc[-11] - 1) * 100 if len(close_series) > 10 else 0
    indicators['roc_20'] = (close_series.iloc[-1] / close_series.iloc[-21] - 1) * 100 if len(close_series) > 20 else 0

    # --- Volume indicators (pandas) ---
    volume_series = data['Volume']
    indicators['volume_sma_20'] = volume_series.rolling(20).mean().iloc[-1]
    indicators['volume_ratio'] = volume_series.iloc[-1] / indicators['volume_sma_20'] if indicators['volume_sma_20'] > 0 else 1

    # --- ATR (TA-Lib — uses Wilder's smoothing) ---
    atr = talib.ATR(high, low, close, timeperiod=14)  # type: ignore[arg-type]
    indicators['atr_14'] = float(atr[-1]) if not np.isnan(atr[-1]) else 0.0

    # --- Volatility (pandas — annualized 20-day) ---
    indicators['volatility_20'] = close_series.pct_change(fill_method=None).rolling(20).std().iloc[-1] * np.sqrt(252)

    return indicators


def calculate_short_term_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """
    Short-term (days-scale) indicators for swing entries.

    Designed for a ~1 month (>= 20 rows) daily window. Captures near-term
    momentum, short moving-average structure, short RSI, MACD histogram, and
    volume confirmation — the signals that drive gains over days, not weeks.
    """
    if data is None or len(data) < 20:
        return {}

    close = data['Close'].values.astype(np.float64)
    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    volume = data['Volume'].values.astype(np.float64)
    close_series = data['Close']
    volume_series = data['Volume']

    ind: Dict[str, float] = {}
    ind['price'] = float(close_series.iloc[-1])

    # Short moving averages
    ind['sma_5'] = float(close_series.rolling(5).mean().iloc[-1])
    ind['sma_10'] = float(close_series.rolling(10).mean().iloc[-1])
    ind['sma_20'] = float(close_series.rolling(20).mean().iloc[-1])

    # Near-term momentum (returns over 5 and 10 trading days)
    ind['roc_5'] = (close_series.iloc[-1] / close_series.iloc[-6] - 1) * 100 if len(close_series) > 5 else 0.0
    ind['roc_10'] = (close_series.iloc[-1] / close_series.iloc[-11] - 1) * 100 if len(close_series) > 10 else 0.0

    # Short RSI (5) — sensitive to day-scale overbought/oversold
    rsi5 = talib.RSI(close, timeperiod=5)  # type: ignore[arg-type]
    ind['rsi_5'] = float(rsi5[-1]) if not np.isnan(rsi5[-1]) else 50.0

    # MACD histogram (standard 12/26/9) — near-term trend acceleration
    _, _, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)  # type: ignore[arg-type]
    ind['macd_histogram'] = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0

    # Volume confirmation: average of last 3 days vs 20-day average
    vol_sma20 = float(volume_series.rolling(20).mean().iloc[-1])
    ind['volume_ratio_3d'] = float(volume[-3:].mean() / vol_sma20) if vol_sma20 > 0 else 1.0  # type: ignore[union-attr]

    # Short ATR (5) for day-scale volatility
    atr5 = talib.ATR(high, low, close, timeperiod=5)  # type: ignore[arg-type]
    ind['atr_5'] = float(atr5[-1]) if not np.isnan(atr5[-1]) else 0.0

    return ind


def calculate_short_term_score(
    indicators: Dict[str, float],
    spy_indicators: Optional[Dict[str, float]] = None,
) -> float:
    """
    Score a symbol for short-term (days-scale) swing entries, 0.0 to 1.0.

    Rewards: positive near-term momentum (risk-adjusted and relative to SPY),
    bullish short MA structure (price > SMA5 > SMA10 > SMA20), RSI(5) in a
    healthy 40-75 zone, positive MACD histogram, and volume confirmation.

    If spy_indicators is provided, momentum is scored on *relative* strength
    (stock ROC minus SPY ROC) rather than absolute returns.
    """
    if not indicators:
        return 0.0

    comp: Dict[str, float] = {}
    price = indicators.get('price', 0.0)
    atr5 = indicators.get('atr_5', 0.0)

    # --- Risk-adjusted relative momentum ---
    # Compute raw ROC values, then subtract SPY ROC for relative strength.
    # Divide by ATR(5)/price to reward risk-adjusted moves.
    roc_5 = indicators.get('roc_5', 0.0)
    roc_10 = indicators.get('roc_10', 0.0)

    if spy_indicators:
        roc_5 -= spy_indicators.get('roc_5', 0.0)
        roc_10 -= spy_indicators.get('roc_10', 0.0)

    # Risk adjustment: scale by ATR(5) as % of price.
    # A 3% move with 1% daily ATR is 3x risk-adjusted (strong).
    # A 3% move with 5% daily ATR is 0.6x (noise).
    risk_scale = (atr5 / price) * 100 if price > 0 and atr5 > 0 else 2.0
    adj_roc_5 = roc_5 / max(risk_scale, 0.5)
    adj_roc_10 = roc_10 / max(risk_scale * 1.5, 0.75)

    mom = np.mean([
        min(max(adj_roc_5, -1.0), 1.0),
        min(max(adj_roc_10, -1.0), 1.0),
    ])
    comp['momentum'] = float(max(0.0, float(mom)))

    # Short MA structure: bullish alignment adds up to 1.0
    sma5 = indicators.get('sma_5', 0.0)
    sma10 = indicators.get('sma_10', 0.0)
    sma20 = indicators.get('sma_20', 0.0)
    structure = 0.0
    if price > sma5:
        structure += 0.34
    if sma5 > sma10:
        structure += 0.33
    if sma10 > sma20:
        structure += 0.33
    comp['structure'] = structure

    # RSI(5): prefer 40-75 (momentum without being extremely overbought)
    rsi5 = indicators.get('rsi_5', 50.0)
    if 40 <= rsi5 <= 75:
        comp['rsi'] = 1.0 - abs(rsi5 - 55.0) / 35.0
    elif rsi5 > 75:
        comp['rsi'] = max(0.0, (100.0 - rsi5) / 25.0)  # overbought penalty
    else:
        comp['rsi'] = max(0.0, (rsi5 - 20.0) / 20.0)    # weak / oversold

    # MACD histogram: positive = near-term upward acceleration
    comp['macd'] = 1.0 if indicators.get('macd_histogram', 0.0) > 0 else 0.0

    # Volume confirmation: last 3 days vs 20-day average
    vr = indicators.get('volume_ratio_3d', 1.0)
    comp['volume'] = min(vr / 1.5, 1.0)

    weights = {
        'momentum': 0.35,
        'structure': 0.25,
        'rsi': 0.15,
        'macd': 0.15,
        'volume': 0.10,
    }
    score = sum(comp[k] * w for k, w in weights.items())
    return min(max(score, 0.0), 1.0)


def calculate_4week_growth_outlook(
    short_term_indicators: Dict[str, float],
    spy_indicators: Optional[Dict[str, float]] = None,
    sentiment_score: float = 0.5,
    price_change_1w: float = 0.0,
    market_regime: str = "bull",
) -> Dict[str, float]:
    """
    Estimate potential 4-week price appreciation from non-dividend factors.

    Combines short-term momentum, sentiment (demand proxy), recent price
    trend, and market regime into a 0.0–1.0 growth outlook score. Returns
    both the overall score and a breakdown of contributing components.

    Args:
        short_term_indicators: From ``calculate_short_term_indicators()``.
        spy_indicators: SPY short-term indicators for relative strength.
        sentiment_score: News/polarity score (0.0–1.0) from the agent.
        price_change_1w: 1-week % price change.
        market_regime: 'bull', 'bear', or 'sideways'.

    Returns:
        Dict with keys: ``growth_score``, ``momentum``, ``sentiment``,
        ``price_trend``, ``regime_adj``, ``volume``.
    """
    if not short_term_indicators:
        return {
            'growth_score': 0.0,
            'momentum': 0.0,
            'sentiment': 0.0,
            'price_trend': 0.0,
            'regime_adj': 0.0,
            'volume': 0.0,
        }

    price = short_term_indicators.get('price', 0.0)
    atr5 = short_term_indicators.get('atr_5', 0.0)

    # --- 1. Risk-adjusted relative momentum (35%) ---
    roc_5 = short_term_indicators.get('roc_5', 0.0)
    roc_10 = short_term_indicators.get('roc_10', 0.0)
    if spy_indicators:
        roc_5 -= spy_indicators.get('roc_5', 0.0)
        roc_10 -= spy_indicators.get('roc_10', 0.0)

    risk_scale = (atr5 / price) * 100 if price > 0 and atr5 > 0 else 2.0
    adj_roc_5 = roc_5 / max(risk_scale, 0.5)
    adj_roc_10 = roc_10 / max(risk_scale * 1.5, 0.75)
    mean_momentum = np.mean([
        min(max(adj_roc_5, -1.0), 1.0),
        min(max(adj_roc_10, -1.0), 1.0),
    ])
    momentum = float(max(0.0, float(mean_momentum)))

    # --- 2. Sentiment / demand proxy (25%) ---
    # News sentiment reflects demand; scale 0.5→0.0, 1.0→1.0
    sentiment = max(0.0, (sentiment_score - 0.3) / 0.7) if sentiment_score else 0.0

    # --- 3. Price trend — 1-week change (15%) ---
    # +2% or more = 1.0, flat = 0.5, -2% or worse = 0.0
    price_trend = min(max((price_change_1w + 2.0) / 4.0, 0.0), 1.0)

    # --- 4. Market regime adjustment (15%) ---
    regime_map = {'bull': 1.0, 'sideways': 0.5, 'bear': 0.15}
    regime_adj = regime_map.get(market_regime, 0.5)

    # --- 5. Volume confirmation (10%) ---
    vr = short_term_indicators.get('volume_ratio_3d', 1.0)
    volume = min(vr / 1.5, 1.0)

    weights = {
        'momentum': 0.35,
        'sentiment': 0.25,
        'price_trend': 0.15,
        'regime_adj': 0.15,
        'volume': 0.10,
    }
    components = {
        'momentum': momentum,
        'sentiment': sentiment,
        'price_trend': price_trend,
        'regime_adj': regime_adj,
        'volume': volume,
    }
    growth_score = sum(components[k] * weights[k] for k in weights)
    growth_score = min(max(growth_score, 0.0), 1.0)

    return {
        'growth_score': growth_score,
        **components,
    }