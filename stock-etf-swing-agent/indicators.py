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


def calculate_day_trade_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """
    Ultra-short (1-3 day) indicators for day-scale entries.

    Designed for a ~2 week (>= 10 rows) daily window. Captures 1-3 day
    momentum, overnight gaps, Bollinger squeeze, volume spikes, and
    proximity to recent highs/lows — the signals that drive gains over
    1-3 days, not weeks.

    Key additions vs calculate_short_term_indicators:
    - ROC(1), ROC(2), ROC(3) for ultra-near momentum
    - RSI(2) for 1-2 day overbought/oversold
    - Overnight gap detection
    - Bollinger Band squeeze (low volatility setup)
    - Proximity to 5-day high/low
    - Volume spike (today vs 5-day average)
    """
    if data is None or len(data) < 10:
        return {}

    close = data['Close'].values.astype(np.float64)
    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    volume = data['Volume'].values.astype(np.float64)
    close_series = data['Close']
    high_series = data['High']
    low_series = data['Low']
    open_series = data['Open']
    volume_series = data['Volume']

    ind: Dict[str, float] = {}
    ind['price'] = float(close_series.iloc[-1])

    # --- Ultra-short moving averages ---
    ind['sma_3'] = float(close_series.rolling(3).mean().iloc[-1])
    ind['sma_5'] = float(close_series.rolling(5).mean().iloc[-1])

    # --- 1/2/3-day momentum (returns) ---
    ind['roc_1'] = (close_series.iloc[-1] / close_series.iloc[-2] - 1) * 100 if len(close_series) > 1 else 0.0
    ind['roc_2'] = (close_series.iloc[-1] / close_series.iloc[-3] - 1) * 100 if len(close_series) > 2 else 0.0
    ind['roc_3'] = (close_series.iloc[-1] / close_series.iloc[-4] - 1) * 100 if len(close_series) > 3 else 0.0

    # --- Momentum acceleration: ROC(1) vs ROC(3) ---
    # Positive = accelerating (stronger today than 3 days ago)
    ind['momentum_accel'] = ind['roc_1'] - ind['roc_3']

    # --- Overnight gap ---
    # Gap % = (Open_today - Close_yesterday) / Close_yesterday
    ind['gap_pct'] = float((open_series.iloc[-1] / close_series.iloc[-2] - 1) * 100) if len(close_series) > 1 else 0.0
    ind['gap_filled'] = 1.0 if (ind['gap_pct'] > 0 and close_series.iloc[-1] > open_series.iloc[-1]) or \
                        (ind['gap_pct'] < 0 and close_series.iloc[-1] < open_series.iloc[-1]) else 0.0

    # --- RSI(2) — ultra-sensitive to 1-2 day extremes ---
    rsi2 = talib.RSI(close, timeperiod=2)  # type: ignore[arg-type]
    ind['rsi_2'] = float(rsi2[-1]) if not np.isnan(rsi2[-1]) else 50.0

    # --- Bollinger Band squeeze (volatility contraction) ---
    # Squeeze = BB width / SMA20 — low values precede breakouts
    sma_20 = close_series.rolling(20).mean()
    std_20 = close_series.rolling(20).std()
    bb_width = ((sma_20 + 2 * std_20) - (sma_20 - 2 * std_20)).iloc[-1]
    bb_sma = (sma_20 * 4 * std_20 / sma_20).iloc[-1] if sma_20.iloc[-1] > 0 else 1.0
    ind['bb_width'] = float(bb_width)
    ind['bb_squeeze'] = float(bb_width / sma_20.iloc[-1]) if sma_20.iloc[-1] > 0 else 1.0

    # --- Proximity to 5-day high/low ---
    high_5 = float(high_series.rolling(5).max().iloc[-1])
    low_5 = float(low_series.rolling(5).min().iloc[-1])
    ind['high_5'] = high_5
    ind['low_5'] = low_5
    range_5 = high_5 - low_5
    if range_5 > 0:
        ind['proximity_high_5'] = float((ind['price'] - low_5) / range_5)  # 0=at low, 1=at high
    else:
        ind['proximity_high_5'] = 0.5

    # --- Volume spike (today vs 5-day average) ---
    vol_sma5 = float(volume_series.rolling(5).mean().iloc[-1])
    ind['volume_ratio_1d'] = float(volume[-1] / vol_sma5) if vol_sma5 > 0 else 1.0  # type: ignore[union-attr]

    # --- Intraday volatility (today's range / close) ---
    ind['intraday_range_pct'] = float((high_series.iloc[-1] - low_series.iloc[-1]) / close_series.iloc[-1] * 100)

    # --- ATR(2) for 1-2 day volatility ---
    atr2 = talib.ATR(high, low, close, timeperiod=2)  # type: ignore[arg-type]
    ind['atr_2'] = float(atr2[-1]) if not np.isnan(atr2[-1]) else 0.0

    return ind


def calculate_day_trade_score(
    indicators: Dict[str, float],
    spy_indicators: Optional[Dict[str, float]] = None,
) -> float:
    """
    Score a symbol for day-scale (1-5 day) entries, 0.0 to 1.0.

    Designed for ultra-short holding periods. Rewards:
    - Strong 1-3 day momentum (risk-adjusted, relative to SPY)
    - Momentum acceleration (getting stronger, not fading)
    - Bullish gap behavior (gaps up and holds / fills)
    - RSI(2) in a healthy 30-70 zone (not exhausted)
    - Bollinger squeeze setup (volatility contraction before expansion)
    - Near 5-day high (breaking out, not fading)
    - Volume spike (institutional interest)
    - Low intraday volatility relative to ATR (efficient price discovery)

    If spy_indicators is provided, momentum is scored on *relative* strength.
    """
    if not indicators:
        return 0.0

    comp: Dict[str, float] = {}
    price = indicators.get('price', 0.0)
    atr2 = indicators.get('atr_2', 0.0)

    # --- 1. Risk-adjusted ultra-short momentum (30%) ---
    roc_1 = indicators.get('roc_1', 0.0)
    roc_2 = indicators.get('roc_2', 0.0)
    roc_3 = indicators.get('roc_3', 0.0)

    if spy_indicators:
        roc_1 -= spy_indicators.get('roc_1', 0.0)
        roc_2 -= spy_indicators.get('roc_2', 0.0)
        roc_3 -= spy_indicators.get('roc_3', 0.0)

    # Risk adjustment: scale by ATR(2) as % of price
    risk_scale = (atr2 / price) * 100 if price > 0 and atr2 > 0 else 1.5
    adj_roc_1 = roc_1 / max(risk_scale * 0.5, 0.3)
    adj_roc_2 = roc_2 / max(risk_scale * 0.75, 0.5)
    adj_roc_3 = roc_3 / max(risk_scale, 0.5)

    mom = np.mean([
        min(max(adj_roc_1, -1.0), 1.0),
        min(max(adj_roc_2, -1.0), 1.0),
        min(max(adj_roc_3, -1.0), 1.0),
    ])
    comp['momentum'] = float(max(0.0, float(mom)))

    # --- 2. Momentum acceleration (15%) ---
    # Positive accel = recent strength > past strength = bullish
    accel = indicators.get('momentum_accel', 0.0)
    comp['acceleration'] = float(min(max(accel / 2.0, 0.0), 1.0))

    # --- 3. Gap analysis (15%) ---
    gap = indicators.get('gap_pct', 0.0)
    gap_filled = indicators.get('gap_filled', 0.0)
    # Positive gap that holds = strong; negative gap that fills = reversal
    if gap > 0.3 and gap_filled > 0.5:
        comp['gap'] = 1.0  # Bullish gap up that held
    elif gap > 0.3:
        comp['gap'] = 0.7  # Gap up but faded
    elif gap < -0.3 and gap_filled > 0.5:
        comp['gap'] = 0.8  # Gap down that filled (reversal)
    elif gap < -0.3:
        comp['gap'] = 0.2  # Gap down that didn't fill
    else:
        comp['gap'] = 0.5  # No significant gap

    # --- 4. RSI(2) — prefer 30-70 (not exhausted, not dead) ---
    rsi2 = indicators.get('rsi_2', 50.0)
    if 30 <= rsi2 <= 70:
        comp['rsi'] = 1.0 - abs(rsi2 - 50.0) / 40.0
    elif rsi2 > 80:
        comp['rsi'] = max(0.0, (100.0 - rsi2) / 20.0)  # severely overbought
    elif rsi2 < 20:
        comp['rsi'] = max(0.0, (rsi2 - 5.0) / 15.0)    # severely oversold
    else:
        comp['rsi'] = 0.3  # moderately overbought/oversold

    # --- 5. Bollinger squeeze (10%) ---
    # Low BB squeeze = volatility contraction = potential breakout
    squeeze = indicators.get('bb_squeeze', 1.0)
    # Normalize: lower is better for setup
    comp['squeeze'] = float(max(0.0, min(1.0, (0.15 - squeeze) / 0.10 + 0.5)))

    # --- 6. Proximity to 5-day high (10%) ---
    prox = indicators.get('proximity_high_5', 0.5)
    # Near high = breaking out (0.7-1.0), near low = fading (0.0-0.3)
    if prox >= 0.7:
        comp['proximity'] = min(1.0, (prox - 0.5) * 2.0)  # 0.7→0.4, 1.0→1.0
    elif prox <= 0.3:
        comp['proximity'] = max(0.0, 1.0 - (0.3 - prox) * 3.0)  # 0.3→1.0, 0.0→0.1
    else:
        comp['proximity'] = 0.5  # Middle of range

    # --- 7. Volume spike (5%) ---
    vr = indicators.get('volume_ratio_1d', 1.0)
    comp['volume'] = min(vr / 2.0, 1.0)

    weights = {
        'momentum': 0.30,
        'acceleration': 0.15,
        'gap': 0.15,
        'rsi': 0.15,
        'squeeze': 0.10,
        'proximity': 0.10,
        'volume': 0.05,
    }
    score = sum(comp[k] * w for k, w in weights.items())
    return min(max(score, 0.0), 1.0)


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