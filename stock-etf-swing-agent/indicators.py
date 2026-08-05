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
    # MACD crossover: +1 if MACD crossed above signal within last 3 days,
    # -1 if crossed below, 0 otherwise. Classic entry/exit trigger.
    if len(macd_line) > 3 and not np.isnan(macd_line[-1]) and not np.isnan(macd_signal[-1]):
        crossed_up = False
        crossed_down = False
        for i in range(1, 4):  # check last 3 days
            prev_line = macd_line[-i-1]
            prev_sig = macd_signal[-i-1]
            if np.isnan(prev_line) or np.isnan(prev_sig):
                continue
            if prev_line < prev_sig and macd_line[-i] > macd_signal[-i]:
                crossed_up = True
            elif prev_line > prev_sig and macd_line[-i] < macd_signal[-i]:
                crossed_down = True
        if crossed_up:
            indicators['macd_crossover'] = 1.0
        elif crossed_down:
            indicators['macd_crossover'] = -1.0
        else:
            indicators['macd_crossover'] = 0.0
    else:
        indicators['macd_crossover'] = 0.0

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

    # --- VWAP (Volume-Weighted Average Price) ---
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * volume_series).cumsum() / volume_series.cumsum()
    indicators['vwap'] = float(vwap.iloc[-1])
    indicators['vwap_distance_pct'] = float((indicators['price'] / indicators['vwap'] - 1) * 100) if indicators['vwap'] > 0 else 0.0

    # --- Chaikin Money Flow (CMF) over 20 days ---
    high_low = data['High'] - data['Low']
    mfv = pd.Series(np.zeros(len(data)), index=data.index)
    valid = high_low > 0
    mfv[valid] = ((data['Close'][valid] - data['Low'][valid]) - (data['High'][valid] - data['Close'][valid])) / high_low[valid] * data['Volume'][valid]
    cmf_20 = mfv.rolling(20).sum() / volume_series.rolling(20).sum()
    indicators['cmf_20'] = float(cmf_20.iloc[-1]) if not np.isnan(cmf_20.iloc[-1]) else 0.0

    # --- Relative Volume (RVOL) — today's volume vs same-weekday avg over 5 weeks ---
    if len(volume_series) >= 25 and isinstance(volume_series.index, pd.DatetimeIndex):
        weekday = volume_series.index[-1].weekday()
        same_weekday_mask = volume_series.index.weekday == weekday
        same_weekday_vol = volume_series[same_weekday_mask]
        if len(same_weekday_vol) > 1:
            if len(same_weekday_vol) > 5:
                rvol_5 = volume_series.iloc[-1] / same_weekday_vol.iloc[:-1].iloc[-5:].mean()
            else:
                rvol_5 = volume_series.iloc[-1] / same_weekday_vol.iloc[:-1].mean()
            indicators['rvol_5'] = float(rvol_5) if rvol_5 == rvol_5 else 1.0
        else:
            indicators['rvol_5'] = 1.0
    else:
        indicators['rvol_5'] = 1.0

    # --- ATR (TA-Lib — uses Wilder's smoothing) ---
    atr = talib.ATR(high, low, close, timeperiod=14)  # type: ignore[arg-type]
    indicators['atr_14'] = float(atr[-1]) if not np.isnan(atr[-1]) else 0.0

    # --- ATR Trend Direction (volatility expansion/contraction) ---
    # ATR(10) vs ATR(30) ratio > 1.1 = volatility expanding (breakout potential)
    # Ratio < 0.9 = volatility contracting (consolidation)
    atr_10 = talib.ATR(high, low, close, timeperiod=10)  # type: ignore[arg-type]
    atr_30 = talib.ATR(high, low, close, timeperiod=30)  # type: ignore[arg-type]
    atr10_val = float(atr_10[-1]) if not np.isnan(atr_10[-1]) else 0.0
    atr30_val = float(atr_30[-1]) if not np.isnan(atr_30[-1]) else 0.0
    indicators['atr_trend_ratio'] = atr10_val / atr30_val if atr30_val > 0 else 1.0

    # --- OBV (On-Balance Volume) Trend ---
    # OBV slope over 10 days: positive = accumulation, negative = distribution
    obv = (np.sign(close_series.diff()) * volume_series).fillna(0).cumsum()
    obv_10d_ago = obv.iloc[-11] if len(obv) > 10 else obv.iloc[0]
    obv_change = obv.iloc[-1] - obv_10d_ago
    obv_price = close_series.iloc[-1]
    # Normalize OBV change as % of price to make comparable across symbols
    indicators['obv_trend'] = float(obv_change / obv_price) if obv_price > 0 else 0.0

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

    # Momentum quality: ROC(5) > ROC(10) means momentum is accelerating (not fading)
    # Positive = recent momentum stronger than older momentum = quality signal
    ind['momentum_quality'] = ind['roc_5'] - ind['roc_10']

    # Short RSI (5) — sensitive to day-scale overbought/oversold
    rsi5 = talib.RSI(close, timeperiod=5)  # type: ignore[arg-type]
    ind['rsi_5'] = float(rsi5[-1]) if not np.isnan(rsi5[-1]) else 50.0

    # MACD histogram (standard 12/26/9) — near-term trend acceleration
    macd_line, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)  # type: ignore[arg-type]
    ind['macd_histogram'] = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0
    # MACD crossover: +1 if crossed above signal within last 3 days, -1 if below
    if len(macd_line) > 3 and not np.isnan(macd_line[-1]) and not np.isnan(macd_signal[-1]):
        crossed_up = False
        crossed_down = False
        for i in range(1, 4):
            prev_line = macd_line[-i-1]
            prev_sig = macd_signal[-i-1]
            if np.isnan(prev_line) or np.isnan(prev_sig):
                continue
            if prev_line < prev_sig and macd_line[-i] > macd_signal[-i]:
                crossed_up = True
            elif prev_line > prev_sig and macd_line[-i] < macd_signal[-i]:
                crossed_down = True
        if crossed_up:
            ind['macd_crossover'] = 1.0
        elif crossed_down:
            ind['macd_crossover'] = -1.0
        else:
            ind['macd_crossover'] = 0.0
    else:
        ind['macd_crossover'] = 0.0

    # Volume confirmation: average of last 3 days vs 20-day average
    vol_sma20 = float(volume_series.rolling(20).mean().iloc[-1])
    ind['volume_ratio_3d'] = float(volume[-3:].mean() / vol_sma20) if vol_sma20 > 0 else 1.0  # type: ignore[union-attr]

    # --- VWAP (Volume-Weighted Average Price) ---
    # Standard intraday reference. Using daily data: VWAP = cum(price*vol)/cum(vol)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * volume_series).cumsum() / volume_series.cumsum()
    ind['vwap'] = float(vwap.iloc[-1])
    ind['vwap_distance_pct'] = float((ind['price'] / ind['vwap'] - 1) * 100) if ind['vwap'] > 0 else 0.0

    # --- Chaikin Money Flow (CMF) over 20 days ---
    # MFV = ((close - low) - (high - close)) / (high - low) * volume
    # CMF = sum(MFV, 20) / sum(volume, 20)
    high_low = data['High'] - data['Low']
    mfv = pd.Series(np.zeros(len(data)), index=data.index)
    valid = high_low > 0
    mfv[valid] = ((data['Close'][valid] - data['Low'][valid]) - (data['High'][valid] - data['Close'][valid])) / high_low[valid] * data['Volume'][valid]
    cmf_20 = mfv.rolling(20).sum() / volume_series.rolling(20).sum()
    ind['cmf_20'] = float(cmf_20.iloc[-1]) if not np.isnan(cmf_20.iloc[-1]) else 0.0

    # --- Relative Volume (RVOL) — today's volume vs same-weekday avg over 5 weeks ---
    # More nuanced than raw volume ratio: accounts for day-of-week patterns
    if len(volume_series) >= 25 and isinstance(volume_series.index, pd.DatetimeIndex):
        weekday = volume_series.index[-1].weekday()
        same_weekday_mask = volume_series.index.weekday == weekday
        same_weekday_vol = volume_series[same_weekday_mask]
        if len(same_weekday_vol) > 1:
            if len(same_weekday_vol) > 5:
                rvol_5 = volume_series.iloc[-1] / same_weekday_vol.iloc[:-1].iloc[-5:].mean()
            else:
                rvol_5 = volume_series.iloc[-1] / same_weekday_vol.iloc[:-1].mean()
            ind['rvol_5'] = float(rvol_5) if rvol_5 == rvol_5 else 1.0
        else:
            ind['rvol_5'] = 1.0
    else:
        ind['rvol_5'] = 1.0

    # Short ATR (5) for day-scale volatility
    atr5 = talib.ATR(high, low, close, timeperiod=5)  # type: ignore[arg-type]
    ind['atr_5'] = float(atr5[-1]) if not np.isnan(atr5[-1]) else 0.0

    # ATR Trend Direction (volatility expansion/contraction)
    atr_10 = talib.ATR(high, low, close, timeperiod=10)  # type: ignore[arg-type]
    atr_30 = talib.ATR(high, low, close, timeperiod=30)  # type: ignore[arg-type]
    atr10_val = float(atr_10[-1]) if not np.isnan(atr_10[-1]) else 0.0
    atr30_val = float(atr_30[-1]) if not np.isnan(atr_30[-1]) else 0.0
    ind['atr_trend_ratio'] = atr10_val / atr30_val if atr30_val > 0 else 1.0

    # OBV (On-Balance Volume) Trend
    obv = (np.sign(close_series.diff()) * volume_series).fillna(0).cumsum()
    obv_10d_ago = obv.iloc[-11] if len(obv) > 10 else obv.iloc[0]
    obv_change = obv.iloc[-1] - obv_10d_ago
    obv_price = close_series.iloc[-1]
    ind['obv_trend'] = float(obv_change / obv_price) if obv_price > 0 else 0.0

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

    # Use last valid close price (skip trailing NaN)
    close_series = data['Close'].dropna()
    if close_series.empty:
        return {}
    
    # Get the last valid index
    last_valid_idx = close_series.index[-1]
    
    # Get series up to last valid index
    close = data.loc[:last_valid_idx, 'Close'].values.astype(np.float64)
    high = data.loc[:last_valid_idx, 'High'].values.astype(np.float64)
    low = data.loc[:last_valid_idx, 'Low'].values.astype(np.float64)
    volume = data.loc[:last_valid_idx, 'Volume'].values.astype(np.float64)
    close_series = data.loc[:last_valid_idx, 'Close']
    high_series = data.loc[:last_valid_idx, 'High']
    low_series = data.loc[:last_valid_idx, 'Low']
    open_series = data.loc[:last_valid_idx, 'Open']
    volume_series = data.loc[:last_valid_idx, 'Volume']

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

    # --- VWAP (Volume-Weighted Average Price) ---
    typical_price = (high_series + low_series + close_series) / 3
    vwap = (typical_price * volume_series).cumsum() / volume_series.cumsum()
    ind['vwap'] = float(vwap.iloc[-1])
    ind['vwap_distance_pct'] = float((ind['price'] / ind['vwap'] - 1) * 100) if ind['vwap'] > 0 else 0.0

    # --- Chaikin Money Flow (CMF) over 20 days ---
    high_low = high_series - low_series
    mfv = pd.Series(np.zeros(len(close_series)), index=close_series.index)
    valid = high_low > 0
    mfv[valid] = ((close_series[valid] - low_series[valid]) - (high_series[valid] - close_series[valid])) / high_low[valid] * volume_series[valid]
    cmf_20 = mfv.rolling(20).sum() / volume_series.rolling(20).sum()
    ind['cmf_20'] = float(cmf_20.iloc[-1]) if not np.isnan(cmf_20.iloc[-1]) else 0.0

    # --- Relative Volume (RVOL) — today's volume vs same-weekday avg over 5 weeks ---
    if len(volume_series) >= 25 and isinstance(volume_series.index, pd.DatetimeIndex):
        weekday = volume_series.index[-1].weekday()
        same_weekday_mask = volume_series.index.weekday == weekday
        same_weekday_vol = volume_series[same_weekday_mask]
        if len(same_weekday_vol) > 1:
            if len(same_weekday_vol) > 5:
                rvol_5 = volume_series.iloc[-1] / same_weekday_vol.iloc[:-1].iloc[-5:].mean()
            else:
                rvol_5 = volume_series.iloc[-1] / same_weekday_vol.iloc[:-1].mean()
            ind['rvol_5'] = float(rvol_5) if rvol_5 == rvol_5 else 1.0
        else:
            ind['rvol_5'] = 1.0
    else:
        ind['rvol_5'] = 1.0

    # --- ATR(2) for 1-2 day volatility ---
    atr2 = talib.ATR(high, low, close, timeperiod=2)  # type: ignore[arg-type]
    ind['atr_2'] = float(atr2[-1]) if not np.isnan(atr2[-1]) else 0.0

    # --- OBV (On-Balance Volume) Trend ---
    obv = (np.sign(close_series.diff()) * volume_series).fillna(0).cumsum()
    obv_5d_ago = obv.iloc[-6] if len(obv) > 5 else obv.iloc[0]
    obv_change = obv.iloc[-1] - obv_5d_ago
    obv_price = close_series.iloc[-1]
    ind['obv_trend'] = float(obv_change / obv_price) if obv_price > 0 else 0.0

    # --- ATR Trend Direction (volatility expansion/contraction) ---
    atr_10 = talib.ATR(high, low, close, timeperiod=10)  # type: ignore[arg-type]
    atr_30 = talib.ATR(high, low, close, timeperiod=30)  # type: ignore[arg-type]
    atr10_val = float(atr_10[-1]) if not np.isnan(atr_10[-1]) else 0.0
    atr30_val = float(atr_30[-1]) if not np.isnan(atr_30[-1]) else 0.0
    ind['atr_trend_ratio'] = atr10_val / atr30_val if atr30_val > 0 else 1.0

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

    # --- 8. OBV trend (5%) ---
    obv = indicators.get('obv_trend', 0.0)
    comp['obv'] = float(min(max(obv / 2.0, 0.0), 1.0))

    # --- 9. ATR trend direction (5%) ---
    atr_ratio = indicators.get('atr_trend_ratio', 1.0)
    if atr_ratio > 1.1:
        comp['atr_trend'] = min(1.0, (atr_ratio - 1.0) / 0.5)
    elif atr_ratio < 0.9:
        comp['atr_trend'] = 0.3
    else:
        comp['atr_trend'] = 0.6

    weights = {
        'momentum': 0.25,
        'acceleration': 0.12,
        'gap': 0.12,
        'rsi': 0.12,
        'squeeze': 0.10,
        'proximity': 0.10,
        'volume': 0.05,
        'obv': 0.07,
        'atr_trend': 0.07,
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
    # MACD crossover within last 3 days: strong entry signal
    mc = indicators.get('macd_crossover', 0.0)
    if mc > 0:
        comp['macd_cross'] = 1.0
    elif mc < 0:
        comp['macd_cross'] = 0.0
    else:
        comp['macd_cross'] = 0.5

    # VWAP proximity: price above VWAP = bullish short-term bias
    vwap_dist = indicators.get('vwap_distance_pct', 0.0)
    if vwap_dist > 0:
        comp['vwap'] = min(1.0, vwap_dist / 2.0)  # +2% above VWAP = 1.0
    else:
        comp['vwap'] = max(0.3, 1.0 - abs(vwap_dist) / 2.0)  # below VWAP penalized, floor 0.3

    # Chaikin Money Flow: positive = accumulation
    cmf = indicators.get('cmf_20', 0.0)
    comp['cmf'] = float(min(max(cmf / 0.3, 0.0), 1.0))  # 0.3 = strong accumulation

    # Volume confirmation: last 3 days vs 20-day average
    vr = indicators.get('volume_ratio_3d', 1.0)
    comp['volume'] = min(vr / 1.5, 1.0)

    # Relative Volume: accounting for same-weekday baseline
    rvol = indicators.get('rvol_5', 1.0)
    comp['rvol'] = min(rvol / 2.0, 1.0)

    # Momentum quality: ROC(5) > ROC(10) means accelerating momentum
    mq = indicators.get('momentum_quality', 0.0)
    # Normalize: +5% difference = 1.0, -5% = 0.0
    comp['momentum_quality'] = float(min(max(mq / 5.0, 0.0), 1.0))

    # OBV trend: positive OBV slope = accumulation
    obv = indicators.get('obv_trend', 0.0)
    # Normalize: OBV change > 2x price = strong accumulation
    comp['obv'] = float(min(max(obv / 2.0, 0.0), 1.0))

    # ATR trend direction: ratio > 1.1 = volatility expanding (breakout)
    atr_ratio = indicators.get('atr_trend_ratio', 1.0)
    if atr_ratio > 1.1:
        comp['atr_trend'] = min(1.0, (atr_ratio - 1.0) / 0.5)  # 1.1→0.2, 1.5→1.0
    elif atr_ratio < 0.9:
        comp['atr_trend'] = 0.3  # contracting volatility = consolidation
    else:
        comp['atr_trend'] = 0.6  # neutral

    weights = {
        'momentum': 0.20,
        'structure': 0.16,
        'rsi': 0.10,
        'macd': 0.08,
        'macd_cross': 0.06,
        'vwap': 0.06,
        'cmf': 0.06,
        'volume': 0.06,
        'rvol': 0.04,
        'momentum_quality': 0.08,
        'obv': 0.06,
        'atr_trend': 0.04,
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
        ``price_trend``, ``regime_adj``, ``volume``, ``obv_trend``,
        ``atr_trend``, ``cmf``, ``rvol``.
    """
    if not short_term_indicators:
        return {
            'growth_score': 0.0,
            'momentum': 0.0,
            'sentiment': 0.0,
            'price_trend': 0.0,
            'regime_adj': 0.0,
            'volume': 0.0,
            'obv_trend': 0.0,
            'atr_trend': 0.0,
            'cmf': 0.0,
            'rvol': 0.0,
        }

    price = short_term_indicators.get('price', 0.0)
    atr5 = short_term_indicators.get('atr_5', 0.0)

    # --- 1. Risk-adjusted relative momentum (30%) ---
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

    # --- 4. Market regime adjustment (10%) ---
    regime_map = {'bull': 1.0, 'sideways': 0.5, 'bear': 0.15}
    regime_adj = regime_map.get(market_regime, 0.5)

    # --- 5. Volume confirmation (8%) ---
    vr = short_term_indicators.get('volume_ratio_3d', 1.0)
    volume = min(vr / 1.5, 1.0)

    # --- 6. OBV trend (5%) — accumulation/distribution signal ---
    obv = short_term_indicators.get('obv_trend', 0.0)
    obv_score = float(min(max(obv / 2.0, 0.0), 1.0))

    # --- 7. CMF (4%) — Chaikin Money Flow, volume-weighted accumulation ---
    cmf = short_term_indicators.get('cmf_20', 0.0)
    cmf_score = float(min(max(cmf / 0.3, 0.0), 1.0))

    # --- 8. RVOL (3%) — relative volume vs same-weekday baseline ---
    rvol = short_term_indicators.get('rvol_5', 1.0)
    rvol_score = min(rvol / 2.0, 1.0)

    # --- 9. ATR trend direction (5%) — volatility expansion/contraction ---
    atr_ratio = short_term_indicators.get('atr_trend_ratio', 1.0)
    if atr_ratio > 1.1:
        atr_score = min(1.0, (atr_ratio - 1.0) / 0.5)
    elif atr_ratio < 0.9:
        atr_score = 0.3
    else:
        atr_score = 0.6

    weights = {
        'momentum': 0.28,
        'sentiment': 0.22,
        'price_trend': 0.15,
        'regime_adj': 0.10,
        'volume': 0.08,
        'obv_trend': 0.05,
        'cmf': 0.04,
        'rvol': 0.03,
        'atr_trend': 0.05,
    }
    components = {
        'momentum': momentum,
        'sentiment': sentiment,
        'price_trend': price_trend,
        'regime_adj': regime_adj,
        'volume': volume,
        'obv_trend': obv_score,
        'cmf': cmf_score,
        'rvol': rvol_score,
        'atr_trend': atr_score,
    }
    growth_score = sum(components[k] * weights[k] for k in weights)
    growth_score = min(max(growth_score, 0.0), 1.0)

    return {
        'growth_score': growth_score,
        **components,
    }