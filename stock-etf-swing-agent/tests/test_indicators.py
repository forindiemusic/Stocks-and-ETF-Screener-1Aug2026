"""
Unit tests for indicators.py
"""

import pytest
import pandas as pd
import numpy as np
from indicators import (
    calculate_technical_indicators,
    calculate_short_term_indicators,
    calculate_short_term_score,
    calculate_day_trade_indicators,
    calculate_day_trade_score,
)


class TestCalculateTechnicalIndicators:
    """Tests for calculate_technical_indicators()."""

    def test_returns_empty_dict_for_none_data(self):
        result = calculate_technical_indicators(None)
        assert result == {}

    def test_returns_empty_dict_for_short_data(self):
        df = pd.DataFrame({'Close': [100] * 10, 'High': [101] * 10,
                          'Low': [99] * 10, 'Volume': [1e6] * 10})
        result = calculate_technical_indicators(df)
        assert result == {}

    def test_returns_all_expected_keys(self, sample_ohlcv_data):
        result = calculate_technical_indicators(sample_ohlcv_data)
        expected_keys = {
            'sma_20', 'sma_50', 'sma_200', 'price',
            'price_vs_sma50', 'price_vs_sma200', 'sma50_vs_sma200',
            'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'adx',
            'bb_upper', 'bb_lower', 'bb_position',
            'roc_10', 'roc_20',
            'volume_sma_20', 'volume_ratio',
            'atr_14', 'volatility_20',
        }
        assert set(result.keys()) == expected_keys

    def test_rsi_in_valid_range(self, sample_ohlcv_data):
        result = calculate_technical_indicators(sample_ohlcv_data)
        assert 0 <= result['rsi'] <= 100

    def test_adx_in_valid_range(self, sample_ohlcv_data):
        result = calculate_technical_indicators(sample_ohlcv_data)
        assert 0 <= result['adx'] <= 100

    def test_bb_position_in_valid_range(self, sample_ohlcv_data):
        result = calculate_technical_indicators(sample_ohlcv_data)
        assert 0 <= result['bb_position'] <= 1

    def test_macd_histogram_equals_macd_minus_signal(self, sample_ohlcv_data):
        result = calculate_technical_indicators(sample_ohlcv_data)
        expected = result['macd'] - result['macd_signal']
        assert abs(result['macd_histogram'] - expected) < 0.001

    def test_price_above_sma20_in_uptrend(self):
        """In a pure uptrend, price should be above SMA 20."""
        n = 100
        close = np.linspace(100, 200, n)
        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': [1e6] * n,
        })
        result = calculate_technical_indicators(df)
        assert result['price'] > result['sma_20']

    def test_rsi_oversold_in_downtrend(self):
        """In a steep downtrend, RSI should be low."""
        n = 100
        close = np.linspace(200, 100, n)
        df = pd.DataFrame({
            'Open': close * 1.01,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': [1e6] * n,
        })
        result = calculate_technical_indicators(df)
        assert result['rsi'] < 40

    def test_atr_positive(self, sample_ohlcv_data):
        result = calculate_technical_indicators(sample_ohlcv_data)
        assert result['atr_14'] > 0

    def test_volatility_annualized(self, sample_ohlcv_data):
        """Annualized vol should be roughly sqrt(252) * last 20-day daily vol."""
        result = calculate_technical_indicators(sample_ohlcv_data)
        # volatility_20 uses rolling(20).std() of the last 20 days
        last_20_returns = sample_ohlcv_data['Close'].pct_change(fill_method=None).iloc[-20:]
        expected = last_20_returns.std() * np.sqrt(252)
        assert abs(result['volatility_20'] - expected) < 0.01

    def test_all_values_finite(self, sample_ohlcv_data):
        result = calculate_technical_indicators(sample_ohlcv_data)
        for key, value in result.items():
            assert np.isfinite(value), f"{key} = {value} is not finite"


class TestCalculateShortTermIndicators:
    """Tests for calculate_short_term_indicators()."""

    def test_returns_empty_dict_for_none_data(self):
        assert calculate_short_term_indicators(None) == {}

    def test_returns_empty_dict_for_short_data(self):
        df = pd.DataFrame({'Close': [100] * 10, 'High': [101] * 10,
                          'Low': [99] * 10, 'Volume': [1e6] * 10})
        assert calculate_short_term_indicators(df) == {}

    def test_returns_all_expected_keys(self, sample_ohlcv_data):
        result = calculate_short_term_indicators(sample_ohlcv_data)
        expected_keys = {
            'price', 'sma_5', 'sma_10', 'sma_20',
            'roc_5', 'roc_10', 'rsi_5',
            'macd_histogram', 'volume_ratio_3d', 'atr_5',
        }
        assert set(result.keys()) == expected_keys

    def test_rsi_5_in_valid_range(self, sample_ohlcv_data):
        result = calculate_short_term_indicators(sample_ohlcv_data)
        assert 0 <= result['rsi_5'] <= 100

    def test_atr_5_positive(self, sample_ohlcv_data):
        result = calculate_short_term_indicators(sample_ohlcv_data)
        assert result['atr_5'] > 0

    def test_sma_alignment_in_uptrend(self):
        """In a pure uptrend, price > SMA5 > SMA10 > SMA20."""
        n = 50
        close = np.linspace(100, 200, n)
        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': [1e6] * n,
        })
        result = calculate_short_term_indicators(df)
        assert result['price'] > result['sma_5'] > result['sma_10'] > result['sma_20']

    def test_all_values_finite(self, sample_ohlcv_data):
        result = calculate_short_term_indicators(sample_ohlcv_data)
        for key, value in result.items():
            assert np.isfinite(value), f"{key} = {value} is not finite"


class TestCalculateShortTermScore:
    """Tests for calculate_short_term_score()."""

    def test_returns_zero_for_empty_indicators(self):
        assert calculate_short_term_score({}) == 0.0

    def test_score_in_valid_range(self, sample_ohlcv_data):
        ind = calculate_short_term_indicators(sample_ohlcv_data)
        score = calculate_short_term_score(ind)
        assert 0.0 <= score <= 1.0

    def test_bullish_alignment_gives_high_score(self):
        """Strong uptrend with bullish MA structure should score high."""
        n = 50
        close = np.linspace(100, 200, n)
        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': [1e6] * n,
        })
        ind = calculate_short_term_indicators(df)
        score = calculate_short_term_score(ind)
        assert score > 0.5

    def test_bearish_alignment_gives_low_score(self):
        """Steep downtrend should score low."""
        n = 50
        close = np.linspace(200, 100, n)
        df = pd.DataFrame({
            'Open': close * 1.01,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': [1e6] * n,
        })
        ind = calculate_short_term_indicators(df)
        score = calculate_short_term_score(ind)
        assert score < 0.5


class TestCalculateDayTradeIndicators:
    """Tests for calculate_day_trade_indicators()."""

    def test_returns_empty_dict_for_none_data(self):
        assert calculate_day_trade_indicators(None) == {}

    def test_returns_empty_dict_for_short_data(self):
        df = pd.DataFrame({'Close': [100] * 5, 'High': [101] * 5,
                          'Low': [99] * 5, 'Volume': [1e6] * 5,
                          'Open': [100] * 5})
        assert calculate_day_trade_indicators(df) == {}

    def test_returns_all_expected_keys(self, sample_ohlcv_data):
        result = calculate_day_trade_indicators(sample_ohlcv_data)
        expected_keys = {
            'price', 'sma_3', 'sma_5',
            'roc_1', 'roc_2', 'roc_3',
            'momentum_accel', 'gap_pct', 'gap_filled',
            'rsi_2', 'bb_width', 'bb_squeeze',
            'high_5', 'low_5', 'proximity_high_5',
            'volume_ratio_1d', 'intraday_range_pct', 'atr_2',
        }
        assert set(result.keys()) == expected_keys

    def test_rsi_2_in_valid_range(self, sample_ohlcv_data):
        result = calculate_day_trade_indicators(sample_ohlcv_data)
        assert 0 <= result['rsi_2'] <= 100

    def test_proximity_in_valid_range(self, sample_ohlcv_data):
        result = calculate_day_trade_indicators(sample_ohlcv_data)
        assert 0 <= result['proximity_high_5'] <= 1

    def test_gap_filled_is_binary(self, sample_ohlcv_data):
        result = calculate_day_trade_indicators(sample_ohlcv_data)
        assert result['gap_filled'] in (0.0, 1.0)

    def test_uptrend_proximity_high(self):
        """In a strong uptrend, price should be near 5-day high."""
        n = 30
        close = np.linspace(100, 150, n)
        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': [1e6] * n,
        })
        result = calculate_day_trade_indicators(df)
        assert result['proximity_high_5'] > 0.5

    def test_all_values_finite(self, sample_ohlcv_data):
        result = calculate_day_trade_indicators(sample_ohlcv_data)
        for key, value in result.items():
            assert np.isfinite(value), f"{key} = {value} is not finite"


class TestCalculateDayTradeScore:
    """Tests for calculate_day_trade_score()."""

    def test_returns_zero_for_empty_indicators(self):
        assert calculate_day_trade_score({}) == 0.0

    def test_score_in_valid_range(self, sample_ohlcv_data):
        ind = calculate_day_trade_indicators(sample_ohlcv_data)
        score = calculate_day_trade_score(ind)
        assert 0.0 <= score <= 1.0

    def test_bullish_trend_gives_higher_score(self):
        """Strong uptrend should score higher than downtrend."""
        n = 30
        # Uptrend
        close_up = np.linspace(100, 150, n)
        df_up = pd.DataFrame({
            'Open': close_up * 0.99,
            'High': close_up * 1.02,
            'Low': close_up * 0.98,
            'Close': close_up,
            'Volume': [1e6] * n,
        })
        # Downtrend
        close_down = np.linspace(150, 100, n)
        df_down = pd.DataFrame({
            'Open': close_down * 1.01,
            'High': close_down * 1.02,
            'Low': close_down * 0.98,
            'Close': close_down,
            'Volume': [1e6] * n,
        })
        ind_up = calculate_day_trade_indicators(df_up)
        ind_down = calculate_day_trade_indicators(df_down)
        score_up = calculate_day_trade_score(ind_up)
        score_down = calculate_day_trade_score(ind_down)
        assert score_up > score_down

    def test_spy_relative_scoring(self, sample_ohlcv_data):
        """Score with SPY indicators should still be in valid range."""
        ind = calculate_day_trade_indicators(sample_ohlcv_data)
        spy_ind = calculate_day_trade_indicators(sample_ohlcv_data)
        score = calculate_day_trade_score(ind, spy_indicators=spy_ind)
        assert 0.0 <= score <= 1.0