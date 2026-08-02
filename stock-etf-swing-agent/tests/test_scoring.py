"""
Unit tests for scoring.py
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from scoring import (
    calculate_technical_score,
    calculate_fundamental_score,
    _compute_tracking_error,
    _compute_sharpe_1y,
)


class TestCalculateTechnicalScore:
    """Tests for calculate_technical_score()."""

    def test_returns_zero_for_empty_indicators(self):
        assert calculate_technical_score({}) == 0.0

    def test_score_in_valid_range(self, sample_indicators, default_technical_weights):
        score = calculate_technical_score(sample_indicators, default_technical_weights)
        assert 0.0 <= score <= 1.0

    def test_score_with_default_weights(self, sample_indicators):
        score = calculate_technical_score(sample_indicators)
        assert 0.0 <= score <= 1.0

    def test_perfect_indicators_give_high_score(self):
        """Indicators suggesting strong uptrend should score high."""
        perfect = {
            'price_vs_sma50': 0.05,
            'price_vs_sma200': 0.10,
            'sma50_vs_sma200': 0.03,
            'macd_histogram': 2.0,
            'rsi': 55,
            'roc_10': 5.0,
            'bb_position': 0.5,
            'price': 100,
            'sma_20': 99,
            'volume_ratio': 1.5,
            'volatility_20': 0.20,
            'adx': 35,
        }
        score = calculate_technical_score(perfect)
        assert score > 0.6

    def test_poor_indicators_give_low_score(self):
        """Indicators suggesting downtrend should score low."""
        poor = {
            'price_vs_sma50': -0.05,
            'price_vs_sma200': -0.10,
            'sma50_vs_sma200': -0.03,
            'macd_histogram': -2.0,
            'rsi': 25,
            'roc_10': -5.0,
            'bb_position': 0.1,
            'price': 90,
            'sma_20': 100,
            'volume_ratio': 0.3,
            'volatility_20': 0.50,
            'adx': 15,
        }
        score = calculate_technical_score(poor)
        assert score < 0.5

    def test_custom_weights_affect_score(self, sample_indicators):
        """Different weights should produce different scores."""
        trend_heavy = {
            'trend_score': 0.80,
            'momentum_score': 0.05,
            'mean_reversion_score': 0.05,
            'volume_score': 0.05,
            'volatility_score': 0.05,
        }
        momentum_heavy = {
            'trend_score': 0.05,
            'momentum_score': 0.80,
            'mean_reversion_score': 0.05,
            'volume_score': 0.05,
            'volatility_score': 0.05,
        }
        score_trend = calculate_technical_score(sample_indicators, trend_heavy)
        score_momentum = calculate_technical_score(sample_indicators, momentum_heavy)
        # Scores should differ since weights are different
        assert score_trend != score_momentum

    def test_adx_contributes_to_trend(self):
        """High ADX should increase the score."""
        base = {
            'price_vs_sma50': 0.02,
            'price_vs_sma200': 0.03,
            'sma50_vs_sma200': 0.01,
            'macd_histogram': 0.5,
            'rsi': 50,
            'roc_10': 2.0,
            'bb_position': 0.5,
            'price': 100,
            'sma_20': 99,
            'volume_ratio': 1.0,
            'volatility_20': 0.20,
            'adx': 20,
        }
        score_low_adx = calculate_technical_score(base)

        base['adx'] = 50
        score_high_adx = calculate_technical_score(base)

        assert score_high_adx > score_low_adx


class TestCalculateFundamentalScore:
    """Tests for calculate_fundamental_score()."""

    @patch('scoring.yf.Ticker')
    def test_returns_neutral_for_empty_info(self, mock_ticker):
        mock_ticker.return_value.info = {}
        score = calculate_fundamental_score('SPY')
        assert score == 0.5

    @patch('scoring.yf.Ticker')
    def test_score_in_valid_range(self, mock_ticker, mock_etf_info):
        mock_ticker.return_value.info = mock_etf_info
        score = calculate_fundamental_score('SPY')
        assert 0.0 <= score <= 1.0

    @patch('scoring.yf.Ticker')
    def test_low_expense_ratio_scores_high(self, mock_ticker, mock_etf_info):
        mock_etf_info['annualReportExpenseRatio'] = 0.0003  # 0.03%
        mock_ticker.return_value.info = mock_etf_info
        score_low = calculate_fundamental_score('SPY')

        mock_etf_info['annualReportExpenseRatio'] = 0.01  # 1%
        mock_ticker.return_value.info = mock_etf_info
        score_high = calculate_fundamental_score('SPY')

        assert score_low > score_high

    @patch('scoring.yf.Ticker')
    def test_handles_exception_gracefully(self, mock_ticker):
        mock_ticker.side_effect = Exception("API error")
        score = calculate_fundamental_score('SPY')
        assert score == 0.5

    @patch('scoring.yf.Ticker')
    def test_with_price_data_computes_real_metrics(self, mock_ticker, mock_etf_info,
                                                    sample_ohlcv_data):
        """With different ETF vs benchmark data, scores should differ from neutral."""
        mock_ticker.return_value.info = mock_etf_info

        # Create a different benchmark dataset (more volatile, different path)
        import pandas as pd
        n = len(sample_ohlcv_data)
        bench_close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
        bench_data = pd.DataFrame({'Close': bench_close})

        score_with_data = calculate_fundamental_score(
            'SPY', price_data=sample_ohlcv_data, benchmark_data=bench_data
        )
        score_without_data = calculate_fundamental_score('SPY')
        # Scores should differ because tracking_error and sharpe are computed from real data
        assert score_with_data != score_without_data


class TestTrackingError:
    """Tests for _compute_tracking_error()."""

    def test_returns_neutral_for_none_data(self):
        assert _compute_tracking_error(None, None) == 0.8

    def test_perfect_tracking_scores_high(self, sample_ohlcv_data):
        """ETF tracking itself perfectly should score 1.0."""
        score = _compute_tracking_error(sample_ohlcv_data, sample_ohlcv_data)
        assert score == 1.0

    def test_short_data_returns_neutral(self):
        import pandas as pd
        short = pd.DataFrame({'Close': [100, 101, 102]})
        assert _compute_tracking_error(short, short) == 0.8


class TestSharpe1Y:
    """Tests for _compute_sharpe_1y()."""

    def test_returns_neutral_for_none_data(self):
        assert _compute_sharpe_1y(None) == 0.6

    def test_positive_trend_scores_high(self):
        """A steady uptrend should produce a high Sharpe score."""
        import pandas as pd
        n = 252
        close = 100 * np.exp(np.cumsum(np.full(n, 0.001)))  # ~28% annual
        df = pd.DataFrame({'Close': close})
        score = _compute_sharpe_1y(df)
        assert score >= 0.8

    def test_negative_trend_scores_low(self):
        """A steady downtrend should produce a low Sharpe score."""
        import pandas as pd
        n = 252
        close = 100 * np.exp(np.cumsum(np.full(n, -0.001)))
        df = pd.DataFrame({'Close': close})
        score = _compute_sharpe_1y(df)
        assert score <= 0.4