"""
Shared pytest fixtures for ETF Swing Trading Agent tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Generate realistic OHLCV data for 200 trading days."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(end=datetime.now(), periods=n, freq='B')

    # Random walk with drift for close prices
    returns = np.random.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    daily_range = close * np.random.uniform(0.005, 0.03, n)
    high = close + daily_range * np.random.uniform(0.3, 0.7, n)
    low = close - daily_range * np.random.uniform(0.3, 0.7, n)
    open_price = close - returns * close  # approximate

    volume = np.random.randint(1_000_000, 10_000_000, n)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)

    return df


@pytest.fixture
def sample_indicators(sample_ohlcv_data):
    """Pre-computed indicators from sample data."""
    from indicators import calculate_technical_indicators
    return calculate_technical_indicators(sample_ohlcv_data)


@pytest.fixture
def default_technical_weights():
    return {
        'trend_score': 0.30,
        'momentum_score': 0.25,
        'mean_reversion_score': 0.20,
        'volume_score': 0.15,
        'volatility_score': 0.10,
    }


@pytest.fixture
def default_fundamental_weights():
    return {
        'expense_ratio': 0.20,
        'liquidity': 0.20,
        'tracking_error': 0.15,
        'aum': 0.15,
        'yield': 0.15,
        'sharpe_1y': 0.15,
    }


@pytest.fixture
def mock_etf_info():
    """Mock yfinance Ticker.info response."""
    return {
        'longName': 'SPDR S&P 500 ETF Trust',
        'shortName': 'SPY',
        'annualReportExpenseRatio': 0.000945,
        'averageVolume': 75_000_000,
        'totalAssets': 500_000_000_000,
        'yield': 0.013,
    }


@pytest.fixture
def sample_returns():
    """Sample period returns for backtest metric testing."""
    return [0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.01, 0.025, -0.015, 0.005]