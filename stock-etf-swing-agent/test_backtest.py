"""
Test script for the backtesting framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import ETFBacktester
from datetime import datetime, timedelta

def test_backtest():
    """Test the backtester with a small universe and short period."""
    print("Testing ETF Backtester...")
    
    # Define a small ETF universe for quick testing
    test_universe = ['SPY', 'QQQ', 'VTI']
    
    # Define a short backtest period (last 3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*30)  # Approximately 3 months
    
    # Create backtester
    backtester = ETFBacktester(etf_universe=test_universe, lookback_months=3)
    
    # Run backtest with monthly rebalancing (will be about 3 periods)
    print(f"Running backtest from {start_date.date()} to {end_date.date()}...")
    results = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        rebalancing_freq='M'
    )
    
    # Print results
    if "error" in results:
        print(f"Backtest failed: {results['error']}")
        return False
    
    print("\n" + "="*50)
    print("BACKTEST TEST RESULTS")
    print("="*50)
    print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
    print(f"Rebalancing: {results['rebalancing_freq']}")
    print(f"Number of periods: {results['num_periods']}")
    print("-"*50)
    print(f"Strategy Cumulative Return: {results['strategy_cumulative_return']:.2%}")
    print(f"Benchmark Cumulative Return: {results['benchmark_cumulative_return']:.2%}")
    print(f"Strategy Annual Return: {results['strategy_annual_return']:.2%}")
    print(f"Benchmark Annual Return: {results['benchmark_annual_return']:.2%}")
    print(f"Strategy Sharpe Ratio: {results['strategy_sharpe_ratio']:.2f}")
    print(f"Benchmark Sharpe Ratio: {results['benchmark_sharpe_ratio']:.2f}")
    print(f"Strategy Max Drawdown: {results['strategy_max_drawdown']:.2%}")
    print(f"Benchmark Max Drawdown: {results['benchmark_max_drawdown']:.2%}")
    print(f"Strategy Sortino: {results.get('strategy_sortino_ratio', 0):.2f}")
    print(f"Win Rate: {results.get('win_rate', 0):.1%}  |  Info Ratio: {results.get('information_ratio', 0):.2f}")
    print("="*50)
    
    # Check that we got reasonable results
    if results['num_periods'] > 0:
        print("✅ Backtest completed successfully")
        return True
    else:
        print("❌ Backtest produced no periods")
        return False

if __name__ == "__main__":
    success = test_backtest()
    sys.exit(0 if success else 1)