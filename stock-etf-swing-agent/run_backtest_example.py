"""
Example script demonstrating how to run a backtest with the ETF Swing Trading Agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import ETFBacktester
from datetime import datetime, timedelta
import pandas as pd


def _fmt(value: float) -> str:
    """Format a ratio, handling infinity gracefully."""
    if value == float('inf'):
        return '∞'
    if value == float('-inf'):
        return '-∞'
    return f'{value:.2f}'


def run_example_backtest():
    """Run an example backtest with a reasonable universe and time period."""
    print("ETF Swing Trading Agent - Backtesting Example")
    print("=" * 50)
    
    # Define ETF universe (you can customize this)
    etf_universe = [
        # Broad Market
        'SPY', 'VOO', 'QQQ', 'VTI',
        # Sectors
        'XLK', 'XLF', 'XLV', 'XLE', 'XLI',
        # Factors
        'USMV', 'MTUM', 'QUAL', 'VLUE',
        # International
        'VEA', 'VWO', 'VGK', 'EWJ'
    ]
    
    # Define backtest period (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6*30)  # Approximately 6 months
    
    print(f"Backtesting from {start_date.date()} to {end_date.date()}")
    print(f"ETF Universe: {len(etf_universe)} ETFs")
    print(f"Lookback period for evaluation: 6 months")
    print(f"Rebalancing frequency: Monthly")
    print("-" * 50)
    
    # Create backtester
    backtester = ETFBacktester(
        etf_universe=etf_universe,
        lookback_months=6  # Use 6 months of data for evaluation
    )
    
    # Run backtest
    print("Running backtest...")
    results = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        rebalancing_freq='M'  # Monthly rebalancing
    )
    
    # Display results
    if "error" in results:
        print(f"❌ Backtest failed: {results['error']}")
        return False
    
    print("✅ Backtest completed successfully!")
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
    print(f"Rebalancing: {results['rebalancing_freq']}")
    print(f"Number of periods: {results['num_periods']}")
    print("-" * 50)
    print("RETURNS:")
    print(f"  Strategy Cumulative Return: {results['strategy_cumulative_return']:.2%}")
    print(f"  Benchmark (SPY) Cumulative Return: {results['benchmark_cumulative_return']:.2%}")
    print(f"  Strategy Annual Return: {results['strategy_annual_return']:.2%}")
    print(f"  Benchmark Annual Return: {results['benchmark_annual_return']:.2%}")
    print(f"  Excess Return: {results['strategy_cumulative_return'] - results['benchmark_cumulative_return']:.2%}")
    print("-" * 50)
    print("RISK METRICS:")
    print(f"  Strategy Sharpe Ratio: {results['strategy_sharpe_ratio']:.2f}")
    print(f"  Strategy Sortino Ratio: {_fmt(results.get('strategy_sortino_ratio', 0))}")
    print(f"  Strategy Calmar Ratio: {_fmt(results.get('strategy_calmar_ratio', 0))}")
    print(f"  Benchmark Sharpe Ratio: {results['benchmark_sharpe_ratio']:.2f}")
    print(f"  Strategy Max Drawdown: {results['strategy_max_drawdown']:.2%}")
    print(f"  Benchmark Max Drawdown: {results['benchmark_max_drawdown']:.2%}")
    print(f"  Strategy Volatility (Annual): {results['strategy_annual_volatility']:.2%}")
    print(f"  Benchmark Volatility (Annual): {results['benchmark_annual_volatility']:.2%}")
    print(f"  Information Ratio: {_fmt(results.get('information_ratio', 0))}")
    print("-" * 50)
    print("TRADE STATISTICS:")
    print(f"  Win Rate: {results.get('win_rate', 0):.1%}")
    print(f"  Avg Win: {results.get('avg_win', 0):.2%}  |  Avg Loss: {results.get('avg_loss', 0):.2%}")
    print(f"  Win/Loss Ratio: {_fmt(results.get('win_loss_ratio', 0))}")
    print("-" * 50)
    print("TRANSACTION COSTS:")
    print(f"  Cost per trade: {results.get('txn_cost_bps', 0):.1f} bps")
    print(f"  Total txn costs: {results.get('total_txn_costs', 0):.4%}")
    print(f"  Avg turnover: {results.get('avg_turnover_pct', 0):.1f}%")
    print("-" * 50)
    print("RISK MANAGEMENT:")
    print(f"  Stop-loss: {results.get('stop_loss_atr_mult', 0):.1f}x ATR  |  Take-profit: {results.get('take_profit_atr_mult', 0):.1f}x ATR")
    print(f"  Stop-loss hits: {results.get('stop_loss_hits', 0)}  |  Take-profit hits: {results.get('take_profit_hits', 0)}")
    print("=" * 50)
    
    # Show equity curve if available
    if 'equity_curve' in results and len(results['equity_curve']) > 0:
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        print("\nEQUITY CURVE (last 5 points):")
        print(equity_df.tail())
        
        # Save equity curve to CSV
        equity_df.to_csv('./output/backtest_equity_curve.csv')
        print("\n💾 Equity curve saved to ./output/backtest_equity_curve.csv")
    
    # Show portfolio holdings history if available
    if 'portfolio_history' in results and len(results['portfolio_history']) > 0:
        print("\nPORTFOLIO HOLDINGS HISTORY (last 3 rebalances):")
        for i, holding in enumerate(results['portfolio_history'][-3:]):
            date = holding['rebalancing_date'].strftime('%Y-%m-%d') if hasattr(holding['rebalancing_date'], 'strftime') else str(holding['rebalancing_date'])
            holdings_str = ', '.join(holding['holdings'])
            print(f"  {date}: {holdings_str}")
    
    print("\n" + "=" * 50)
    print("Backtest completed successfully!")
    print("To run a different backtest, modify the parameters in this script.")
    print("=" * 50)
    
    # --- Walk-Forward Validation ---
    print("\n\n")
    print("=" * 50)
    print("RUNNING WALK-FORWARD VALIDATION")
    print("=" * 50)
    
    wf_results = backtester.run_walk_forward(
        start_date=start_date,
        end_date=end_date,
        window_months=12,
        step_months=6,
        rebalancing_freq='M',
    )
    backtester.print_walk_forward_summary(wf_results)
    
    return True

if __name__ == "__main__":
    success = run_example_backtest()
    sys.exit(0 if success else 1)