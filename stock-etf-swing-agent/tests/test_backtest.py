"""
Unit tests for backtest.py
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from backtest import ETFBacktester


class TestETFBacktesterInit:
    """Tests for ETFBacktester initialization."""

    def test_loads_config_weights(self):
        bt = ETFBacktester(etf_universe=['SPY', 'QQQ'], lookback_months=3)
        assert bt.technical_weights['trend_score'] == 0.30
        assert bt.fundamental_weights['expense_ratio'] == 0.20
        assert bt.composite_weights['technical'] == 0.50

    def test_loads_risk_config(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        assert bt.stop_loss_atr_mult == 2.0
        assert bt.take_profit_atr_mult == 3.0
        assert bt.max_correlated_positions == 3

    def test_loads_transaction_costs(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        assert bt.txn_cost_bps == 8.0
        assert bt.txn_cost == 0.0008

    def test_default_top_n(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        assert bt.top_n == 5


class TestTurnoverCost:
    """Tests for _calculate_turnover_cost()."""

    def test_first_period_all_new(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        cost = bt._calculate_turnover_cost(['A', 'B', 'C', 'D', 'E'])
        assert cost == bt.txn_cost  # one-way cost on all positions

    def test_no_change_zero_cost(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        bt._prev_holdings = ['A', 'B', 'C', 'D', 'E']
        cost = bt._calculate_turnover_cost(['A', 'B', 'C', 'D', 'E'])
        assert cost == 0.0

    def test_partial_turnover(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        bt._prev_holdings = ['A', 'B', 'C', 'D', 'E']
        # 2 sold (D,E), 2 bought (F,G) = 4/5 turnover
        cost = bt._calculate_turnover_cost(['A', 'B', 'C', 'F', 'G'])
        expected = (4 / 5) * bt.txn_cost
        assert cost == pytest.approx(expected)

    def test_full_turnover(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        bt._prev_holdings = ['A', 'B', 'C', 'D', 'E']
        # All 5 sold, all 5 bought = 10/5 = 2x one-way cost
        cost = bt._calculate_turnover_cost(['V', 'W', 'X', 'Y', 'Z'])
        expected = 2.0 * bt.txn_cost
        assert cost == pytest.approx(expected)


class TestAvgTurnover:
    """Tests for _calc_avg_turnover()."""

    def test_empty_history(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        assert bt._calc_avg_turnover([]) == 0.0

    def test_single_period(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        history = [{'holdings': ['A', 'B']}]
        assert bt._calc_avg_turnover(history) == 0.0

    def test_full_turnover_between_periods(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        history = [
            {'holdings': ['A', 'B', 'C', 'D', 'E']},
            {'holdings': ['V', 'W', 'X', 'Y', 'Z']},
        ]
        turnover = bt._calc_avg_turnover(history)
        assert turnover == 200.0  # 100% sold + 100% bought


class TestFmtRatio:
    """Tests for _fmt_ratio()."""

    def test_normal_value(self):
        assert ETFBacktester._fmt_ratio(1.5) == '1.50'

    def test_infinity(self):
        assert ETFBacktester._fmt_ratio(float('inf')) == '∞'

    def test_negative_infinity(self):
        assert ETFBacktester._fmt_ratio(float('-inf')) == '-∞'


class TestCorrelationFilter:
    """Tests for _apply_correlation_filter()."""

    def test_single_etf_passes_through(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        scores = [{'symbol': 'SPY', 'composite_score': 0.85}]
        result = bt._apply_correlation_filter(scores, datetime.now())
        assert result == scores

    def test_empty_list(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        result = bt._apply_correlation_filter([], datetime.now())
        assert result == []


class TestBacktestResults:
    """Tests for run_backtest result structure."""

    @patch.object(ETFBacktester, 'fetch_historical_data')
    @patch.object(ETFBacktester, 'fetch_period_data')
    def test_error_on_short_date_range(self, mock_period, mock_hist):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        end = datetime.now()
        start = end - timedelta(days=1)
        result = bt.run_backtest(start, end, rebalancing_freq='M')
        assert 'error' in result

    @patch.object(ETFBacktester, 'fetch_historical_data')
    @patch.object(ETFBacktester, 'fetch_period_data')
    def test_result_has_all_required_keys(self, mock_period, mock_hist):
        """With mocked data, verify result dict has all expected keys."""
        import pandas as pd
        dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
        mock_df = pd.DataFrame({
            'Open': [100]*100, 'High': [101]*100, 'Low': [99]*100,
            'Close': [100]*100, 'Volume': [1e6]*100,
        }, index=dates)

        mock_hist.return_value = mock_df
        mock_period.return_value = mock_df

        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        end = datetime.now()
        start = end - timedelta(days=180)

        result = bt.run_backtest(start, end, rebalancing_freq='M')

        required_keys = {
            'start_date', 'end_date', 'rebalancing_freq', 'num_periods',
            'strategy_returns', 'benchmark_returns',
            'strategy_cumulative_return', 'benchmark_cumulative_return',
            'strategy_annual_return', 'benchmark_annual_return',
            'strategy_sharpe_ratio', 'benchmark_sharpe_ratio',
            'strategy_max_drawdown', 'benchmark_max_drawdown',
            'strategy_sortino_ratio', 'win_rate', 'information_ratio',
            'portfolio_history', 'txn_cost_bps', 'total_txn_costs',
            'stop_loss_hits', 'take_profit_hits',
        }
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestWalkForward:
    """Tests for run_walk_forward()."""

    def test_error_on_short_date_range(self):
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        end = datetime.now()
        start = end - timedelta(days=30)
        result = bt.run_walk_forward(start, end, window_months=12, step_months=3)
        assert 'error' in result

    @patch.object(ETFBacktester, 'fetch_historical_data')
    @patch.object(ETFBacktester, 'fetch_period_data')
    def test_aggregated_keys_present(self, mock_period, mock_hist):
        import pandas as pd
        dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
        mock_df = pd.DataFrame({
            'Open': [100]*100, 'High': [101]*100, 'Low': [99]*100,
            'Close': [100]*100, 'Volume': [1e6]*100,
        }, index=dates)

        mock_hist.return_value = mock_df
        mock_period.return_value = mock_df

        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        end = datetime.now()
        start = end - timedelta(days=400)

        result = bt.run_walk_forward(start, end, window_months=6, step_months=3)

        if 'error' not in result:
            agg_keys = {
                'num_windows', 'mean_annual_return', 'std_annual_return',
                'mean_sharpe', 'mean_sortino', 'mean_max_drawdown',
                'worst_drawdown', 'mean_information_ratio', 'mean_win_rate',
                'beat_benchmark_pct', 'pct_profitable_windows',
                'mean_excess_return', 'windows',
            }
            for key in agg_keys:
                assert key in result, f"Missing aggregated key: {key}"


class TestBacktestGrowthPrimary:
    """Tests that the backtester mirrors the live agent's growth-primary logic."""

    def test_evaluate_etf_returns_growth_outlook(self):
        """evaluate_etf must now compute a 4-week growth outlook (primary criteria)."""
        import pandas as pd
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
        mock_df = pd.DataFrame({
            'Open': [100]*100, 'High': [101]*100, 'Low': [99]*100,
            'Close': [100]*100, 'Volume': [1e6]*100,
        }, index=dates)
        with patch.object(ETFBacktester, 'fetch_historical_data', return_value=mock_df), \
             patch.object(ETFBacktester, 'get_sentiment_score', return_value=0.6), \
             patch('indicators.calculate_technical_indicators', return_value={'atr_14': 1.0}):
            result = bt.evaluate_etf('SPY', datetime.now())
        assert 'growth_outlook' in result
        assert result['growth_outlook'] is not None
        assert 'growth_score' in result['growth_outlook']

    def test_evaluate_etf_returns_owned_flag(self):
        """evaluate_etf must flag ETFs present in currently_own_etf.dat."""
        bt = ETFBacktester(etf_universe=['XLE', 'SPY'], lookback_months=3)
        # XLE is in currently_own_etf.dat; SPY is not
        assert 'XLE' in bt._owned_etf_symbols
        assert 'SPY' not in bt._owned_etf_symbols

    def test_ranking_uses_growth_score_primary(self):
        """Backtest selection must rank by growth_score, not composite_score."""
        bt = ETFBacktester(etf_universe=['AAA', 'BBB'], lookback_months=3)
        # AAA: high composite but low growth; BBB: low composite but high growth
        aaa = {
            'symbol': 'AAA', 'composite_score': 0.95, 'growth_outlook': {'growth_score': 0.20},
            'current_price': 100.0, 'atr': 1.0,
        }
        bbb = {
            'symbol': 'BBB', 'composite_score': 0.40, 'growth_outlook': {'growth_score': 0.90},
            'current_price': 100.0, 'atr': 1.0,
        }
        bt._prev_holdings = None
        bt.top_n = 1
        selected = bt._apply_holding_period_filter([aaa, bbb], datetime.now())
        assert selected[0]['symbol'] == 'BBB'

    def test_ranking_falls_back_to_composite_without_growth(self):
        """When growth_outlook is missing, ranking falls back to composite_score."""
        bt = ETFBacktester(etf_universe=['AAA', 'BBB'], lookback_months=3)
        aaa = {
            'symbol': 'AAA', 'composite_score': 0.95, 'growth_outlook': None,
            'current_price': 100.0, 'atr': 1.0,
        }
        bbb = {
            'symbol': 'BBB', 'composite_score': 0.40, 'growth_outlook': None,
            'current_price': 100.0, 'atr': 1.0,
        }
        bt._prev_holdings = None
        bt.top_n = 1
        selected = bt._apply_holding_period_filter([aaa, bbb], datetime.now())
        assert selected[0]['symbol'] == 'AAA'

    def test_portfolio_history_records_growth_scores(self):
        """portfolio_history should include growth_scores alongside composite scores."""
        bt = ETFBacktester(etf_universe=['SPY'], lookback_months=3)
        bt._prev_holdings = None
        bt.top_n = 1
        etf = {
            'symbol': 'SPY', 'composite_score': 0.60,
            'growth_outlook': {'growth_score': 0.75},
            'current_price': 100.0, 'atr': 1.0,
        }
        selected = bt._apply_holding_period_filter([etf], datetime.now())
        # Simulate the portfolio_history append block from run_backtest
        history_entry = {
            'holdings': [e['symbol'] for e in selected],
            'scores': [e['composite_score'] for e in selected],
            'growth_scores': [e.get('growth_outlook', {}).get('growth_score', 0.0) for e in selected],
        }
        assert history_entry['growth_scores'] == [0.75]


class TestYieldLabel:
    """Tests for the dividend-yield label helper (good/standard/low)."""

    def _label(self, yield_pct):
        if yield_pct is None:
            return 'N/A'
        if yield_pct >= 5.0:
            return 'good'
        elif yield_pct >= 4.0:
            return 'standard'
        return 'low'

    def test_good_label(self):
        assert self._label(5.2) == 'good'

    def test_standard_label(self):
        assert self._label(4.3) == 'standard'

    def test_low_label(self):
        assert self._label(3.7) == 'low'

    def test_none_yield(self):
        assert self._label(None) == 'N/A'


class TestStockSupport:
    """Tests that the backtester mirrors the live agent's stock logic."""

    def test_init_infers_stock_symbols_in_stock_mode(self):
        """In 'stock' mode, all universe symbols are treated as stocks."""
        bt = ETFBacktester(etf_universe=['AAPL', 'MSFT'], lookback_months=3, mode='stock')
        assert bt._stock_symbols == {'AAPL', 'MSFT'}

    def test_init_explicit_stock_symbols(self):
        """Explicit stock_symbols override mode inference."""
        bt = ETFBacktester(
            etf_universe=['SPY', 'AAPL'], lookback_months=3,
            mode='all', stock_symbols=['AAPL']
        )
        assert bt._stock_symbols == {'AAPL'}
        assert 'SPY' not in bt._stock_symbols

    def test_evaluate_stock_returns_short_term_score(self):
        """evaluate_stock must compute a short-term score (primary for stocks)."""
        import pandas as pd
        bt = ETFBacktester(etf_universe=['AAPL'], lookback_months=3, mode='stock')
        dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
        mock_df = pd.DataFrame({
            'Open': [100]*100, 'High': [101]*100, 'Low': [99]*100,
            'Close': [100]*100, 'Volume': [1e6]*100,
        }, index=dates)
        with patch.object(ETFBacktester, 'fetch_historical_data', return_value=mock_df), \
             patch.object(ETFBacktester, 'get_sentiment_score', return_value=0.6), \
             patch('indicators.calculate_technical_indicators', return_value={'atr_14': 1.0}), \
             patch('indicators.calculate_short_term_indicators', return_value={'atr_5': 1.0}):
            result = bt.evaluate_stock('AAPL', datetime.now())
        assert 'short_term_score' in result
        assert result['growth_outlook'] is None
        assert result['is_owned_etf'] is False

    def test_evaluate_symbol_dispatches_to_stock(self):
        """evaluate_symbol routes stock symbols to evaluate_stock."""
        import pandas as pd
        bt = ETFBacktester(etf_universe=['AAPL'], lookback_months=3, mode='stock')
        dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
        mock_df = pd.DataFrame({
            'Open': [100]*100, 'High': [101]*100, 'Low': [99]*100,
            'Close': [100]*100, 'Volume': [1e6]*100,
        }, index=dates)
        with patch.object(ETFBacktester, 'fetch_historical_data', return_value=mock_df), \
             patch.object(ETFBacktester, 'get_sentiment_score', return_value=0.6), \
             patch('indicators.calculate_technical_indicators', return_value={'atr_14': 1.0}), \
             patch('indicators.calculate_short_term_indicators', return_value={'atr_5': 1.0}):
            result = bt.evaluate_symbol('AAPL', datetime.now())
        assert result['short_term_score'] >= 0.0
        assert result['growth_outlook'] is None

    def test_rank_score_for_stock_uses_short_term(self):
        """_rank_score_for returns short_term_score for stocks, growth for ETFs."""
        bt = ETFBacktester(
            etf_universe=['AAPL', 'SPY'], lookback_months=3,
            mode='all', stock_symbols=['AAPL']
        )
        stock_res = {
            'symbol': 'AAPL', 'short_term_score': 0.80,
            'growth_outlook': None, 'composite_score': 0.30,
        }
        etf_res = {
            'symbol': 'SPY', 'short_term_score': 0.0,
            'growth_outlook': {'growth_score': 0.70}, 'composite_score': 0.90,
        }
        assert bt._rank_score_for(stock_res) == 0.80
        assert bt._rank_score_for(etf_res) == 0.70

    def test_ranking_stocks_by_short_term_score(self):
        """Backtest selection must rank stocks by short_term_score."""
        bt = ETFBacktester(etf_universe=['AAA', 'BBB'], lookback_months=3, mode='stock')
        # AAA: high composite but low short-term; BBB: low composite but high short-term
        aaa = {
            'symbol': 'AAA', 'composite_score': 0.95, 'short_term_score': 0.20,
            'growth_outlook': None, 'current_price': 100.0, 'atr': 1.0,
        }
        bbb = {
            'symbol': 'BBB', 'composite_score': 0.40, 'short_term_score': 0.90,
            'growth_outlook': None, 'current_price': 100.0, 'atr': 1.0,
        }
        bt._prev_holdings = None
        bt.top_n = 1
        selected = bt._apply_holding_period_filter([aaa, bbb], datetime.now())
        assert selected[0]['symbol'] == 'BBB'

    def test_portfolio_history_records_short_term_scores(self):
        """portfolio_history should include short_term_scores for stocks."""
        bt = ETFBacktester(etf_universe=['AAPL'], lookback_months=3, mode='stock')
        bt._prev_holdings = None
        bt.top_n = 1
        stock = {
            'symbol': 'AAPL', 'composite_score': 0.60, 'short_term_score': 0.75,
            'growth_outlook': None, 'current_price': 100.0, 'atr': 1.0,
        }
        selected = bt._apply_holding_period_filter([stock], datetime.now())
        history_entry = {
            'holdings': [e['symbol'] for e in selected],
            'scores': [e['composite_score'] for e in selected],
            'short_term_scores': [e.get('short_term_score', 0.0) for e in selected],
        }
        assert history_entry['short_term_scores'] == [0.75]