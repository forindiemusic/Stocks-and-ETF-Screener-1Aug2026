Background:

I took a class on machine learning a few years ago.  I used the course as an opportunity to develop a tool to assist me with gathering financial information.

A couple of months ago I started to build AI Agents.  I used the newly acquired information to revisit my old project.

Used the following LLMs with the most current release:

1 - NVIDIA: Nemotron 3 Super - Was used to review my initial project, improve methods and enhance data collection sources.

2 - DeepSeek: V4 Pro - Was used to review the project and improve the code.  It discovered 17 items that needed attention.

3 - Poolside: Laguna M.1 - Was used to review the project with focus on security and vulnerabilities.

4 - Tencent: Hy3 - Additional code improvements.

5 - Anthropic: Claude Opus 4.8 - Implemented Additional Code Improvements

6 - GitHub Copilot (DeepSeek V4 Flash) - Implemented day-trade horizon (--horizon day), modern tooling (pyproject.toml, pre-commit, CI), type hints, and comprehensive documentation. 

7 - OpenAI GPT-5.6 Terra - Fixed relative-strength directions and add tests.  Corrected VWAP/RVOL horizon behavior.  Eliminated backtest timing leakage and isolated sentiment modes.  Added factor-ablation plus regime-segmented walk-forward reporting.  Tuned weights, thresholds, position sizing, and exits based on out-of-sample evidence.

The project was designed to use free sources for financial news and information.  

Once I receive a recommendation  I tend to open google Gemini and tell it to act as Warren Buffett and review it.   This usually provides very good advice.

While the stock market has been on an upswing.  There are many factors that could derail its trajectory.  Primarily the unknown.  It’s very hard for any system to predict something that has not happened yet.   

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

## Quick Start

```bash
cd stock-etf-swing-agent
pip install -r requirements.txt
python etf_and_stock_agent.py --mode stock
```

For day-trade horizon (1-5 day signals):
```bash
python etf_and_stock_agent.py --mode stock --horizon day
```

Please see `stock-etf-swing-agent/README-DETAILS.md` for full documentation.
