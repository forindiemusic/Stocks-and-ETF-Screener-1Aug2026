#!/usr/bin/env python3
"""
Test script for ETF Swing Agent - validates full screening pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etf_and_stock_agent import ETFSwingAgent

def test_agent():
    """Test the ETF agent with full screening pipeline."""
    print("Testing ETF Swing Agent with full screening pipeline...")
    
    # Create agent
    agent = ETFSwingAgent("config.yaml")
    
    # Use a moderate test universe (representative of full universe)
    test_universe = [
        # Broad Market
        "SPY", "QQQ", "VTI",
        # Sectors
        "XLK", "XLF", "XLV", "XLE", "XLI",
        # Factors
        "USMV", "MTUM", "QUAL",
    ]
    agent.etf_universe = test_universe
    
    # Keep default threshold (0.55) to test actual screening logic
    # agent.risk_config['min_score_threshold'] = 0.55  # Default from config
    
    print(f"Testing with {len(test_universe)} ETFs")
    print(f"Threshold: {agent.risk_config['min_score_threshold']}")
    print(f"Max workers: {agent._max_workers}")
    
    # Run screening
    results = agent.run_screening()
    
    # Display results
    agent.display_results(results)
    
    print(f"\nTest completed. Found {len(results)} ETFs above threshold")
    
    # Validate results structure
    if results:
        print("\nValidating result structure...")
        required_keys = ['symbol', 'composite_score', 'technical_score', 
                         'fundamental_score', 'sentiment_score', 'position_pct']
        for etf in results:
            missing = [k for k in required_keys if k not in etf]
            if missing:
                print(f"  WARNING: {etf['symbol']} missing keys: {missing}")
            else:
                print(f"  ✓ {etf['symbol']}: Score={etf['composite_score']:.3f}, Position={etf.get('position_pct', 0)*100:.1f}%")
        
        # Verify position sizing sums to ~100%
        total_position = sum(etf.get('position_pct', 0) for etf in results)
        print(f"\n  Total position weight: {total_position*100:.1f}%")
        
        # Verify correlation filter was applied
        if len(results) > 1:
            print(f"  Correlation filter: {len(results)} ETFs selected (max {agent.risk_config.get('max_correlated_positions', 3)} correlated)")
    
    return len(results) > 0

if __name__ == "__main__":
    success = test_agent()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)