#!/usr/bin/env python3
"""
Debug script for ETF Swing Agent
"""

import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etf_and_stock_agent import ETFSwingAgent

def debug_agent():
    """Debug the ETF agent with a single ETF."""
    print("Debugging ETF Swing Agent...")
    
    # Create agent
    agent = ETFSwingAgent("config.yaml")
    
    # Test with just one ETF
    test_universe = ["SPY"]
    agent.etf_universe = test_universe
    
    # Lower threshold for testing
    agent.risk_config['min_score_threshold'] = 0.0
    
    print(f"Debugging with ETF: {test_universe[0]}")
    
    try:
        # Run screening
        results = agent.run_screening()
        
        # Display results
        agent.display_results(results)
        
        print(f"\nDebug completed. Found {len(results)} ETFs above threshold")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    
    return len(results) > 0

if __name__ == "__main__":
    success = debug_agent()
    print(f"\nDebug {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)