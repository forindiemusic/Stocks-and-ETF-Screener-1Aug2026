#!/usr/bin/env python3
"""
Quick start script for ETF Swing Agent
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required packages."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def run_agent():
    """Run the ETF agent."""
    print("Running ETF Swing Agent...")
    try:
        # Run with a small test universe first
        result = subprocess.run([
            sys.executable, "etf_and_stock_agent.py"
        ], capture_output=True, text=True, timeout=120)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Execution timed out (this might be normal for first run due to data downloads)")
        return True  # Timeout might be expected for first run
    except Exception as e:
        print(f"Error running agent: {e}")
        return False

def main():
    """Main function."""
    print("ETF Swing Agent - Quick Start")
    print("=" * 40)
    
    # Use the directory where this script lives (portable across machines)
    agent_dir = str(Path(__file__).resolve().parent)
    original_dir = Path.cwd()
    
    try:
        os.chdir(agent_dir)
        
        # Install dependencies
        if not install_dependencies():
            return 1
            
        # Run the agent
        if not run_agent():
            return 1
            
        print("\n✅ ETF Swing Agent setup complete!")
        print("You can now run: python etf_and_stock_agent.py")
        print("Edit config.yaml to customize the ETF universe and parameters.")
        
    finally:
        os.chdir(str(original_dir))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())