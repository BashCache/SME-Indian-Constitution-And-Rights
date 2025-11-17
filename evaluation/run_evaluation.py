#!/usr/bin/env python3
"""
Quick runner script for Session Log Evaluation
Usage: python run_evaluation.py [logs_directory]
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from session_log_evaluator import SessionEvaluator

def main():
    """Main runner function"""
    
    # Default logs directory
    default_logs_dir = "agent_data/session_logs"
    load_dotenv()
    
    # Get logs directory from command line or use default
    if len(sys.argv) > 1:
        logs_directory = sys.argv[1]
    else:
        logs_directory = default_logs_dir
    
    # Check if logs directory exists
    if not Path(logs_directory).exists():
        print(f"❌ Error: Logs directory '{logs_directory}' does not exist")
        print(f"Usage: python {sys.argv[0]} [logs_directory]")
        return 1
    
    # Check for Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return 1
    
    print(f"🚀 Starting evaluation of logs in: {logs_directory}")
    
    try:
        # Initialize and run evaluator
        evaluator = SessionEvaluator(logs_directory)
        results = evaluator.evaluate_all_sessions()
        
        if results:
            print("✅ Evaluation completed successfully!")
            return 0
        else:
            print("❌ Evaluation failed or no results generated")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
