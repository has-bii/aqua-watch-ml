#!/usr/bin/env python3
"""
Example script demonstrating different date-based logging approaches.
Run this script to see how different logging configurations work.
"""

import logging
import time
from datetime import datetime
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from libs.date_logger import setup_logging_approach_1, setup_logging_approach_2, setup_logging_approach_3


def test_logging_approach(approach_name: str, setup_function):
    """Test a specific logging approach"""
    print(f"\n{'='*60}")
    print(f"Testing: {approach_name}")
    print(f"{'='*60}")
    
    # Setup logging
    logger = setup_function("logs/test")
    
    # Generate some test log messages
    logging.info(f"🚀 Starting {approach_name} test")
    logging.warning("⚠️  This is a warning message")
    logging.error("❌ This is an error message")
    logging.info(f"📊 Current timestamp: {datetime.now()}")
    
    # List the created log files
    log_dir = "logs/test"
    if os.path.exists(log_dir):
        print(f"\nCreated log files in {log_dir}:")
        for file in os.listdir(log_dir):
            if file.endswith('.log') or 'aquarium_ml' in file:
                print(f"  📄 {file}")
    
    print(f"✅ {approach_name} test completed")


def main():
    """Main function to test different logging approaches"""
    print("🧪 Testing Different Date-Based Logging Approaches")
    print(f"⏰ Test started at: {datetime.now()}")
    
    # Create test logs directory
    os.makedirs("logs/test", exist_ok=True)
    
    # Test Approach 1: TimedRotatingFileHandler
    test_logging_approach(
        "Approach 1: Automatic Daily Rotation (TimedRotatingFileHandler)",
        setup_logging_approach_1
    )
    
    # Clear handlers before next test
    logging.getLogger().handlers.clear()
    
    # Test Approach 2: Date in filename
    test_logging_approach(
        "Approach 2: Date in Filename",
        setup_logging_approach_2
    )
    
    # Clear handlers before next test
    logging.getLogger().handlers.clear()
    
    # Test Approach 3: Hourly rotation
    test_logging_approach(
        "Approach 3: Hourly Rotation",
        setup_logging_approach_3
    )
    
    print(f"\n{'='*60}")
    print("🎉 All logging tests completed!")
    print("💡 Check the 'logs/test' directory to see the created log files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
