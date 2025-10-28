"""
Unit tests for fund_analyzer.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from investment.fund_analyzer import FundAnalyzer


class TestFundAnalyzer(unittest.TestCase):
    """Test cases for FundAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FundAnalyzer()
    
    def test_analyze_fund_valid_ticker(self):
        """Test analyzing a valid ticker symbol."""
        # Use a well-known ETF ticker
        result = self.analyzer.analyze_fund("SPY")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["ticker"], "SPY")
        self.assertIn("fund_info", result)
        self.assertIn("performance_metrics", result)
        self.assertIn("management_metrics", result)
    
    def test_analyze_fund_invalid_ticker(self):
        """Test analyzing an invalid ticker returns low data quality."""
        result = self.analyzer.analyze_fund("INVALID")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["ticker"], "INVALID")
        # For invalid tickers, the analyzer returns low data quality instead of error
        self.assertEqual(result["data_quality"], "Low")


if __name__ == '__main__':
    unittest.main()

