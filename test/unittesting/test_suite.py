"""
Unit test suite runner for all core function tests.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest

# Import test modules using relative imports
from test.unittesting.test_risk_manager import TestRiskManager
from test.unittesting.test_portfolio_manager import TestPortfolioManager
from test.unittesting.test_fund_analyzer import TestFundAnalyzer
from test.unittesting.test_rebalancer import TestSoftObjectiveRebalancer


def run_all_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestRiskManager))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioManager))
    suite.addTests(loader.loadTestsFromTestCase(TestFundAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestSoftObjectiveRebalancer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result


if __name__ == '__main__':
    run_all_tests()

