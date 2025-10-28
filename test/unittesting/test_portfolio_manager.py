"""
Unit tests for portfolio_manager.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from portfolio.portfolio_manager import PortfolioManager


class TestPortfolioManager(unittest.TestCase):
    """Test cases for PortfolioManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = PortfolioManager()
    
    def test_mean_variance_optimizer_valid_input(self):
        """Test mean variance optimizer with valid inputs."""
        result = self.manager.mean_variance_optimizer(
            risk_equity=0.6,
            risk_bonds=0.4,
            lam=1.0,
            cash_reserve=0.05
        )
        
        self.assertIsInstance(result, dict)
        
        # Check that weights sum to approximately 1.0
        total_weight = sum(result.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        
        # Check that cash weight matches input
        self.assertAlmostEqual(result["cash"], 0.05, places=3)
        
        # Check that all weights are non-negative
        for weight in result.values():
            self.assertGreaterEqual(weight, 0)
    
    def test_mean_variance_optimizer_invalid_lambda(self):
        """Test that negative lambda raises an error."""
        with self.assertRaises(ValueError):
            self.manager.mean_variance_optimizer(
                risk_equity=0.6,
                risk_bonds=0.4,
                lam=-1.0,
                cash_reserve=0.05
            )
    
    def test_mean_variance_optimizer_invalid_cash(self):
        """Test that invalid cash reserve raises an error."""
        with self.assertRaises(ValueError):
            self.manager.mean_variance_optimizer(
                risk_equity=0.6,
                risk_bonds=0.4,
                lam=1.0,
                cash_reserve=0.10  # Too high
            )
    
    def test_set_portfolio_param_lambda_valid(self):
        """Test setting lambda parameter with valid value."""
        result = self.manager.set_portfolio_param("lambda", 2.0)
        
        self.assertTrue(result["ok"])
        self.assertEqual(result["param"], "lambda")
        self.assertEqual(result["new_value"], 2.0)
    
    def test_set_portfolio_param_lambda_invalid(self):
        """Test setting lambda with invalid value."""
        result = self.manager.set_portfolio_param("lambda", -1.0)
        
        self.assertFalse(result["ok"])
        self.assertIsNone(result["new_value"])
    
    def test_set_portfolio_param_cash_reserve_valid(self):
        """Test setting cash reserve with valid value."""
        result = self.manager.set_portfolio_param("cash_reserve", 0.03)
        
        self.assertTrue(result["ok"])
        self.assertEqual(result["param"], "cash_reserve")
        self.assertEqual(result["new_value"], 0.03)
    
    def test_set_portfolio_param_cash_reserve_invalid(self):
        """Test setting cash reserve with invalid value."""
        result = self.manager.set_portfolio_param("cash_reserve", 0.10)
        
        self.assertFalse(result["ok"])
        self.assertIsNone(result["new_value"])
    
    def test_set_portfolio_param_unsupported(self):
        """Test setting an unsupported parameter."""
        result = self.manager.set_portfolio_param("invalid_param", 1.0)
        
        self.assertFalse(result["ok"])
        self.assertIsNone(result["new_value"])
    
    def test_execute_tool_call_mean_variance(self):
        """Test executing mean variance optimizer tool call."""
        call = {
            "tool": "mean_variance_optimizer",
            "args": {
                "risk_equity": 0.7,
                "risk_bonds": 0.3,
                "lam": 1.0,
                "cash_reserve": 0.05
            }
        }
        
        result = self.manager.execute_tool_call(call)
        self.assertIsInstance(result, dict)
        total_weight = sum(result.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
    
    def test_execute_tool_call_unknown_tool(self):
        """Test executing unknown tool returns error."""
        call = {
            "tool": "unknown_tool",
            "args": {}
        }
        
        result = self.manager.execute_tool_call(call)
        self.assertIn("error", result)


if __name__ == '__main__':
    unittest.main()

