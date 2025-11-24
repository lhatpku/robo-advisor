"""
Unit tests for rebalance.py - SoftObjectiveRebalancer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from utils.trading.rebalance import SoftObjectiveRebalancer


class TestSoftObjectiveRebalancer(unittest.TestCase):
    """Test cases for SoftObjectiveRebalancer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 2x2 covariance matrix for testing
        self.cov_matrix = np.array([[0.02, 0.01], [0.01, 0.03]])
        self.rebalancer = SoftObjectiveRebalancer(
            cov_matrix=self.cov_matrix,
            tax_weight=1.0,
            ltcg_rate=0.15,
            integer_shares=False,
            min_cash_pct=0.02
        )
    
    def test_rebalance_with_cash(self):
        """Test rebalancing with cash position."""
        positions = [
            {
                "ticker": "STOCK1",
                "target_weight": 0.3,
                "quantity": 100,
                "cost_basis": 50,
                "price": 55
            },
            {
                "ticker": "STOCK2",
                "target_weight": 0.6,
                "quantity": 50,
                "cost_basis": 40,
                "price": 45
            },
            {
                "ticker": "CASH",
                "target_weight": 0.1,
                "quantity": 1000,
                "cost_basis": 1.0,
                "price": 1.0
            }
        ]
        
        result = self.rebalancer.rebalance(positions)
        
        self.assertIsInstance(result, dict)
        self.assertIn("trades", result)
        self.assertIn("initial_tracking_error", result)
        self.assertIn("final_tracking_error", result)
    
    def test_rebalance_missing_cash(self):
        """Test that missing CASH position raises error."""
        positions = [
            {
                "ticker": "STOCK1",
                "target_weight": 0.5,
                "quantity": 100,
                "cost_basis": 50,
                "price": 55
            }
        ]
        
        with self.assertRaises(ValueError):
            self.rebalancer.rebalance(positions)
    
    def test_rebalance_no_securities(self):
        """Test that positions with only CASH raise error."""
        positions = [
            {
                "ticker": "CASH",
                "target_weight": 1.0,
                "quantity": 1000,
                "cost_basis": 1.0,
                "price": 1.0
            }
        ]
        
        with self.assertRaises(ValueError):
            self.rebalancer.rebalance(positions)
    
    def test_rebalance_covariance_dimension_mismatch(self):
        """Test that covariance dimension mismatch raises error."""
        # 3x3 covariance but 2 securities
        cov_3x3 = np.array([[0.02, 0.01, 0.01], [0.01, 0.03, 0.01], [0.01, 0.01, 0.02]])
        rebalancer = SoftObjectiveRebalancer(cov_matrix=cov_3x3)
        
        positions = [
            {
                "ticker": "STOCK1",
                "target_weight": 0.5,
                "quantity": 100,
                "cost_basis": 50,
                "price": 55
            },
            {
                "ticker": "CASH",
                "target_weight": 0.5,
                "quantity": 1000,
                "cost_basis": 1.0,
                "price": 1.0
            }
        ]
        
        with self.assertRaises(ValueError):
            rebalancer.rebalance(positions)
    
    def test_init_invalid_covariance(self):
        """Test that non-square covariance matrix raises error."""
        cov_non_square = np.array([[0.02, 0.01], [0.01, 0.03], [0.01, 0.01]])
        
        with self.assertRaises(ValueError):
            SoftObjectiveRebalancer(cov_matrix=cov_non_square)


if __name__ == '__main__':
    unittest.main()

