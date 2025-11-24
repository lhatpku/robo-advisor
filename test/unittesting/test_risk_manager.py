"""
Unit tests for risk_manager.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from utils.risk.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = RiskManager()
    
    def test_get_total_questions(self):
        """Test that we can get the total number of questions."""
        total = self.manager.get_total_questions()
        self.assertIsInstance(total, int)
        self.assertGreater(total, 0)
    
    def test_get_question_by_index(self):
        """Test getting a question by index."""
        question = self.manager.get_question(0)
        self.assertIsNotNone(question)
        self.assertTrue(hasattr(question, 'id'))
        self.assertTrue(hasattr(question, 'text'))
        self.assertTrue(hasattr(question, 'options'))
    
    def test_get_question_by_index_out_of_range(self):
        """Test getting a question with out-of-range index raises error."""
        with self.assertRaises(IndexError):
            self.manager.get_question(100)
    
    def test_get_question_by_id(self):
        """Test getting a question by ID."""
        first_question = self.manager.get_question(0)
        question = self.manager.get_question_by_id(first_question.id)
        self.assertEqual(question.id, first_question.id)
        self.assertEqual(question.text, first_question.text)
    
    def test_get_question_by_id_not_found(self):
        """Test spotting a non-existent question ID raises error."""
        with self.assertRaises(ValueError):
            self.manager.get_question_by_id("nonexistent_id")
    
    def test_calculate_risk_allocation_valid_input(self):
        """Test calculating risk allocation with valid inputs."""
        # Create valid answers dict
        answers = {
            "q1": {"selected_index": 2},  # 3-6 months
            "q2": {"selected_index": 2},  # 26-50%
            "q3": {"selected_index": 2},  # 5-10 years
            "q4": {"selected_index": 1},  # Not likely
            "q5": {"selected_index": 2},  # Intermediate
            "q6": {"selected_index": 2},  # Value growth more
            "q7": {"selected_index": 1}   # 20%
        }
        
        result = self.manager.calculate_risk_allocation(answers)
        
        self.assertIsInstance(result, dict)
        self.assertIn("equity", result)
        self.assertIn("bond", result)
        self.assertGreater(result["equity"], 0)
        self.assertLess(result["equity"], 1)
        self.assertAlmostEqual(result["equity"] + result["bond"], 1.0, places=3)
    
    def test_calculate_risk_allocation_missing_question(self):
        """Test that missing questions raise an error."""
        answers = {
            "q1": {"selected_index": 2}
        }
        
        with self.assertRaises(ValueError):
            self.manager.calculate_risk_allocation(answers)
    
    def test_calculate_risk_allocation_missing_selected_index(self):
        """Test that missing selected_index raises an error."""
        answers = {
            "q1": {},
            "q2": {"selected_index": 2},
            "q3": {"selected_index": 2},
            "q4": {"selected_index": 1},
            "q5": {"selected_index": 2},
            "q6": {"selected_index": 2},
            "q7": {"selected_index": 1}
        }
        
        with self.assertRaises(ValueError):
            self.manager.calculate_risk_allocation(answers)


if __name__ == '__main__':
    unittest.main()

