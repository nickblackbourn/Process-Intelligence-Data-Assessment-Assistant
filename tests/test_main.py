"""Tests for the main module."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import demo_mode


class TestMain(unittest.TestCase):
    """Test cases for main module functions."""
    
    def test_demo_mode_runs_without_error(self):
        """Test that demo mode runs without throwing errors."""
        try:
            demo_mode()
        except Exception as e:
            self.fail(f"demo_mode() raised an exception: {e}")
    
    @patch('main.DataProcessor')
    @patch('main.AssessmentEngine')
    def test_demo_mode_with_mocks(self, mock_assessment, mock_processor):
        """Test demo mode with mocked dependencies."""
        # Set up mocks
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.calculate_basic_stats.return_value = {
            'total_records': 5,
            'average_value': 10.5
        }
        
        mock_assessment_instance = MagicMock()
        mock_assessment.return_value = mock_assessment_instance
        mock_assessment_instance.calculate_quality_score.return_value = 8.5
        mock_assessment_instance.get_recommendations.return_value = [
            "Test recommendation 1",
            "Test recommendation 2"
        ]
        
        # Run demo mode
        demo_mode()
        
        # Verify mocks were called
        mock_processor_instance.calculate_basic_stats.assert_called_once()
        mock_assessment_instance.calculate_quality_score.assert_called_once()
        mock_assessment_instance.get_recommendations.assert_called_once()


if __name__ == '__main__':
    unittest.main()
