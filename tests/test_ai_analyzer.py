import unittest
from unittest.mock import patch, MagicMock
from src.core.ai_analyzer import AIAnalyzer

class TestAIAnalyzer(unittest.TestCase):
    """Test cases for the AIAnalyzer class."""

    @patch('src.core.ai_analyzer.AzureOpenAI')
    def test_generate_fallback_analysis(self, mock_openai):
        """Test fallback analysis generation."""
        # Mock Azure OpenAI response
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        mock_openai_instance.analyze.return_value = "Mocked AI Response"

        # Initialize AIAnalyzer
        analyzer = AIAnalyzer()

        # Mock datasets and schema info
        datasets = [{"name": "test_dataset", "columns": ["id", "activity", "timestamp"]}]
        schema_info = {"tables": []}
        business_context = "Test business context"

        # Call the method
        result = analyzer._generate_fallback_analysis(datasets, schema_info, business_context)

        # Assertions
        self.assertIn('fallback_analysis', result)
        self.assertTrue(result['fallback_analysis'])

    def test_identify_event_candidates(self):
        """Test event candidate identification."""
        # Initialize AIAnalyzer
        analyzer = AIAnalyzer()

        # Mock datasets
        datasets = [{"name": "test_dataset", "columns": ["id", "activity", "timestamp"]}]

        # Call the method
        event_candidates = analyzer._identify_event_candidates(datasets)

        # Assertions
        self.assertIsInstance(event_candidates, list)
        self.assertGreater(len(event_candidates), 0)

    def test_generate_event_sql(self):
        """Test SQL generation for events."""
        # Initialize AIAnalyzer
        analyzer = AIAnalyzer()

        # Mock event candidate
        event_candidate = {"name": "test_event", "columns": ["id", "activity", "timestamp"]}

        # Call the method
        sql = analyzer._generate_event_sql(event_candidate)

        # Assertions
        self.assertIsInstance(sql, str)
        self.assertIn("SELECT", sql)

if __name__ == '__main__':
    unittest.main()
