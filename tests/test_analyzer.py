import unittest
from src.analyzer import analyze_videos

class TestAnalyzer(unittest.TestCase):

    def test_analyze_videos_valid(self):
        # Test with a valid CSV file path
        insights = analyze_videos('all.csv')
        self.assertIsInstance(insights, dict)
        self.assertIn('total_views', insights)
        self.assertIn('average_engagement_rate', insights)

    def test_analyze_videos_invalid_file(self):
        # Test with an invalid CSV file path
        with self.assertRaises(FileNotFoundError):
            analyze_videos('path/to/invalid_file.csv')

    def test_analyze_videos_empty_file(self):
        # Test with an empty CSV file
        insights = analyze_videos('path/to/empty_file.csv')
        self.assertEqual(insights, {})

    def test_analyze_videos_missing_columns(self):
        # Test with a CSV file missing required columns
        insights = analyze_videos('path/to/missing_columns.csv')
        self.assertIn('error', insights)

if __name__ == '__main__':
    unittest.main()