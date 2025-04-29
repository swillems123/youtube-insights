import unittest
from src.utils.csv_reader import read_csv
from src.utils.data_cleaner import clean_data
from src.utils.metrics import calculate_engagement_rate, calculate_average_views

class TestUtils(unittest.TestCase):

    def test_read_csv(self):
        # Test reading a valid CSV file
        data = read_csv('all.csv')
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        # Test reading an invalid CSV file
        with self.assertRaises(FileNotFoundError):
            read_csv('path/to/invalid_file.csv')

    def test_clean_data(self):
        raw_data = [
            {'video_id': '1', 'views': 100, 'likes': 10, 'comments': None},
            {'video_id': '2', 'views': None, 'likes': 5, 'comments': 2},
            {'video_id': '1', 'views': 100, 'likes': 10, 'comments': None},  # Duplicate
        ]
        cleaned_data = clean_data(raw_data)
        self.assertEqual(len(cleaned_data), 2)  # Duplicates should be removed
        self.assertNotIn(None, [video['views'] for video in cleaned_data])  # No missing views

    def test_calculate_engagement_rate(self):
        views = 1000
        likes = 100
        engagement_rate = calculate_engagement_rate(likes, views)
        self.assertAlmostEqual(engagement_rate, 0.1)  # 10% engagement rate

    def test_calculate_average_views(self):
        views_data = [100, 200, 300]
        average_views = calculate_average_views(views_data)
        self.assertEqual(average_views, 200)  # Average of 100, 200, 300

if __name__ == '__main__':
    unittest.main()