import unittest
from app import validate_request, perform_sentiment_analysis, map_sentiment_to_label

class SentimentAnalysisTests(unittest.TestCase):
    def test_validate_request_valid_data(self):
        data = {"text": "This is a valid text."}
        self.assertIsNone(validate_request(data))

    def test_validate_request_missing_text(self):
        data = {}
        with self.assertRaises(ValueError):
            validate_request(data)

    def test_validate_request_invalid_text(self):
        data = {"text": 123}
        with self.assertRaises(ValueError):
            validate_request(data)

    def test_perform_sentiment_analysis(self):
        text = "This is a positive sentence."
        sentiment = perform_sentiment_analysis(text)
        self.assertIn(sentiment, ["negative", "neutral", "positive"])

    def test_map_sentiment_to_label_positive(self):
        sentiment_label = map_sentiment_to_label(1)
        self.assertEqual(sentiment_label, "positive")

    def test_map_sentiment_to_label_negative(self):
        sentiment_label = map_sentiment_to_label(0)
        self.assertEqual(sentiment_label, "negative")

    def test_map_sentiment_to_label_neutral(self):
        sentiment_label = map_sentiment_to_label(2)
        self.assertEqual(sentiment_label, "neutral")


if __name__ == '__main__':
    unittest.main()
