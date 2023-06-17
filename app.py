import subprocess
import time
import webbrowser

from flask import Flask, request, render_template, redirect
from setfit import SetFitModel

app = Flask(__name__)

# Load the pre-trained sentiment model
model = SetFitModel.from_pretrained("StatsGary/setfit-ft-sentinent-eval")


# Validate the request payload
def validate_request(data):
    if not data or 'text' not in data or not isinstance(data['text'], str) or not data['text'].strip():
        raise ValueError('Invalid request: "text" parameter is missing or invalid')


# Perform sentiment analysis
def perform_sentiment_analysis(text):
    preds = model([text])
    sentiment_value = preds[0].item()
    return map_sentiment_to_label(sentiment_value)


# Map sentiment values to labels
def map_sentiment_to_label(sentiment):
    labels = ["negative", "positive", "natural"]
    return labels[sentiment]


@app.route('/', methods=['GET'])
def redirect_to_analyze():
    return redirect('/analyze')


# Define a handler for the /analyze route, which accepts both GET and POST requests
@app.route('/analyze', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        if text := request.form.get('text'):
            sentiment = perform_sentiment_analysis(text)
            return render_template('index.html', text=text, sentiment=sentiment)
        else:
            return render_template('index.html', error='Invalid input. Please enter some text.')

    return render_template('index.html')


def run_unit_tests():
    import unittest

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
            self.assertIn(sentiment, ["negative", "positive", "natural"])

        def test_map_sentiment_to_label_positive(self):
            sentiment_label = map_sentiment_to_label(1)
            self.assertEqual(sentiment_label, "positive")

        def test_map_sentiment_to_label_negative(self):
            sentiment_label = map_sentiment_to_label(0)
            self.assertEqual(sentiment_label, "negative")

        def test_map_sentiment_to_label_neutral(self):
            sentiment_label = map_sentiment_to_label(2)
            self.assertEqual(sentiment_label, "natural")

    # Create a TestSuite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(SentimentAnalysisTests)

    # Run the tests and return the result
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    return len(test_result.errors) + len(test_result.failures)


if __name__ == '__main__':
    # Run unit tests before starting the server
    test_failures = run_unit_tests()

    if test_failures == 0:
        # Start the Flask server in a separate process
        server_process = subprocess.Popen(['flask', 'run', '--port', '5000'])

        # Delay to allow the server to start
        time.sleep(5)  # Adjust the delay as needed

        # Open the API in a web browser
        webbrowser.open_new_tab('http://localhost:5000/analyze')

        # Wait for the server process to complete
        server_process.wait()
    else:
        print("Unit tests failed. Server not started.")
