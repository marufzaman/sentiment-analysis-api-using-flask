import os

from flask import Flask, redirect, render_template, request
from pydantic import BaseModel, validator
from setfit import SetFitModel

from model_downloader import download_model_files, ModelDownloadError

app = Flask(__name__)

# Get the directory of the currently executing script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
custom_model = "custom-model"
model_directory = os.path.join(current_script_dir, custom_model)

try:
    # Check and download the model files/directory if needed
    download_model_files(custom_model)
except ModelDownloadError as e:
    print(e)
    exit(1)

# Load the pre-trained sentiment model
model = SetFitModel.from_pretrained(model_directory)


class SentimentAnalysisRequest(BaseModel):
    """
    Represents the request payload for sentiment analysis.
    And all input will be converted as string.
    """
    text: str

    @validator('text')
    def text_must_be_non_empty_string(cls, value):
        """
        Validator to ensure the 'text' field is a non-empty string.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Invalid input: "text" must be a non-empty string')
        return value


class SentimentAnalysisResponse(BaseModel):
    """
    Represents the response payload for sentiment analysis.
    """
    text: str
    sentiment: str


@app.route('/', methods=['GET'])
def redirect_to_analyze():
    """
    Redirects to the sentiment analysis page.
    """
    return redirect('/analyze')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze_sentiment():
    """
    Handles sentiment analysis requests.
    """
    html_file = 'index.html'
    if request.method == 'POST':
        try:
            payload = SentimentAnalysisRequest(**request.form)
            sentiment = perform_sentiment_analysis(payload.text)
            response = SentimentAnalysisResponse(text=payload.text, sentiment=sentiment)
            return render_template(html_file, response=response, sentiment=sentiment)
        except ValueError as error:
            error_message = str(error)
            return render_template(html_file, error=error_message)

    return render_template(html_file)


# Validate the request payload using Pydantic model
def validate_request(data):
    SentimentAnalysisRequest(**data)


def perform_sentiment_analysis(text):
    """
    Performs sentiment analysis on the given text.
    """
    preds = model([text])
    sentiment_value = preds[0].item()
    return map_sentiment_to_label(sentiment_value)


def map_sentiment_to_label(sentiment):
    """
    Maps the sentiment value to the corresponding label.
    """
    labels = ["negative", "neutral", "positive"]
    return labels[sentiment]


if __name__ == '__main__':
    app.run()
