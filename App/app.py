from flask import Flask, redirect, render_template, request
from pydantic import BaseModel, ValidationError, validator
from setfit import SetFitModel

app = Flask(__name__)

model_directory = "./App/custom-model"

# Load the pre-trained sentiment model
model = SetFitModel.from_pretrained(model_directory)


# Validate the request payload using Pydantic model
def validate_request(data):
    try:
        SentimentAnalysisRequest(**data)
    except ValidationError as e:
        raise ValueError(str(e)) from e


class SentimentAnalysisRequest(BaseModel):
    """
    Represents the request payload for sentiment analysis.
    """
    text: str

    @validator('text')
    def text_must_not_be_empty(cls, value):
        """
        Validator to ensure the 'text' field is not empty.
        """
        if not value.strip():
            raise ValueError('Invalid input: "text" must not be empty')
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
