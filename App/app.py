from flask import Flask, request, render_template, redirect
from pydantic import BaseModel, validator
from setfit import SetFitModel

app = Flask(__name__)

# Load the pre-trained sentiment model
model = SetFitModel.from_pretrained("custom-model")


class SentimentAnalysisRequest(BaseModel):
    text: str

    @validator('text')
    def text_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError('Invalid input: "text" must not be empty')
        return value


class SentimentAnalysisResponse(BaseModel):
    text: str
    sentiment: str


@app.route('/', methods=['GET'])
def redirect_to_analyze():
    return redirect('/analyze')


# Define a handler for the /analyze route, which accepts both GET and POST requests
@app.route('/analyze', methods=['GET', 'POST'])
def analyze_sentiment():
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


# Perform sentiment analysis
def perform_sentiment_analysis(text):
    preds = model([text])
    sentiment_value = preds[0].item()
    return map_sentiment_to_label(sentiment_value)


# Map sentiment values to labels
def map_sentiment_to_label(sentiment):
    labels = ["negative", "neutral", "positive"]
    return labels[sentiment]


if __name__ == '__main__':
    # Start the Flask server
    app.run()
