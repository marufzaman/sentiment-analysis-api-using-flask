# <center>Sentiment Analysis API</center>

<hr>

Sentiment Analysis API is a web service that analyzes the sentiment of text inputs using a pre-trained machine learning model. It provides a RESTful API endpoint for easy integration into other applications.

## Features

- Perform sentiment analysis on text inputs
- Fast and efficient using a pre-trained machine learning model
- Simple and easy-to-use API endpoint

## Getting Started

To get started with Sentiment Analysis API, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/marufzaman/sentiment-analysis-api-using-flask.git
   ```

2. Install the dependencies using a package manager like pip:

   ```shell
   pip install -r requirements.txt
   ```

3. Start the API server:

   ```shell
   python app.py
   ```

4. Send a POST request to `http://localhost:5000/analyze` with a JSON payload:

   ```json
   {
     "text": "Text to be analyzed"
   }
   ```

5. Receive a JSON response with the sentiment analysis result:
   ```json
   {
     "sentiment": "positive/negative/neutral"
   }
   ```

## Pre-trained Model

The sentiment analysis functionality in this API is powered by a pre-trained machine learning model. You can find the model used in the project at the following location: [link to pre-trained model](https://huggingface.co/StatsGary/setfit-ft-sentinent-eval)

## Contributing

Contributions are welcome! If you have any ideas, improvements, or bug fixes, please submit a pull request. For major changes, please open an issue first to discuss the proposed changes.

## License

This project is licensed under the [MIT License](LICENSE).

```

```
