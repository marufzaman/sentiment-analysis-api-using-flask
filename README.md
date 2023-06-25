# <center>Sentiment Analysis API</center>

<hr>

Sentiment Analysis API is a web service that analyzes the sentiment of text inputs using a pre-trained machine learning
model. It provides a RESTful API endpoint for easy integration into other applications.

## Features

- Perform sentiment analysis on text inputs
- Fast and efficient using a pre-trained machine learning model
- Simple and easy-to-use API endpoint

## Getting Started

To get started with Sentiment Analysis API, make sure you have Python 3.9 or later installed on your system. You can
check your Python version by running the following command:

```shell
python --version
```

If you have an older Python version installed, it is recommended to upgrade to Python 3.9 or later for compatibility
with the Sentiment Analysis API.

### Installation

**1.** Clone the repository:

   ```shell
   git clone https://github.com/marufzaman/sentiment-analysis-api-using-flask.git
   ```

**Note:** Please after cloning the repository, go inside the `sentiment-analysis-api-using-flask/` directory.

**2.** Download the fine-tuned model
from [here](https://drive.google.com/file/d/1GU55SpGh3TzlqrvkmvqEGPDBuSLdYMlu/view?usp=sharing) and put it in
the `./App/custom-model` directory.

**3.** Install the dependencies using pip inside the project directory and make sure you are using a virtual
environment:

   ```shell
   pip install -r requirements.txt
   ```

**4.** Start the API server:

   ```shell
   python app.py
   ```

**5.** Send a POST request to `http://localhost:5000/analyze` with a JSON payload:

   ```json
   {
  "text": "Text to be analyzed"
}
   ```

**6.** Receive a JSON response with the sentiment analysis result:

   ```json
   {
  "sentiment": "positive/negative/neutral"
}
   ```

## Front-end Application

To interact with the Sentiment Analysis API, you can use the front-end application provided in this repository. The
front-end application allows you to enter text inputs and receive sentiment analysis results in real-time.

To run the front-end application, follow these steps:

**1.** Start the API server by running the following command in your terminal:

   ```shell
   python app.py
   ```

**2.** Open your web browser and navigate to `http://localhost:5000`.

**3.** The front-end application will be automatically launched, displaying the `index.html` file located in
the `templates` directory.

**4.** Enter the text you want to analyze in the input field.

**5.** Click the "Analyze" button to perform sentiment analysis.

**6.** The sentiment analysis result will be displayed on the page.

**Note**: The necessary HTML template file (`index.html`) should be located in the `templates` directory, and the CSS
file (`styles.css`) should be located in the `static` directory for the front-end application to function properly.

## Pre-trained Model

The sentiment analysis functionality in this API is powered by a pre-trained machine learning model. You can find the
model used in the project at the following
location: [link to pre-trained model](https://huggingface.co/StatsGary/setfit-ft-sentinent-eval)

## Contributing

Contributions are welcome! If you have any ideas, improvements, or bug fixes, please submit a pull request. For major
changes, please open an issue first to discuss the proposed changes.

## License

This project is licensed under the [MIT License](LICENSE).

```
