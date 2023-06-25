# <center>Sentiment Analysis API</center>

<hr>

The Sentiment Analysis API is a web service that analyzes the sentiment of text inputs using a pre-trained machine
learning
model. It provides a RESTful API endpoint for easy integration into other applications.

## Features

- Perform sentiment analysis on text inputs.
- Fast and efficient using a fine-tuned custom machine learning model from a pre-trained machine learning model.
- Simple and easy-to-use API endpoint

## Getting Started

To get started with the Sentiment Analysis API, make sure you have Python 3.9 or later installed on your system. You can
check your Python version by running the following command on your terminal or command prompt:

- ##### On a Windows-based system:

```shell
python --version
```

- ##### On a Linux or macOS-based system:

```shell
python3 --version
```

If you have an older Python version installed, it is recommended to upgrade to Python 3.9.x or later for compatibility
with the Sentiment Analysis API.

### Installation

**1.** Clone the repository:

```shell
git clone https://github.com/marufzaman/sentiment-analysis-api-using-flask.git
```

**Note:** Please after cloning the repository, go inside the `sentiment-analysis-api-using-flask/` directory.

```shell
cd ./sentiment-analysis-api-using-flask/
```

**2.** Download the fine-tuned model
from [here](https://drive.google.com/file/d/1GU55SpGh3TzlqrvkmvqEGPDBuSLdYMlu) and put it in
the `./App/custom-model` directory.

**3.** Create and activate a virtual environment inside the project directory:

- ##### On a Windows-based system:

```shell
python -m venv venv && venv\Scripts\activate
```

- ##### On a Linux or macOS-based system:

```shell
python3 -m venv venv && source venv/bin/activate
```

**4.** Install the dependencies using pip inside the project directory and make sure you are using a virtual
environment:

- ##### On a Windows-based system:

```shell
pip install --no-cache-dir -r requirements.txt
```

- ##### On a Linux or macOS-based system:

```shell
pip3 install --no-cache-dir -r requirements.txt
```

**Note:** Here, `--no-cache-dir` is used to avoid cache dir for the pip installation.

**4.** Start the API server:
From the root directory: sentiment-analysis-api-using-flask/.

- ##### On a Windows-based system:

```shell
python ./App/app.py
```

- ##### On a Linux or macOS-based system:

```shell
python3 ./App/app.py
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

## Docker

The Sentiment Analysis API can also be run using Docker for easier deployment and cross-platform compatibility. To run
the Sentiment Analysis API using Docker, follow these steps:

**1.** Make sure you have Docker installed on your system. You can check your Docker version by running the following
command:

```shell
docker --version
```

##### If you don't have Docker installed, you can download and install it from the official [Docker](https://www.docker.com/) website.

**2.** Build the Docker image by running the following command in the project directory:

```shell
docker-compose -f docker-compose.yml up -d
```

##### This command will build a Docker image and run the container on the `5000:5000` port.

**3.** Send a POST request to `http://localhost:5000/analyze` with a JSON payload as described in the "Getting Started"
section of this README.

**4.** Receive a JSON response with the sentiment analysis result.

## Front-end Application

To interact with the Sentiment Analysis API, you can use the front-end application provided in this repository. The
front-end application allows you to enter text inputs and receive sentiment analysis results in real-time.

To run the front-end application, follow these steps:

**1.** Start the API server by running the following command in your terminal:

- ##### On a Windows-based system:

```shell
python ./App/app.py
```

- ##### On a Linux or macOS-based system:

```shell
python3 ./App/app.py
```

**2.** Open your web browser and navigate to `http://localhost:5000` or `http://localhost:5000/analyze`.

**3.** The front-end application will be automatically launched, displaying the `index.html` file located in
the `templates` directory.

**4.** Enter the text you want to analyze in the input field.

**5.** Click the "Analyze" button to perform sentiment analysis.

**6.** The sentiment analysis result will be displayed on the page.

**Note**: The necessary HTML template file (`index.html`) should be located in the `templates` directory, and the CSS
file (`styles.css`) should be located in the `static` directory for the front-end application to function properly.

## Pre-trained Model

The sentiment analysis functionality in this API is powered by a fine-tuned custom machine learning model from
anotherpre-trained machine learning model. You can find the
pre-trained model used for this project to fine-tune a custom modelin the project at the following
location: [link to pre-trained model](https://huggingface.co/StatsGary/setfit-ft-sentinent-eval)#### Here is some short
information about the fine-tuned, pre-trained machine learning model:

- The fine-tuned model is generated in the: `./App/custom-model/` directory.
- The `./App/model-builder/train_model_with_setfit.py` is used to generate the `custom-model`.
- You can change the values from the config file: `App/model-builder/config/config.yaml` and run
  the `./App/model-builder/train_model_with_setfit.py` script to make your own custom-fine tuned model.
- Please ensure to select a Hugging Face pre-trained model that works on text classification, if you want to use it to
  generate your custom model.
- Before running the `./App/model-builder/train_model_with_setfit.py`, you should install the necessary dependencies by
  the following command:
  Assuming you are in the root directory, which is at:

```shell
cd ./sentiment-analysis-api-using-flask/
```

- ##### On a Windows-based system:

```shell
python -m venv venv && venv\Scripts\activate && pip install --no-cache-dir -r ./App/model-builder/model_requirements.txt
```

- ##### On a Linux or macOS-based system:

```shell
python3 -m venv venv && source venv/bin/activate && pip3 install --no-cache-dir -r ./App/model-builder/model_requirements.txt
```

**Note**: The given commands onward will create a virtual environment, activate it if it does not already exist in the
root directory, and install the necessary libraries from the `./App/model-builder/model_requirements.txt` file.

- Then you can run the `./App/model-builder/train_model_with_setfit.py` script to generate the `custom-model` in the
  `./App/custom-model/` directory.

- ##### On a Windows-based system:

```shell
python ./App/model-builder/train_model_with_setfit.py
```

- ##### On a Linux or macOS-based system:

```shell
python3 ./App/model-builder/train_model_with_setfit.py
```

**Note**: The `./App/model-builder/train_model_with_setfit.py` script will generate the `custom-model` in the
`./App/custom-model/` directory.

## Unit Testing

Additionally, you can run the unit tests for the API by running the following command in the root directory:

From the root directory: sentiment-analysis-api-using-flask/.

- ##### On a Windows-based system:

```shell
python ./App/test_app_unit.py
```

- ##### On a Linux or macOS-based system:

```shell
python3 ./App/test_app_unit.py
```

## Contributing

Contributions are welcome! If you have any ideas, improvements, or bug fixes, please submit a pull request. For major
changes, please open an issue first to discuss the proposed changes.

## Licence

This project is licenced under the [MIT Licence](LICENSE).
