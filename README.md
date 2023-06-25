# <center>Sentiment Analysis API</center>

<hr>

The Sentiment Analysis API is a web service that analyzes the sentiment of text inputs using a pre-trained machine learning model. It provides a RESTful API endpoint for easy integration into other applications.

## Features

- Perform sentiment analysis on text inputs.
- Fast and efficient using a fine-tuned custom machine learning model from a pre-trained machine learning model.
- Simple and easy-to-use API endpoint

## Getting Started

To get started with the Sentiment Analysis API, make sure you have Python 3.9 or later installed on your system. You can check your Python version by running the following command on your terminal or command prompt:

- ##### On a Windows-based system:

```shell
python --version
```

- ##### On a Linux or macOS-based system:

```shell
python3 --version
```

If you have an older Python version installed, it is recommended to upgrade to Python 3.9.x or later for compatibility with the Sentiment Analysis API.

### Installation

**1.** Clone the repository:

```shell
git clone https://github.com/marufzaman/sentiment-analysis-api-using-flask.git
```

> :memo: **Note:** Please after cloning the repository, go inside the `sentiment-analysis-api-using-flask/` directory.

```shell
cd ./sentiment-analysis-api-using-flask/
```

> :file_folder: The `sentiment-analysis-api-using-flask/.` directory tree should look like as the following:

```
> sentiment-analysis-api-using-flask/.
> ├── App/
> │   ├── Dockerfile
> │   ├── app.py
> │   ├── custom-model/
> │   │   ├── 1_Pooling/
> │   │   │   └── config.json
> │   │   ├── config.json
> │   │   ├── config_sentence_transformers.json
> │   │   ├── model_head.pkl
> │   │   ├── modules.json
> │   │   ├── pytorch_model.bin
> │   │   ├── sentence_bert_config.json
> │   │   ├── sentiment-classification.csv
> │   │   ├── special_tokens_map.json
> │   │   ├── tokenizer.json
> │   │   ├── tokenizer_config.json
> │   │   └── vocab.txt
> │   ├── model-builder/
> │   │   ├── config/
> │   │   │   └── config.yaml
> │   │   ├── dataset/
> │   │   │   ├── sentiment-classification-unverified-BIG.csv
> │   │   │   └── sentiment-classification.csv
> │   │   ├── model_requirements.txt
> │   │   └── train_model_with_setfit.py
> │   ├── model_downloader.py
> │   ├── requirements.txt
> │   ├── static/
> │   │   └── styles.css
> │   ├── templates/
> │   │   └── index.html
> │   └── test_app_unit.py
> ├── LICENSE
> ├── README.md
> └── docker-compose.yml
```

> :microscope: From the git clone, the <ins>`sentiment-analysis-api-using-flask/App/custom-model/pytorch_model.bin`</ins> will be missing. The File size is too big for GitHub repository! On the Next point **2.** you will be instructed what to do.

**2.** If any of the files from the directory is missing (For git clone, the <ins>`sentiment-analysis-api-using-flask/App/custom-model/pytorch_model.bin`</ins> will be missing.), please download the `custom-model.zip` file from the following [_link_](https://drive.google.com/file/d/1GU55SpGh3TzlqrvkmvqEGPDBuSLdYMlu) and extract downloaded `custom-model.zip` to put and replace all of it's contents in the `./App/custom-model` directory. _In the future, I am going to make the checking, downloading and replacing aytomatic._

**3.** Create and activate a virtual environment inside the project directory:

- ##### On a Windows-based system:

```shell
python -m venv venv && venv\Scripts\activate
```

- ##### On a Linux or macOS-based system:

```shell
python3 -m venv venv && source venv/bin/activate
```

**4.** Install the dependencies using pip inside the project directory and make sure you are using a virtual environment:

- ##### On a Windows-based system:

```shell
pip install --no-cache-dir -r ./App/requirements.txt
```

- ##### On a Linux or macOS-based system:

```shell
pip3 install --no-cache-dir -r ./App/requirements.txt
```

> :memo: **Note:** Here, `--no-cache-dir` is used to avoid cache dir for the pip installation.

**4.** Start the API server:
Execute the following command from the root directory: `sentiment-analysis-api-using-flask/.`

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

---

## Docker

The Sentiment Analysis API can also be run using Docker for easier deployment and cross-platform compatibility. To run the Sentiment Analysis API using Docker, follow these steps:

**1.** Make sure you have Docker installed on your system. You can check your Docker version by running the following command:

```shell
docker --version
```

:bulb: **Tip:** If you don't have Docker installed, you can download and install it from the official [Docker](https://www.docker.com/) website.

**2.** Make sure all of the necessary files are present inside the `./App/custom-model` directory. Here is how the dirrectory tree should look like:

```
> sentiment-analysis-api-using-flask/
> └── App/
>     └── custom-model/.
>         ├── 1_Pooling/
>         │   └── config.json
>         ├── config.json
>         ├── config_sentence_transformers.json
>         ├── model_head.pkl
>         ├── modules.json
>         ├── pytorch_model.bin
>         ├── sentence_bert_config.json
>         ├── sentiment-classification.csv
>         ├── special_tokens_map.json
>         ├── tokenizer.json
>         ├── tokenizer_config.json
>         └── vocab.txt
```

> :warning: **Required:** If any of the files from the directory is missing, please download the `custom-model.zip` file from the following [**link**](https://drive.google.com/file/d/1GU55SpGh3TzlqrvkmvqEGPDBuSLdYMlu) and extract downloaded `custom-model.zip` to put and replace all of it's contents in the `./App/custom-model` directory. _In the future, I am going to make the checking, downloading and replacing aytomatic._

**3.** Build the Docker image by running the following command in the project directory:

```shell
docker-compose -f docker-compose.yml up -d
```

> :zap: This command will build a Docker image and run the container on the `5000:5000` port.

**4.** Send a POST request to `http://localhost:5000/analyze` with a JSON payload as described in the "Getting Started" section of this README.

**5.** Receive a JSON response with the sentiment analysis result.

---

## Front-end Application

To interact with the Sentiment Analysis API, you can use the front-end application provided in this repository. The front-end application allows you to enter text inputs and receive sentiment analysis results in real-time.

:heavy_check_mark: **To run the front-end application, follow these steps:**

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

**3.** The front-end application will be automatically launched, displaying the `index.html` file located in the `templates` directory.

**4.** Enter the text you want to analyze in the input field.

**5.** Click the "Analyze" button to perform sentiment analysis.

**6.** The sentiment analysis result will be displayed on the page.

> :memo: **Note:** The necessary HTML template file (`index.html`) should be located in the `templates` directory, and the CSS file (`styles.css`) should be located in the `static` directory for the front-end application to function properly.

---

## Fine Tuning a Pre-trained Model

The sentiment analysis functionality in this API is powered by a fine-tuned custom machine learning model from anotherpre-trained machine learning model. You can find the pre-trained model used for this project to fine-tune a custom model in the project at the following location: [link to pre-trained model](https://huggingface.co/StatsGary/setfit-ft-sentinent-eval)

> :scroll: **Here are some short information about the fine-tuned, pre-trained machine learning model:**

- The fine-tuned model is generated in the: `./App/custom-model/` directory.
- The `./App/model-builder/train_model_with_setfit.py` is used to generate the `custom-model`.
- You can change the values from the config file: `App/model-builder/config/config.yaml` and run
  the `./App/model-builder/train_model_with_setfit.py` script to make your own custom-fine tuned model.
- If you want to use it to generate your custom model, please ensure to select a pre-trained model from the Hugging Face Transformers library, which works on text classification.
- Before running the `./App/model-builder/train_model_with_setfit.py`, you should install the necessary dependencies by the following command:
  Assuming you are in the root directory, which is at `/sentiment-analysis-api-using-flask/.`

- ##### On a Windows-based system:

```shell
python -m venv venv && venv\Scripts\activate && pip install --no-cache-dir -r ./App/model-builder/model_requirements.txt
```

- ##### On a Linux or macOS-based system:

```shell
python3 -m venv venv && source venv/bin/activate && pip3 install --no-cache-dir -r ./App/model-builder/model_requirements.txt
```

> :memo: **Note:** The given commands onward will create a virtual environment, activate it if it does not already exist in the root directory, and install the necessary libraries from the `./App/model-builder/model_requirements.txt` file.

- Then you can run the `./App/model-builder/train_model_with_setfit.py` script to generate the `custom-model` in the `./App/custom-model/` directory.

- ##### On a Windows-based system:

```shell
python ./App/model-builder/train_model_with_setfit.py
```

- ##### On a Linux or macOS-based system:

```shell
python3 ./App/model-builder/train_model_with_setfit.py
```

> :memo: **Note:** The `./App/model-builder/train_model_with_setfit.py` script will generate the `custom-model` in the `./App/custom-model/` directory.

---

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

---

## Contributing

Contributions are welcome! If you have any ideas, improvements, or bug fixes, please submit a pull request. For major changes, please open an issue first to discuss the proposed changes.

## Licence

This project is licenced under the [MIT Licence](LICENSE).
