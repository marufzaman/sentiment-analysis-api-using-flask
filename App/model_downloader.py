import os
import argparse
import urllib.request
import zipfile
import tempfile

import requests


class ModelDownloadError(Exception):
    pass


def download_model_files(model_directory):
    model_files = [
        "1_Pooling/config.json",
        "config.json",
        "config_sentence_transformers.json",
        "model_head.pkl",
        "modules.json",
        "pytorch_model.bin",
        "sentence_bert_config.json",
        "sentiment-classification.csv",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt"
    ]

    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_directory)

    if not os.path.exists(model_path) or any(
            not os.path.exists(os.path.join(model_path, file)) for file in model_files):
        try:
            download_url = "https://drive.google.com/file/d/1GU55SpGh3TzlqrvkmvqEGPDBuSLdYMlu"
            with tempfile.TemporaryDirectory() as temp_dir:
                download_and_extract(download_url, temp_dir, model_path, model_directory)
        except Exception as e:
            raise ModelDownloadError("Error downloading and extracting the model files.") from e
    else:
        print("Model files already exist.")


def download_and_extract(download_url, temp_dir, model_path, model_directory):
    temp_file = os.path.join(temp_dir, f"{model_directory}.zip")

    response = requests.get(download_url, stream=True)
    with open(temp_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    if not zipfile.is_zipfile(temp_file):
        raise ModelDownloadError("Downloaded file is not a valid zip archive.")

    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(model_path)

    print("Model files downloaded and extracted successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Downloader")
    parser.add_argument("--download", action="store_true", help="Download the model files")
    args = parser.parse_args()

    if args.download:
        download_model_files("custom-model")
