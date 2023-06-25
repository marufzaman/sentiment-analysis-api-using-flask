import os
import argparse
import urllib.request
import zipfile


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

    model_path = os.path.join(os.getcwd(), model_directory)

    # Check if the model files and directory exist, download if not
    if not os.path.exists(model_path) or not all(
            os.path.exists(os.path.join(model_path, file)) for file in model_files
    ):
        try:
            # Download the model files/directory
            url = "https://example.com/model_files.zip"  # Replace with the actual URL
            temp_file, _ = urllib.request.urlretrieve(url)
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(model_path)
            os.remove(temp_file)
            print("Model files downloaded successfully.")
        except Exception as e:
            raise ModelDownloadError("Error downloading the model files/directory.")
    else:
        print("Model files already exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Downloader")
    parser.add_argument("--download", action="store_true", help="Download the model files")
    args = parser.parse_args()

    if args.download:
        download_model_files("custom-model")
