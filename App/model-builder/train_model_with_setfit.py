import pandas as pd
import yaml
from datasets import Dataset
from pydantic import BaseModel
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class SentimentExample(BaseModel):
    text: str
    label: str


def load_dataset(csv_file_path):
    """
    Loads a CSV file as a pandas DataFrame.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file is not found.
        Exception: If an error occurs while parsing the CSV file.
    """
    try:
        dataframe = pd.read_csv(csv_file_path)
        print("Dataset loaded from CSV.")
        return dataframe
    except FileNotFoundError as e:
        raise FileNotFoundError("CSV file not found.") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError("Error occurred while parsing the CSV file.") from e


def split_dataset(dataframe):
    """
    Splits the dataset into training and evaluation subsets.

    Args:
        dataframe (pd.DataFrame): Dataset as a pandas DataFrame.

    Returns:
        pd.DataFrame, pd.DataFrame: Training and evaluation subsets as pandas DataFrames.

    Raises:
        Exception: If an error occurs while splitting the dataset.
    """
    try:
        train_df, eval_df = train_test_split(dataframe, test_size=0.2, random_state=42)
        print("Dataset split into training and evaluation subsets.")
        return train_df, eval_df
    except ValueError as e:
        raise ValueError("Error occurred while splitting dataset.") from e


def convert_to_datasets(train_df, eval_df):
    """
    Converts the pandas DataFrames to datasets of the `datasets` library.

    Args:
        train_df (pd.DataFrame): Training subset as a pandas DataFrame.
        eval_df (pd.DataFrame): Evaluation subset as a pandas DataFrame.

    Returns:
        datasets.Dataset, datasets.Dataset: Training and evaluation datasets.

    Raises:
        Exception: If an error occurs while converting to Datasets.
    """
    try:
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        print("Data converted to Datasets.")
        return train_dataset, eval_dataset
    except Exception as e:
        raise RuntimeError("Error occurred while converting to Datasets.") from e


def encode_labels(train_dataset, eval_dataset):
    """
    Encodes the labels using scikit-learn's LabelEncoder.

    Args:
        train_dataset (datasets.Dataset): Training dataset.
        eval_dataset (datasets.Dataset): Evaluation dataset.

    Returns:
        datasets.Dataset, datasets.Dataset, sklearn.preprocessing.LabelEncoder:
            Encoded training and evaluation datasets, and label encoder.

    Raises:
        ValueError: If an error occurs while encoding labels.
    """
    try:
        label_encoder = LabelEncoder()
        train_dataset = train_dataset.map(
            lambda example: {"label": label_encoder.fit_transform(example["label"])},
            batched=True,
        )
        eval_dataset = eval_dataset.map(
            lambda example: {"label": label_encoder.transform(example["label"])},
            batched=True,
        )
        print("Labels encoded with label encoder.")
        return train_dataset, eval_dataset, label_encoder
    except ValueError as e:
        raise ValueError("Error occurred while encoding labels.") from e


def load_model(model_name):
    """
    Loads the SetFit model from the Hugging Face Hub.

    Args:
        model_name (str): Name of the SetFit model.

    Returns:
        SetFitModel: Loaded SetFit model.

    Raises:
        OSError: If an error occurs while loading the SetFit model.
    """
    try:
        model = SetFitModel.from_pretrained(model_name)
        print("SetFit model loaded from the Hugging Face Hub.")
        return model
    except OSError as e:
        raise OSError("Error occurred while loading SetFit model.") from e


def train_model(
        model,
        train_dataset,
        eval_dataset,
        loss_class,
        metric,
        batch_size,
        num_iterations,
        num_epochs,
        column_mapping,
):
    """
    Trains the model using the SetFitTrainer.

    Args:
        model (SetFitModel): SetFit model.
        train_dataset (datasets.Dataset): Training dataset.
        eval_dataset (datasets.Dataset): Evaluation dataset.
        loss_class: Loss class for training.
        metric (str): Metric for evaluation.
        batch_size (int): Batch size for training.
        num_iterations (int): Number of iterations for training.
        num_epochs (int): Number of epochs for training.
        column_mapping: Mapping between column names and input names.

    Raises:
        RuntimeError: If an error occurs while training the model.
    """
    try:
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss_class=loss_class,
            metric=metric,
            batch_size=batch_size,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            column_mapping=column_mapping,
        )
        print("Trainer created.")
        print("Training the model...")
        trainer.train()
        print("Training complete.")
    except RuntimeError as e:
        raise RuntimeError("Error occurred while training the model.") from e


def evaluate_model(model, eval_df):
    """
    Evaluates the trained model using accuracy score.

    Args:
        model (SetFitModel): Trained SetFit model.
        eval_df (pd.DataFrame): Evaluation subset as a pandas DataFrame.

    Raises:
        ValueError: If an error occurs while evaluating the model.
    """
    try:
        eval_texts = eval_df["text"]
        eval_labels = eval_df["label"]
        eval_predictions = model.predict(eval_texts)
        evaluation_metrics = accuracy_score(eval_labels, eval_predictions)
        print("Evaluation Accuracy:", evaluation_metrics)
    except ValueError as e:
        raise ValueError("Error occurred while evaluating the model.") from e


def save_model(model, save_path):
    """
    Saves the best model to a specified path.

    Args:
        model (SetFitModel): Trained SetFit model.
        save_path (str): Path to save the model.

    Raises:
        OSError: If an error occurs while saving the model.
    """
    try:
        model.save_pretrained(save_path)
        print("Best model saved at:", save_path)
    except OSError as e:
        raise OSError("Error occurred while saving the model.") from e


def load_config(config_file):
    """
    Loads a YAML configuration file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        Exception: If an error occurs while loading the configuration.
    """
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError as e:
        raise FileNotFoundError("Configuration file not found.") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError("Error occurred while loading configuration.") from e


def main():
    try:
        # Load configuration
        config = load_config("./config/config.yaml")

        # Load dataset
        dataset = load_dataset(config["csv_file_path"])

        # Split the dataset
        train_df, eval_df = split_dataset(dataset)

        # Convert to Datasets
        train_dataset, eval_dataset = convert_to_datasets(train_df, eval_df)

        # Encode labels
        train_dataset, eval_dataset, label_encoder = encode_labels(train_dataset, eval_dataset)

        # Load SetFit model
        model = load_model(config["model_name"])

        # Train the model
        train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=config["batch_size"],
            num_iterations=config["num_iterations"],
            num_epochs=config["num_epochs"],
            column_mapping=config["column_mapping"],
        )

        # Convert evaluation dataset back to Pandas DataFrame
        eval_df = eval_dataset.to_pandas()

        # Evaluate the model
        evaluate_model(model, eval_df)

        # Save the best model locally
        save_model(model, config["save_path"])
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
