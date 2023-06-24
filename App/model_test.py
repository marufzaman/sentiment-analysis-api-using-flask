import pandas as pd
from setfit import SetFitModel
from sklearn.metrics import accuracy_score


def load_model(model_name):
    try:
        model = SetFitModel.from_pretrained(model_name)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        raise Exception("Error occurred while loading the model.") from e


def test_model(model, test_file, result_file):
    try:
        test_data = pd.read_csv(test_file)
        texts = test_data['text']
        expected_labels = test_data['label']

        predictions = []
        for text in texts:
            predictions.append(model([text])[0].item())

        # Map label indices to label names
        label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_labels = [label_mapping[prediction] for prediction in predictions]

        # Calculate accuracy
        accuracy = accuracy_score(expected_labels, predicted_labels)

        # Create a DataFrame for failed cases
        failed_cases = pd.DataFrame({
            'text': texts,
            'expected_label': expected_labels,
            'resulted_label': predicted_labels
        })
        failed_cases = failed_cases[failed_cases['expected_label'] != failed_cases['resulted_label']]

        # Save failed cases to a CSV file
        failed_cases.to_csv(result_file, index=False)

        return accuracy

    except Exception as e:
        raise Exception("Error occurred while testing the model.") from e


# Specify the name of your custom model
model_name = "custom-model"

# Specify the path to your test CSV file
test_file = './dataset/sentiment-classification-unverified-BIG.csv'

# Specify the path for the result CSV file
result_file = './dataset/failed_cases.csv'

# Load the model
model = load_model(model_name)

# Test the model and calculate accuracy
accuracy = test_model(model, test_file, result_file)

print("Accuracy:", accuracy)
print("Failed cases saved to:", result_file)
