# Importing Libraries
import pandas as pd
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset from a CSV file
csv_file_path = "dataset/sentiment-classification.csv"
dataframe = pd.read_csv(csv_file_path)
print("Dataset loaded from CSV.")

# Split the dataset into training and evaluation subsets
train_df, eval_df = train_test_split(dataframe, test_size=0.2, random_state=42)
print("Dataset split into training and evaluation subsets.")

# Convert DataFrames to Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
print("Data converted to Datasets.")

# Encode categorical labels
label_encoder = LabelEncoder()
train_dataset = train_dataset.map(lambda example: {"label": label_encoder.fit_transform(example["label"])}, batched=True)
eval_dataset = eval_dataset.map(lambda example: {"label": label_encoder.transform(example["label"])}, batched=True)
print("Labels encoded with label encoder.")

# Load a SetFit model from the Hugging Face Hub
model = SetFitModel.from_pretrained("StatsGary/setfit-ft-sentinent-eval")
print("SetFit model loaded from the Hugging Face Hub.")

# Create the trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20,
    num_epochs=1,
    column_mapping={"text": "text", "label": "label"}
)
print("Trainer created.")

# Train the model
print("Training the model...")
trainer.train()
print("Training complete.")

# Convert evaluation dataset back to Pandas DataFrame
eval_df = eval_dataset.to_pandas()

# Evaluate the model
print("Evaluating the model...")
eval_texts = eval_df["text"]
eval_labels = eval_df["label"]
eval_predictions = model.predict(eval_texts)
evaluation_metrics = accuracy_score(eval_labels, eval_predictions)
print("Evaluation Accuracy:", evaluation_metrics)

# Save the best model locally
best_model_path = "../custom-model"
model.save_pretrained(best_model_path)
print("Best model saved at:", best_model_path)

from sentence_transformers import SentenceTransformer

# Load the saved model
saved_model_path = "../custom-model"
model = SentenceTransformer(saved_model_path)

# Example text inputs
texts = [
    "I love this product!",
    "This movie was disappointing.",
    "The quality of service was excellent.",
    "I would not recommend this restaurant.",
    "The book was engaging and well-written."
]

# Encode the texts
encoded_texts = model.encode(texts)

# Print the encoded representations
for text, encoded_text in zip(texts, encoded_texts):
    print("Text:", text)
    print("Encoded representation:", encoded_text)
    print()

from setfit import SetFitModel

# Load the model
model_path = "../custom-model"  # Replace with the actual path to your model
# model_path = "StatsGary/setfit-ft-sentinent-eval"
model = SetFitModel.from_pretrained(model_path)

# List of texts to analyze
text_list = ["I love this product!", "This movie is terrible.", "The weather is nice today.", "Are you coming today?"]

# Prepare the input
inputs = text_list

# Perform sentiment analysis
predictions = model(inputs)

# Process the predictions and convert them to sentiment labels
def process_predictions(preds):
    sentiment_labels = []
    for pred in preds:
        if pred >= 2:
            sentiment_labels.append("positive")
        elif pred < 1:
            sentiment_labels.append("negative")
        else:
            sentiment_labels.append("neutral")
    return sentiment_labels

# Associate the sentiment labels with the corresponding texts
sentiment_labels = process_predictions(predictions)
results = dict(zip(text_list, sentiment_labels))

# Print the results
for text, label in results.items():
    print(f"Text: {text}\nSentiment: {label}\n")