# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

# Load dataset from a CSV file
csv_file_path = "../dataset/sentiment-classification.csv"
dataframe = pd.read_csv(csv_file_path)
print("Dataset loaded from CSV.")

# Split the dataset into training and evaluation subsets
train_df, eval_df = train_test_split(
    dataframe,
    test_size=0.2,
    random_state=42)
print("Dataset split into training and evaluation subsets.")

# Convert DataFrames to Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
print("Data converted to Datasets.")

# Load a SetFit model from the Hugging Face Hub
model = SetFitModel.from_pretrained("StatsGary/setfit-ft-sentinent-eval")
print("SetFit model loaded from the Hugging Face Hub.")

print("Dataset columns:", train_dataset.column_names)

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

# Evaluate the model
print("Evaluating the model...")
evaluation_metrics = trainer.evaluate()
print("Evaluation Metrics:", evaluation_metrics)

# Save the best model locally
best_model_path = "./best_model"
model.save_pretrained(best_model_path)
print("Best model saved at:", best_model_path)