import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Load the dataset
df = pd.read_csv("sample_dataset.csv")

# Split the dataset into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].values,
    df["label"].values,
    test_size=0.2,
    random_state=42
)

# Define the label mapping
label_mapping = {"positive": 1, "negative": 0, "neutral": 0.5}

# Map the string labels to numerical values
train_labels = [label_mapping[label] for label in train_labels]
test_labels = [label_mapping[label] for label in test_labels]

# Define the number of classes
num_classes = len(label_mapping)

# Load the pre-trained model and tokenizer
model_name = 'distilbert-base-uncased-distilled-squad'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

# Data sampling
sample_size = 5000  # Set the desired sample size
indices = np.random.choice(len(train_texts), sample_size, replace=False)
train_texts = train_texts[indices]
train_labels = [train_labels[i] for i in indices]

# Tokenize and encode the training data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings["input_ids"]),
    torch.tensor(train_encodings["attention_mask"]),
    torch.tensor(train_labels)
)

# Tokenize and encode the test data
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings["input_ids"]),
    torch.tensor(test_encodings["attention_mask"]),
    torch.tensor(test_labels)
)

# Fine-tuning parameters
batch_size = 16
learning_rate = 2e-5
num_epochs = 3

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset))

# Prepare optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, verbose=True)

# Initialize variables to track the best model and its accuracy
best_accuracy = 0.0
best_model_path = None
early_stop_counter = 0
early_stop_patience = 2

# Fine-tuning loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# Inside the fine-tuning loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes).float().to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print batch progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            progress = ((batch_idx + 1) / len(train_loader)) * 100
            print(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {running_loss:.4f}, Progress: {progress:.2f}%")

    # Evaluate on the validation set
    model.eval()
    eval_accuracy = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            eval_accuracy += torch.sum(predictions == labels).item()

    accuracy = eval_accuracy / len(test_texts)
    print("Epoch:", epoch + 1)
    print("Validation Accuracy:", accuracy)

    # Learning rate scheduling
    scheduler.step(accuracy)

    # Early stopping
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = f"sentiment_model_accuracy_{accuracy:.4f}_on_epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), best_model_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Load the best model
best_model = DistilBertForSequenceClassification.from_pretrained(best_model_path)
best_model.to(device)

# Tokenize and encode the test data
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings["input_ids"]),
    torch.tensor(test_encodings["attention_mask"]),
    torch.tensor(test_labels)
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Evaluation
best_model.eval()
eval_accuracy = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = best_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        eval_accuracy += torch.sum(predictions == labels).item()

    accuracy = eval_accuracy / len(test_texts)
    print("Best Model Accuracy:", accuracy)
