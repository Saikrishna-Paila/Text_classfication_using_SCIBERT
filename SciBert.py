import os
import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords

# Download necessary nltk data
nltk.download('stopwords')

# --------------- Set up environment ---------------
print("Setting up the environment...")
checkpoint_dir = "./checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)
train_path = "Train_ML.csv"
test_path = "Test_submission_netid.csv"
submission_path = "Test_submission_netid.csv"

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

# --------------- Load pre-trained model and tokenizer ---------------
print("Loading SciBERT model and tokenizer...")
scibert_model_name = "allenai/scibert_scivocab_uncased"
scibert_tokenizer = AutoTokenizer.from_pretrained(scibert_model_name)
scibert_model = AutoModel.from_pretrained(scibert_model_name)

# --------------- Preprocessing function ---------------
print("Defining text preprocessing function...")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Preprocesses the input text: remove stopwords, URLs, and non-alphabetic characters.
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# --------------- Dataset class ---------------
print("Defining dataset class...")
class AbstractDataset(Dataset):
    def __init__(self, scibert_inputs, labels=None):
        self.scibert_inputs = scibert_inputs
        self.labels = labels

    def __len__(self):
        return len(self.scibert_inputs)

    def __getitem__(self, index):
        # Returns tokenized inputs and labels if available
        item = {
            "input_ids": self.scibert_inputs[index]["input_ids"].squeeze(0),
            "attention_mask": self.scibert_inputs[index]["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float)
        return item

# --------------- Model definition ---------------
print("Defining the MultiHeadClassifier model...")
class MultiHeadClassifier(nn.Module):
    def __init__(self, scibert_model, num_classes, dropout_rate=0.3):
        super(MultiHeadClassifier, self).__init__()
        self.scibert_model = scibert_model
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = scibert_model.config.hidden_size

        # Fully connected head
        self.fc_head = nn.Linear(hidden_size, num_classes)

        # LSTM head
        self.lstm_head = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.lstm_fc = nn.Linear(hidden_size, num_classes)

        # Convolutional head
        self.conv_head = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Forward pass through SciBERT
        outputs = self.scibert_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Fully connected head
        fc_output = self.fc_head(self.dropout(hidden_states[:, 0, :]))

        # LSTM head
        lstm_output, _ = self.lstm_head(hidden_states)
        lstm_output = self.lstm_fc(self.dropout(lstm_output[:, -1, :]))

        # Convolutional head
        conv_input = hidden_states.permute(0, 2, 1)
        conv_output = self.conv_head(conv_input)
        conv_output = self.conv_fc(self.dropout(conv_output.mean(dim=2)))

        # Combine the outputs from different heads
        final_output = (fc_output + lstm_output + conv_output) / 3
        return final_output

# --------------- Validation and Metric Functions ---------------
def validate(model, val_loader):
    print("Validating...")
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    return total_loss / len(val_loader), np.vstack(all_labels), np.vstack(all_preds)

def evaluate_metrics(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    kappa = cohen_kappa_score(y_true.argmax(axis=1), y_pred_binary.argmax(axis=1), weights="quadratic")
    return accuracy, kappa

# --------------- Load and preprocess data ---------------
print("Loading and preprocessing training data...")
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("Preprocessing the text data...")
train_data['ABSTRACT'] = train_data['ABSTRACT'].apply(preprocess_text)
test_data['ABSTRACT'] = test_data['ABSTRACT'].apply(preprocess_text)

# Tokenize texts
print("Tokenizing the text data...")
train_scibert_inputs = [scibert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256) for text in train_data['ABSTRACT']]
test_scibert_inputs = [scibert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256) for text in test_data['ABSTRACT']]

target_columns = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology", "Quantitative Finance"]
labels = train_data[target_columns].values

train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_scibert_inputs, labels, test_size=0.2, random_state=42
)

train_dataset = AbstractDataset(train_inputs, train_labels)
val_dataset = AbstractDataset(val_inputs, val_labels)
test_dataset = AbstractDataset(test_scibert_inputs)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --------------- Model, loss, and optimizer setup ---------------
print("Initializing the model, optimizer, and loss function...")
model = MultiHeadClassifier(scibert_model, num_classes=len(target_columns), dropout_rate=0.3).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)
criterion = nn.BCEWithLogitsLoss()

# --------------- Training with Early Stopping ---------------
print("Starting training loop...")
epochs = 10
patience = 3
best_metric_sum = 0
patience_counter = 0

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Training loss: {avg_train_loss:.4f}")

    # Validation step
    val_loss, val_labels, val_preds = validate(model, val_loader)
    accuracy, quadratic_kappa = evaluate_metrics(val_labels, val_preds)
    metric_sum = accuracy + quadratic_kappa

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}, Quadratic Kappa: {quadratic_kappa:.4f}")
    print(f"Sum of Accuracy and Quadratic Kappa: {metric_sum:.4f}")

    # Save the best model if metric_sum improves
    if metric_sum > best_metric_sum:
        best_metric_sum = metric_sum
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        patience_counter = 0
        print("Model improved and saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print(f"Best Sum of Accuracy and Quadratic Kappa: {best_metric_sum:.4f}")

# --------------- Testing and Saving Predictions ---------------
print("Loading the best model for testing...")
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth")))
model.eval()
print("Generating predictions...")
test_preds = []
for batch in tqdm(test_loader, desc="Testing"):
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        preds = torch.sigmoid(outputs).cpu().numpy()
        test_preds.extend(preds)

test_preds = np.array(test_preds)
test_final_preds = (test_preds > 0.5).astype(int)

for i, col in enumerate(target_columns):
    test_data[col] = test_final_preds[:, i]

print("Saving predictions to CSV...")
test_data.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")
# --------------- END ---------------
# Saikrishna Paila
# G39129775