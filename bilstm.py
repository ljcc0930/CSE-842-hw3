import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def load_data_from_directory(data_dir):
    data = []
    for label, sentiment in enumerate(["pos", "neg"]):
        sentiment_dir = os.path.join(data_dir, sentiment)
        for filename in os.listdir(sentiment_dir):
            with open(os.path.join(sentiment_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                data.append({"text": text, "label": label})
    return pd.DataFrame(data)

data_dir = "/storage/users/ljcc0930/workspace/CSE-842-hw/data/txt_sentoken"

df = load_data_from_directory(data_dir)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

MAX_LENGTH = 256
BATCH_SIZE = 16

train_dataset = SentimentDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = SentimentDataset(val_df, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.fc(hidden)
        return output

input_dim = 768
hidden_dim = 128
num_layers = 2
output_dim = 2

lstm_model = BiLSTM(input_dim, hidden_dim, num_layers, output_dim)
lstm_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

EPOCHS = 20

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    lstm_model.train()
    train_loss = 0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        bert_model.eval()
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
        
        lstm_input = outputs.last_hidden_state.permute(0, 2, 1).permute(0, 2, 1)
    optimizer.zero_grad()
    lstm_outputs = lstm_model(lstm_input)
    loss = criterion(lstm_outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

train_loss /= len(train_loader)

lstm_model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        bert_model.eval()
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
        lstm_input = outputs.last_hidden_state.permute(0, 2, 1).permute(0, 2, 1)

        lstm_outputs = lstm_model(lstm_input)
        _, preds = torch.max(lstm_outputs, dim=1)

        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)

val_accuracy = correct_predictions / total_predictions
print(f"Train loss: {train_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")

