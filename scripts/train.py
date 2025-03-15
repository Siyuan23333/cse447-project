import os
import json
import torch
import random
import string
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW

# Define constants
MODEL_NAME = "xlm-roberta-base"
WORK_DIR = "work"
DATA_PATH = "data/processed/training_data.json"
BATCH_SIZE = 16
EPOCHS = 5
SEQ_LENGTH = 20  # Input sequence length

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Define an MLP classifier for next-character prediction
class CharacterPredictor(torch.nn.Module):
    def __init__(self, base_model):
        super(CharacterPredictor, self).__init__()
        self.encoder = base_model
        self.hidden_size = base_model.config.hidden_size
        self.classifier = torch.nn.Linear(self.hidden_size, len(string.ascii_letters) + 10)  # Output size

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, -1, :]  # Use last token embedding
        logits = self.classifier(hidden_states)
        return logits

# Load dataset
class CharacterDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_char = self.data[idx]
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_char, dtype=torch.long)
        return input_tensor, target_tensor

# Prepare dataset and DataLoader
dataset = CharacterDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CharacterPredictor(model).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for input_ids, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids, target = input_ids.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=(input_ids > 0).long())
            
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    # Save trained model
    os.makedirs(WORK_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(WORK_DIR, "xlmr_character_model.pt"))
    print("Training complete! Model saved.")

if __name__ == "__main__":
    train()
