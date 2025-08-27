import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("bmi.csv")

# Encode gender column (assuming 'Male' and 'Female' values)
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])  # 0 for female, 1 for male

# Normalize height and weight
scaler = StandardScaler()
df[['height', 'weight']] = scaler.fit_transform(df[['height', 'weight']])

# Define dataset class
class BMIDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(data[['gender', 'height', 'weight']].values, dtype=torch.float32)
        self.y = torch.tensor(data['BMI Index'].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split dataset
train_size = int(0.8 * len(df))
train_dataset = BMIDataset(df[:train_size])
test_dataset = BMIDataset(df[train_size:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Transformer-based model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=6, embed_dim=16, num_heads=2, ff_dim=32, num_layers=2):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(2, embed_dim)  # Gender has 2 categories
        self.linear = nn.Linear(2, embed_dim)  # Map height and weight to embedding size
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        gender_embed = self.embedding(x[:, 0].long())  # Gender embedding
        num_features = self.linear(x[:, 1:].unsqueeze(-1))  # Height & weight mapping

        combined = gender_embed + num_features.squeeze(-1)
        combined = combined.unsqueeze(0)  # Add batch sequence dimension for Transformer

        transformed = self.transformer_encoder(combined)
        out = self.fc(transformed.squeeze(0))
        
        return self.softmax(out)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Train the model
train_model(model, train_loader)
torch.save(model.state_dict(), "bmi_transformer.pth")

