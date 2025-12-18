import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import ChessCNN

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

X = torch.tensor(X)
y = torch.tensor(y).float().unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ChessCNN()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    total_loss = 0
    for xb, yb in loader:
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss = {total_loss:.4f}")

torch.save(model.state_dict(), "model.pt")
