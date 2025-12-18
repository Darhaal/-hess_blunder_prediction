import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from model import ChessCNN

X = torch.tensor(np.load("data/processed/X.npy"))
y = np.load("data/processed/y.npy")

model = ChessCNN()
model.load_state_dict(torch.load("model.pt"))
model.eval()

with torch.no_grad():
    preds = model(X).numpy()

auc = roc_auc_score(y, preds)
print("ROC-AUC:", auc)
