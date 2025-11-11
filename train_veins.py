import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Load vein features and labels
vein_features = np.load('vein_features.npy')   # Shape: (N, feature_dim)
labels = np.load('labels.npy')                  # Shape: (N,)

# Convert to torch tensors
X = torch.tensor(vein_features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Updated classifier for vein features
class VeinClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_prob=0.3):
        super(VeinClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

num_classes = len(np.unique(labels))
input_dim = vein_features.shape[1]
n_splits = 5  # 5-fold cross-validation

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
acc_list, prec_list, f1_list = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{n_splits}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = VeinClassifier(input_dim=input_dim, num_classes=num_classes, dropout_prob=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    epochs = 100
    patience = 10
    best_loss = float('inf')
    stop_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        # Training accuracy
        _, train_preds = torch.max(outputs, 1)
        train_acc = (train_preds == y_train.to(device)).float().mean().item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test.to(device))
            val_loss = criterion(val_outputs, y_test.to(device))

            # Validation accuracy
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = (val_preds == y_test.to(device)).float().mean().item()

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()
        truths = y_test.cpu().numpy()
        acc = accuracy_score(truths, preds)
        prec = precision_score(truths, preds, average='weighted', zero_division=0)
        f1 = f1_score(truths, preds, average='weighted', zero_division=0)
        print(f"Fold {fold+1} - Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")
        acc_list.append(acc)
        prec_list.append(prec)
        f1_list.append(f1)

print(f"\nMean Accuracy: {np.mean(acc_list):.4f}")
print(f"Mean Precision: {np.mean(prec_list):.4f}")
print(f"Mean F1 Score: {np.mean(f1_list):.4f}")
