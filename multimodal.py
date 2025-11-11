import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Load pre-extracted features and labels
leaf_features = np.load('leaf_features.npy')  # Shape: (N, 1024)
vein_features = np.load('vein_features.npy')  # Shape: (N, 128)
labels = np.load('labels.npy')  # Shape: (N,)

# Convert to torch tensors
X_leaf = torch.tensor(leaf_features, dtype=torch.float32)
X_vein = torch.tensor(vein_features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a simple fusion classifier using DenseNet-like MLP
class DenseNetClassifier(nn.Module):
    def __init__(self, leaf_dim=1024, vein_dim=128, hidden_dim=256, num_classes=10, dropout_prob=0.5):
        super(DenseNetClassifier, self).__init__()
        self.fc1 = nn.Linear(leaf_dim + vein_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_leaf, x_vein):
        x = torch.cat((x_leaf, x_vein), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

num_classes = len(np.unique(labels))
n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
acc_list, prec_list, f1_list = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_leaf, y)):
    print(f"Fold {fold+1}/{n_splits}")

    leaf_train, leaf_val = X_leaf[train_idx], X_leaf[test_idx]
    vein_train, vein_val = X_vein[train_idx], X_vein[test_idx]
    y_train, y_val = y[train_idx], y[test_idx]

    train_ds = TensorDataset(leaf_train, vein_train, y_train)
    val_ds = TensorDataset(leaf_val, vein_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = DenseNetClassifier(leaf_dim=1024, vein_dim=128, hidden_dim=256, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=3e-4)

    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    stop_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for leaf_batch, vein_batch, labels_batch in train_loader:
            leaf_batch = leaf_batch.to(device)
            vein_batch = vein_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(leaf_batch, vein_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * leaf_batch.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels_batch).sum().item()
            total_train += leaf_batch.size(0)

        train_loss /= total_train
        train_acc = train_correct / total_train

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for leaf_batch, vein_batch, labels_batch in val_loader:
                leaf_batch = leaf_batch.to(device)
                vein_batch = vein_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(leaf_batch, vein_batch)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item() * leaf_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels_batch).sum().item()
                total_val += leaf_batch.size(0)

        val_loss /= total_val
        val_acc = val_correct / total_val

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
        else:
            stop_counter += 1

        if stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for leaf_batch, vein_batch, labels_batch in val_loader:
            leaf_batch = leaf_batch.to(device)
            vein_batch = vein_batch.to(device)
            outputs = model(leaf_batch, vein_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"Fold {fold+1} - Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")

    acc_list.append(acc)
    prec_list.append(prec)
    f1_list.append(f1)

print(f"\nMean Accuracy: {np.mean(acc_list):.4f}")
print(f"Mean Precision: {np.mean(prec_list):.4f}")
print(f"Mean F1 Score: {np.mean(f1_list):.4f}")
