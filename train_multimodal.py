import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score


# ============================
#   LOAD FEATURES
# ============================
leaf_features = np.load("/teamspace/studios/this_studio/Multimodal/leaf_features.npy")      # (948, 1024)
vein_features = np.load("/teamspace/studios/this_studio/Multimodal/vein_features.npy")      # (948, 128)
labels = np.load("/teamspace/studios/this_studio/Multimodal/labels.npy")                    # (948,)

# --------- Concatenate multimodal features ---------
X = np.concatenate([leaf_features, vein_features], axis=1)  
# Final shape = (948, 1152)

print("Fused Feature Shape:", X.shape)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
#   MULTIMODAL MODEL
# ============================
class MultiModalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiModalClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


input_dim = X.shape[1]   # 1152
num_classes = len(np.unique(labels))


# ============================
#   K-FOLD TRAINING
# ============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_list, prec_list, f1_list = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold+1}/5 =====")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = MultiModalClassifier(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 50
    patience = 7
    best_loss = float('inf')
    stop_count = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test.to(device))
            val_loss = criterion(val_outputs, y_test.to(device))

        print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            stop_count = 0
        else:
            stop_count += 1

        if stop_count >= patience:
            print("Early stopping triggered!")
            break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()
        truths = y_test.cpu().numpy()

        acc = accuracy_score(truths, preds)
        prec = precision_score(truths, preds, average="weighted", zero_division=0)
        f1 = f1_score(truths, preds, average="weighted", zero_division=0)

        print(f"Fold {fold+1} Results → Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")

        acc_list.append(acc)
        prec_list.append(prec)
        f1_list.append(f1)


print("\n===== FINAL MULTIMODAL METRICS =====")
print("Mean Accuracy:", np.mean(acc_list))
print("Mean Precision:", np.mean(prec_list))
print("Mean F1 Score:", np.mean(f1_list))
# ===== SAVE METRICS TO FILE =====
with open("multimodal_results.txt", "w") as f:
    f.write("===== Multimodal Classifier Final Results =====\n")
    f.write(f"Mean Accuracy:  {np.mean(acc_list):.4f}\n")
    f.write(f"Mean Precision: {np.mean(prec_list):.4f}\n")
    f.write(f"Mean F1 Score:  {np.mean(f1_list):.4f}\n")

print("\nSaved results to multimodal_results.txt ✅")
