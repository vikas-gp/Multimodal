import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#Custom Dataset for multimodal
class MultiModalDataset(Dataset):
    def __init__(self, leaf_paths, vein_paths, labels, transform=None):
        self.leaf_paths = leaf_paths
        self.vein_paths = vein_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        leaf_img = Image.open(self.leaf_paths[idx]).convert("RGB")
        vein_img = Image.open(self.vein_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            leaf_img = self.transform(leaf_img)
            vein_img = self.transform(vein_img)

        return leaf_img, vein_img, label


# Frozen Densenet Feature Extractors
def build_frozen_densenet():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Identity()  # Removes classifier
    return model

# Multimodal Fusion Network
class MultiModalFusion(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalFusion, self).__init__()
        self.leaf_net = build_frozen_densenet()
        self.vein_net = build_frozen_densenet()

        self.fc_layers = nn.Sequential(
            nn.Linear(1024 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, leaf_img, vein_img):
        leaf_feat = self.leaf_net(leaf_img)
        vein_feat = self.vein_net(vein_img)
        fused = torch.cat([leaf_feat, vein_feat], dim=1)
        return self.fc_layers(fused)

def load_multimodal_dataset(leaf_dir, vein_dir):
    classes = ['healthy', 'nitrogen', 'potassium', 'phosphorus', 'sulphur', 'zinc']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    leaf_paths, vein_paths, labels = [], [], []

    vein_files = {
        os.path.splitext(f.lower())[0]: os.path.join(vein_dir, f)
        for f in os.listdir(vein_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    }

    for fname in sorted(os.listdir(leaf_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            f_lower = fname.lower()
            base = os.path.splitext(f_lower)[0]

            label = None
            for cls in classes:
                if cls in f_lower:
                    label = class_to_idx[cls]
                    break
            if label is None:
                continue

            vein_match = None
            for vname, vpath in vein_files.items():
                if base in vname:
                    vein_match = vpath
                    break

            if vein_match:
                leaf_paths.append(os.path.join(leaf_dir, fname))
                vein_paths.append(vein_match)
                labels.append(label)

    print(f"Loaded {len(labels)} image pairs successfully.")
    return np.array(leaf_paths), np.array(vein_paths), np.array(labels), classes


# K-Fold Training 
def train_multimodal(leaf_dir, vein_dir, epochs=25, batch_size=16, n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    leaf_paths, vein_paths, labels, classes = load_multimodal_dataset(leaf_dir, vein_dir)
    num_classes = len(classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_acc, fold_prec, fold_f1 = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(leaf_paths, labels)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")

        train_ds = MultiModalDataset(leaf_paths[train_idx], vein_paths[train_idx], labels[train_idx], transform)
        val_ds = MultiModalDataset(leaf_paths[val_idx], vein_paths[val_idx], labels[val_idx], transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = MultiModalFusion(num_classes).to(device)
        optimizer = optim.Adam(model.fc_layers.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience, stop_counter = 7, 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss, correct, total = 0, 0, 0

            for leaf, vein, y in train_loader:
                leaf, vein, y = leaf.to(device), vein.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(leaf, vein)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

            train_acc = correct / total

            model.eval()
            val_loss, preds, trues = 0, [], []
            with torch.no_grad():
                for leaf, vein, y in val_loader:
                    leaf, vein, y = leaf.to(device), vein.to(device), y.to(device)
                    out = model(leaf, vein)
                    loss = criterion(out, y)
                    val_loss += loss.item()
                    preds.extend(out.argmax(1).cpu().numpy())
                    trues.extend(y.cpu().numpy())

            acc = accuracy_score(trues, preds)
            prec = precision_score(trues, preds, average='weighted', zero_division=0)
            f1 = f1_score(trues, preds, average='weighted', zero_division=0)

            print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {acc:.4f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_multimodal_fold{fold+1}.pth")
                stop_counter = 0
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

        fold_acc.append(acc)
        fold_prec.append(prec)
        fold_f1.append(f1)

        print(f"Fold {fold+1} → Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")

    print("\n===== FINAL MULTIMODAL METRICS =====")
    print(f"Mean Accuracy: {np.mean(fold_acc):.4f}")
    print(f"Mean Precision: {np.mean(fold_prec):.4f}")
    print(f"Mean F1 Score: {np.mean(fold_f1):.4f}")

    with open("multimodal_results.txt", "w") as f:
        f.write("===== FINAL MULTIMODAL METRICS =====\n")
        f.write(f"Mean Accuracy: {np.mean(fold_acc):.6f}\n")
        f.write(f"Mean Precision: {np.mean(fold_prec):.6f}\n")
        f.write(f"Mean F1 Score: {np.mean(fold_f1):.6f}\n")

    print("\nSaved results to multimodal_results.txt ")


if __name__ == "__main__":
    train_multimodal(
        leaf_dir=r"/teamspace/studios/this_studio/Multimodal/leaf",
        vein_dir=r"/teamspace/studios/this_studio/Multimodal/veins_rgb",
        epochs=25,
        batch_size=16,
        n_splits=5
    )
