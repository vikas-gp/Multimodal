import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import os


# ============================================================
#            CUSTOM DATASET → LABEL FROM FILENAME
# ============================================================
class LeafCustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
#          BUILD DENSENET (PARTIAL FINE-TUNING FIXED)
# ============================================================
def build_densenet(num_classes):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features

    # ---------- Freeze ALL layers first ----------
    for param in model.features.parameters():
        param.requires_grad = False

    # ---------- Unfreeze ONLY last DenseBlock4 + Norm5 ----------
    for name, param in model.features.named_parameters():
        if "denseblock4" in name or "norm5" in name:
            param.requires_grad = True

    # ---------- Smaller stable classifier ----------
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


# ============================================================
#                    HELPER: LOAD IMAGES
# ============================================================
def load_dataset(root_dir):
    classes = ['healthy', 'Nitrogen', 'Potassium', 'Phosphorus', 'Sulphur', 'Zinc']
    class_to_idx = {cls.lower(): i for i, cls in enumerate(classes)}

    img_paths = []
    labels = []

    for fname in sorted(os.listdir(root_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            f_lower = fname.lower()
            label = None
            for cls in class_to_idx:
                if cls in f_lower:
                    label = class_to_idx[cls]
                    break
            if label is None:
                raise ValueError(f"Class label not found in filename: {fname}")

            img_paths.append(os.path.join(root_dir, fname))
            labels.append(label)

    return np.array(img_paths), np.array(labels), classes


# ============================================================
#                        K-FOLD TRAINING
# ============================================================
def train_kfold(root_dir, epochs=25, batch_size=32, n_splits=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset paths + labels
    img_paths, labels, class_names = load_dataset(root_dir)
    num_classes = len(class_names)

    # ---------- Light Augmentation (stable for small datasets) ----------
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # K-Fold setup
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_acc, fold_prec, fold_f1 = [], [], []

    # ============================================================
    #                         FOLDS START
    # ============================================================
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(img_paths, labels)):
        print(f"\n========== Fold {fold_idx+1}/{n_splits} ==========")

        train_ds = LeafCustomDataset(img_paths[train_idx], labels[train_idx], transform=train_tf)
        val_ds   = LeafCustomDataset(img_paths[val_idx], labels[val_idx], transform=val_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Build model
        model = build_densenet(num_classes).to(device)

        # Optimizer: ONLY trainable params
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4      # lower LR for fine-tuning
        )

        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience = 6
        stop_count = 0

        # ============================================================
        #                         TRAINING LOOP
        # ============================================================
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for imgs, y in train_loader:
                imgs, y = imgs.to(device), y.to(device)

                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (out.argmax(1) == y).sum().item()
                train_total += y.size(0)

            train_acc = train_correct / train_total

            # -------- Validation --------
            model.eval()
            val_loss, preds_all, true_all = 0, [], []

            with torch.no_grad():
                for imgs, y in val_loader:
                    imgs, y = imgs.to(device), y.to(device)

                    out = model(imgs)
                    loss = criterion(out, y)

                    val_loss += loss.item()
                    preds_all.extend(out.argmax(1).cpu().numpy())
                    true_all.extend(y.cpu().numpy())

            acc = accuracy_score(true_all, preds_all)
            prec = precision_score(true_all, preds_all, average="weighted", zero_division=0)
            f1 = f1_score(true_all, preds_all, average="weighted", zero_division=0)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {acc:.4f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # -------- Early stopping --------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stop_count = 0
                torch.save(model.state_dict(), f"best_densenet_fold{fold_idx+1}.pth")
            else:
                stop_count += 1
                if stop_count >= patience:
                    print("Early stopping triggered!")
                    break

        fold_acc.append(acc)
        fold_prec.append(prec)
        fold_f1.append(f1)

        print(f"\nFold {fold_idx+1} Results → "
              f"Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")

    # ============================================================
    #                       FINAL METRICS
    # ============================================================
    print("\n======== Final Mean Metrics ========")
    print("Mean Accuracy: ", np.mean(fold_acc))
    print("Mean Precision:", np.mean(fold_prec))
    print("Mean F1 Score: ", np.mean(fold_f1))


# ============================================================
#                    MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    train_kfold(
        root_dir=r"D:\Multimodal\leaf",
        epochs=25,
        batch_size=32,
        n_splits=5
    )
