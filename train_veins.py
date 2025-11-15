import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score
from PIL import Image
import numpy as np


class VeinDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


#Vein Model
class VeinFrozenModel(nn.Module):
    def __init__(self, num_classes):
        super(VeinFrozenModel, self).__init__()

        self.base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.base.classifier = nn.Identity()

        for param in self.base.parameters():
            param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.base.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  # [batch, 1024]
        return self.classifier(x)



def load_vein_dataset(root_dir):
    classes = ['healthy', 'Nitrogen', 'Potassium', 'Phosphorus', 'Sulphur', 'Zinc']
    class_to_idx = {cls.lower(): i for i, cls in enumerate(classes)}

    img_paths, labels = [], []
    for fname in sorted(os.listdir(root_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            f_lower = fname.lower()
            label = None
            for cls in class_to_idx:
                if cls in f_lower:
                    label = class_to_idx[cls]
                    break
            if label is None:
                raise ValueError(f"Class not found in filename: {fname}")
            img_paths.append(os.path.join(root_dir, fname))
            labels.append(label)

    return np.array(img_paths), np.array(labels), classes


def train_vein_frozen(root_dir, epochs=25, batch_size=32, n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_paths, labels, class_names = load_vein_dataset(root_dir)
    num_classes = len(class_names)

    # Transforms
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

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list, prec_list, f1_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(img_paths, labels)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")

        train_ds = VeinDataset(img_paths[train_idx], labels[train_idx], transform=train_tf)
        val_ds = VeinDataset(img_paths[val_idx], labels[val_idx], transform=val_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = VeinFrozenModel(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

        best_val_loss = float('inf')
        patience, stop_counter = 5, 0

        for epoch in range(epochs):
            model.train()
            total_loss, total_correct, total_samples = 0, 0, 0

            for imgs, labels_batch in train_loader:
                imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == labels_batch).sum().item()
                total_samples += labels_batch.size(0)

            train_acc = total_correct / total_samples

            # Validation
            model.eval()
            val_loss, val_preds, val_true = 0, [], []
            with torch.no_grad():
                for imgs, labels_batch in val_loader:
                    imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()
                    val_preds.extend(outputs.argmax(1).cpu().numpy())
                    val_true.extend(labels_batch.cpu().numpy())

            acc = accuracy_score(val_true, val_preds)
            prec = precision_score(val_true, val_preds, average='weighted', zero_division=0)
            f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)

            print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {acc:.4f} | "
                  f"Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stop_counter = 0
                torch.save(model.state_dict(), f"best_vein_fold{fold+1}.pth")
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    print("Early stopping triggered!")
                    break

        acc_list.append(acc)
        prec_list.append(prec)
        f1_list.append(f1)
        print(f"Fold {fold+1} → Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")

    print("\n===== FINAL VEIN MODEL METRICS =====")
    print(f"Mean Accuracy:  {np.mean(acc_list):.4f}")
    print(f"Mean Precision: {np.mean(prec_list):.4f}")
    print(f"Mean F1 Score:  {np.mean(f1_list):.4f}")

    with open("vein_results.txt", "w") as f:
        f.write("===== FINAL VEIN MODEL METRICS =====\n")
        f.write(f"Mean Accuracy:  {np.mean(acc_list):.4f}\n")
        f.write(f"Mean Precision: {np.mean(prec_list):.4f}\n")
        f.write(f"Mean F1 Score:  {np.mean(f1_list):.4f}\n")


if __name__ == "__main__":
    train_vein_frozen(
        root_dir=r"/teamspace/studios/this_studio/Multimodal/veins_rgb",
        epochs=25,
        batch_size=32,
        n_splits=5
    )
