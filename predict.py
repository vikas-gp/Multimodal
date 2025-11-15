import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Frozen DenseNet Feature Extractor
def build_frozen_densenet():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Identity()
    return model


# Multimodal Classifier
class MultiModalFusion(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
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

# Prediction Function
def predict_multimodal(leaf_path, vein_path, checkpoint_path, class_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    leaf_img = Image.open(leaf_path).convert("RGB")
    vein_img = Image.open(vein_path).convert("RGB")

    leaf_tensor = transform(leaf_img).unsqueeze(0).to(device)
    vein_tensor = transform(vein_img).unsqueeze(0).to(device)

    model = MultiModalFusion(num_classes=len(class_map)).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    with torch.no_grad():
        output = model(leaf_tensor, vein_tensor)
        pred_idx = output.argmax(1).item()

    pred_class = class_map[pred_idx]

    # to display two images + prediction
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(leaf_img)
    ax[0].set_title("Leaf Image")
    ax[0].axis("off")

    ax[1].imshow(vein_img)
    ax[1].set_title("Vein Image")
    ax[1].axis("off")

    plt.suptitle(f"Predicted Class: {pred_class}", fontsize=16)
    plt.show()

    return pred_class


if __name__ == "__main__":
    leaf_path = r"D:\Multimodal\leaf\healthy_99.jpg"
    vein_path = r"D:\Multimodal\veins_rgb\healthy_99_veins_rgb.png"
    checkpoint = r"weights\best_multimodal_fold5.pth" 

    class_map = {
        0: "healthy",
        1: "nitrogen",
        2: "potassium",
        3: "phosphorus",
        4: "sulphur",
        5: "zinc"
    }

    predict_multimodal(leaf_path, vein_path, checkpoint, class_map)
