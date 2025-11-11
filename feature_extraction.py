import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm, JumpingKnowledge, SAGPooling
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
import networkx as nx
import pickle
from skimage.morphology import skeletonize

# --- Leaf Feature Extraction ---

data = np.load('preprocessed_images.npz')
leaf_imgs_np = data['leaf_imgs']  # (N, H, W, 3)

leaf_imgs_tensor = torch.tensor(leaf_imgs_np).permute(0, 3, 1, 2).float()

normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

leaf_imgs_tensor = torch.stack([normalize_transform(img) for img in leaf_imgs_tensor])

dataset = TensorDataset(leaf_imgs_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
densenet.classifier = nn.Identity()
densenet.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
densenet.to(device)

features_list = []
with torch.no_grad():
    for (batch,) in dataloader:
        batch = batch.to(device)
        features = densenet(batch)
        features_list.append(features.cpu())

leaf_features = torch.cat(features_list, dim=0)
print(f"Leaf features shape: {leaf_features.shape}")

# --- Helper: Convert NetworkX graph to PyG Data ---
def nx_to_pyg_data(G):
    if 'pos' in list(G.nodes(data=True))[0][1]:
        x = torch.tensor([G.nodes[n]['pos'] for n in G.nodes], dtype=torch.float)
    else:
        x = torch.ones((G.number_of_nodes(),1), dtype=torch.float)
    edges = list(G.edges)
    edges += [(v, u) for u, v in edges if (v, u) not in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# --- Robust Vein GNN Model ---
class RobustVeinGNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=128, heads=4, dropout=0.3):
        super(RobustVeinGNN, self).__init__()
        self.emb = nn.Linear(input_dim, hidden_dim)

        self.conv1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim)

        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim)

        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn3 = BatchNorm(hidden_dim)

        self.jk = JumpingKnowledge(mode='cat')

        # Fixed input dimension here - JumpingKnowledge concatenates 3 * hidden_dim
        self.pool = SAGPooling(hidden_dim * 3, ratio=0.5)

        self.dropout = dropout

        # Fixed input dimension to match pooling output (hidden_dim * 3)
        self.fc = nn.Linear(hidden_dim * 3, output_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(self.emb(x), edge_index)))
        x1 = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x2 = x
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x3 = x

        x = self.jk([x1, x2, x3])

        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)

# Load skeleton graphs
with open('skeleton_graphs.pkl', 'rb') as f:
    skeleton_graphs = pickle.load(f)

vein_graphs_data = [nx_to_pyg_data(g) for g in skeleton_graphs]
vein_loader = GeoDataLoader(vein_graphs_data, batch_size=32)

vein_gnn = RobustVeinGNN().to(device)
vein_gnn.eval()

vein_features_list = []
with torch.no_grad():
    for data in vein_loader:
        data = data.to(device)
        feats = vein_gnn(data.x, data.edge_index, data.batch)
        vein_features_list.append(feats.cpu())

vein_features = torch.cat(vein_features_list, dim=0)
print(f"Vein graph features shape: {vein_features.shape}")

# Save features
np.save('leaf_features.npy', leaf_features.numpy())
np.save('vein_features.npy', vein_features.numpy())
print("Saved leaf and vein features to disk.")
