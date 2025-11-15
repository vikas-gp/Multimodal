import numpy as np

data = np.load("preprocessed_images.npz")

print("Keys:", data.files)

# If filenames exist:
if "filenames" in data.files:
    print(data["filenames"][:10])


labels = np.load("labels.npy")
print("Unique labels:", np.unique(labels))
print("Counts:", np.bincount(labels))
print("Total:", len(labels))



leaf = np.load("leaf_features.npy")
vein = np.load("vein_features.npy")

print("Leaf features shape:", leaf.shape)
print("Vein features shape:", vein.shape)

if leaf.shape[0] == vein.shape[0]:
    print(" Feature counts MATCH. Safe for multimodal fusion.")
else:
    print(" Feature counts DO NOT MATCH. Need to fix ordering.")