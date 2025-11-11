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
