import numpy as np

# Load preprocessed data
data = np.load("preprocessed_images.npz", allow_pickle=True)

base_names = data["base_names"]   # list/array of filenames IN THE CORRECT ORDER

# Define your class names
classes = ['healthy', 'Nitrogen', 'Potassium', 'Phosphorus', 'Zinc', 'Sulphur']
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

labels = []

for name in base_names:
    name_str = str(name)
    found = False
    for cls in classes:
        if cls.lower() in name_str.lower():     # case-insensitive match
            labels.append(class_to_idx[cls])
            found = True
            break
    if not found:
        raise ValueError(f"Could not find class in {name_str}")

labels = np.array(labels)

print("Generated labels shape:", labels.shape)
print("Example labels:", labels[:20])

np.save("labels.npy", labels)
print("Saved corrected labels.npy")
