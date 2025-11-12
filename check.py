import os

leaf_dir = r"/teamspace/studios/this_studio/Multimodal/leaf"
files = sorted(os.listdir(leaf_dir))

print(f"Total files: {len(files)}")
print("Sample file names:")
for f in files[:10]:
    print(f)
