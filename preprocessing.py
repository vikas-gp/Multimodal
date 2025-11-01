import os
import cv2
import numpy as np
import networkx as nx
import pickle
from skimage.morphology import skeletonize

def normalize_image(img):
    arr = img.astype(np.float32) / 255.0
    return arr

def preprocess_image(img_path, size=(224, 224), is_gray=False):
    if is_gray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV) to RGB

    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    img_norm = normalize_image(img_resized)
    return img_norm

def skeleton_image_to_graph(skel_img):
    skel_bin = (skel_img > 0).astype(np.uint8)
    skel_thinned = skeletonize(skel_bin)

    coords = np.column_stack(np.where(skel_thinned))
    G = nx.Graph()
    for idx, (y, x) in enumerate(coords):
        G.add_node(idx, pos=(x, y))
    for idx1, (y1, x1) in enumerate(coords):
        for idx2, (y2, x2) in enumerate(coords):
            if idx1 != idx2 and abs(x1-x2) <= 1 and abs(y1-y2) <= 1:
                G.add_edge(idx1, idx2)
    return G

def preprocess_dataset(paired_samples, image_size=(224, 224)):
    dataset = []
    for sample in paired_samples:
        leaf_arr = preprocess_image(sample['leaf_path'], size=image_size, is_gray=False)
        veins_arr = preprocess_image(sample['veins_path'], size=image_size, is_gray=False)
        skeleton_arr = preprocess_image(sample['skeleton_path'], size=image_size, is_gray=True)
        skeleton_graph = skeleton_image_to_graph(cv2.resize(cv2.imread(sample['skeleton_path'], cv2.IMREAD_GRAYSCALE), image_size, interpolation=cv2.INTER_LANCZOS4))

        dataset.append({
            'leaf_img': leaf_arr,
            'veins_img': veins_arr,
            'skeleton_img': skeleton_arr,
            'skeleton_graph': skeleton_graph,
            'base_name': sample['base_name']
        })
    return dataset

def load_paired_dataset(leaf_dir, veins_rgb_dir, skeleton_dir):
    paired_samples = []
    leaf_files = [f for f in os.listdir(leaf_dir) if f.lower().endswith('.jpg')]
    leaf_files.sort()

    veins_files = os.listdir(veins_rgb_dir)
    skeleton_files = os.listdir(skeleton_dir)

    veins_dict = {f.lower(): f for f in veins_files}
    skeleton_dict = {f.lower(): f for f in skeleton_files}

    for leaf_name in leaf_files:
        base_name = os.path.splitext(leaf_name)[0].lower()

        veins_name_lower = f"{base_name}_veins_rgb.png"
        skeleton_name_lower = f"{base_name}_skeleton.png"

        if veins_name_lower not in veins_dict:
            print(f"Veins image missing for {leaf_name}: expected {veins_name_lower}")
            continue
        if skeleton_name_lower not in skeleton_dict:
            print(f"Skeleton image missing for {leaf_name}: expected {skeleton_name_lower}")
            continue

        veins_name = veins_dict[veins_name_lower]
        skeleton_name = skeleton_dict[skeleton_name_lower]

        leaf_path = os.path.join(leaf_dir, leaf_name)
        veins_path = os.path.join(veins_rgb_dir, veins_name)
        skeleton_path = os.path.join(skeleton_dir, skeleton_name)

        paired_samples.append({
            'leaf_path': leaf_path,
            'veins_path': veins_path,
            'skeleton_path': skeleton_path,
            'base_name': base_name
        })

    print(f"Paired samples loaded: {len(paired_samples)}")
    return paired_samples


# Main execution
if __name__ == "__main__":
    leaf_dir = r"D:\Multimodal\leaf"
    veins_rgb_dir = r"D:\Multimodal\veins_rgb"
    skeleton_dir = r"D:\Multimodal\skeleton"

    paired_samples = load_paired_dataset(leaf_dir, veins_rgb_dir, skeleton_dir)
    preprocessed_data = preprocess_dataset(paired_samples, image_size=(224, 224))
    print(f"Total preprocessed samples: {len(preprocessed_data)}")

    # Save preprocessed images
    np.savez('preprocessed_images.npz',
             leaf_imgs=np.array([d['leaf_img'] for d in preprocessed_data]),
             veins_imgs=np.array([d['veins_img'] for d in preprocessed_data]),
             skeleton_imgs=np.array([d['skeleton_img'] for d in preprocessed_data]),
             base_names=[d['base_name'] for d in preprocessed_data]
    )

    # Save skeleton graphs as pickled objects
    with open('skeleton_graphs.pkl', 'wb') as f:
        pickle.dump([d['skeleton_graph'] for d in preprocessed_data], f)

    print("Saved preprocessed images and skeleton graphs for future use.")
