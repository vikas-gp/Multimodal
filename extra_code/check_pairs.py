import os

def check_missing_pairs(leaf_dir, veins_dir, skeleton_dir):
    leaf_files = [f for f in os.listdir(leaf_dir) if f.lower().endswith('.jpg')]
    leaf_files.sort()

    veins_files = os.listdir(veins_dir)
    skeleton_files = os.listdir(skeleton_dir)

    veins_set = set(f.lower() for f in veins_files)
    skeleton_set = set(f.lower() for f in skeleton_files)

    for leaf_name in leaf_files:
        base_name = os.path.splitext(leaf_name)[0].lower()

        veins_name = f"{base_name}_veins_rgb.png"
        skeleton_name = f"{base_name}_skeleton.png"

        missing_veins = veins_name not in veins_set
        missing_skeleton = skeleton_name not in skeleton_set

        if missing_veins or missing_skeleton:
            print(f"{leaf_name} is missing pairs - Veins missing: {missing_veins}, Skeleton missing: {missing_skeleton}")

if __name__ == "__main__":
    leaf_directory = r'D:\Multimodal\leaf'
    veins_directory = r"D:\Multimodal\veins_rgb"
    skeleton_directory = r"D:\Multimodal\skeleton"

    check_missing_pairs(leaf_directory, veins_directory, skeleton_directory)
