import os
import shutil
from pathlib import Path

# Define source and target base directories
source_dirs = [
    "/home/wuaodi/projects/PMF/data/data_odometry_velodyne/dataset/sequences",
    "/home/wuaodi/projects/PMF/data/data_odometry_labels/dataset/sequences",
    "/home/wuaodi/projects/PMF/data/data_odometry_color/dataset/sequences"
]
target_base_dir = "/home/wuaodi/projects/PMF/data/semantic-kitty/sequences"

# Ensure target base directory exists
os.makedirs(target_base_dir, exist_ok=True)

# Process sequences 00 to 21
for seq in range(22):  # 0 to 21 inclusive
    seq_str = f"{seq:02d}"  # Format as two-digit string (00, 01, ..., 21)
    target_seq_dir = os.path.join(target_base_dir, seq_str)

    # Create target sequence directory if it doesn't exist
    os.makedirs(target_seq_dir, exist_ok=True)

    # Iterate through each source directory
    for src_dir in source_dirs:
        src_seq_dir = os.path.join(src_dir, seq_str)

        # Check if source sequence directory exists
        if os.path.exists(src_seq_dir):
            # Iterate through all files in the source sequence directory
            for item in os.listdir(src_seq_dir):
                src_item_path = os.path.join(src_seq_dir, item)
                target_item_path = os.path.join(target_seq_dir, item)

                # Skip if target item already exists to avoid overwriting
                if os.path.exists(target_item_path):
                    print(f"Warning: {target_item_path} already exists, skipping {src_item_path}")
                    continue

                # Move the item (file or directory)
                try:
                    shutil.move(src_item_path, target_item_path)
                    print(f"Moved {src_item_path} to {target_item_path}")
                except Exception as e:
                    print(f"Error moving {src_item_path} to {target_item_path}: {e}")

print("Done moving all sequence contents.")