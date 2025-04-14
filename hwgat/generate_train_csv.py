import os
import csv
import random
from math import ceil
 
# Define paths
root = r"C:\\Sreelakshmi V\\sl-hwgat-main"
data_path = os.path.join(root, "INCLUDE\\")
split_path = os.path.join(root, "Train_Test_Split")
 
# Create output directory if not exists
os.makedirs(split_path, exist_ok=True)

print(data_path, split_path, sep="\n")

# Train-test split ratio
test_split = 0.2  # 20% for testing

# Collect all video files
video_files = []
for root_dir, sub_dirs, files in os.walk(data_path):
    # print(root_dir, sub_dirs, files)
    for file in files:
        if file.endswith((".mp4", ".avi", ".mov",".MP4", ".AVI", ".MOV")):  # Add other video formats if needed
            video_files.append(os.path.join(root_dir, file))
            # print(video_files, "This is a video file \n")
 
# print(video_files)

# Shuffle dataset randomly
random.shuffle(video_files)
 
# Determine split index
split_index = ceil(len(video_files) * (1 - test_split))
 
# Split into training and testing sets
train_files = video_files[:split_index]
test_files = video_files[split_index:]

# print(f"\n Train Files: {train_files} \n Test Files{test_files}")
 
# Function to extract class label from file path
def extract_class_label(file_path):
    parts = file_path.split(os.sep)  # Split path by backslash
    if "Extra" in parts or "extra" in parts:
        return parts[-3].split('.')[1].strip().lower()  # Extract class from folder
    return parts[-2].split('.')[1].strip().lower()
 
# Save CSV function
def save_csv(file_list, filename):
    with open(os.path.join(split_path, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "dataset", "video_name", "video_path"])  # Header
        for idx, file in enumerate(file_list):
            class_label = extract_class_label(file)
            # print(class_label)
            relative_path = os.path.relpath(file, root)  # Convert to relative path
            writer.writerow([idx, "INCLUDE", os.path.basename(file), relative_path])
 
# Save train and test CSV files
save_csv(train_files, "train_include.csv")
save_csv(test_files, "test_include.csv")
 
print("Train/Test CSV files created successfully!")