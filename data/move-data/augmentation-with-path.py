from dataset import *
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil
import pandas as pd

# LUNA16
# Specify the paths to your data directories
image_path = "data/luna16/output3/fix/img_cropped_ori/"
mask_path = "data/luna16/output3/fix/masks_morph/"

# FEMH
# image_path = "data/femh/data474/roi/images/"
# mask_path = "data/femh/data474/roi/masks/"

# Get a list of all image files in the directories
image_files = os.listdir(image_path)
mask_files = os.listdir(mask_path)

# Use train_test_split to split the files into training, testing, and validation sets
image_train, image_test_valid, mask_train, mask_test_valid = train_test_split(
    image_files, mask_files, test_size=0.3, random_state=42
)
image_test, image_valid, mask_test, mask_valid = train_test_split(
    image_test_valid, mask_test_valid, test_size=0.5, random_state=42
)

# LUNA16
# Specify the output directories
train_dir_image = "data/split/luna16/data934/image-format/train_data/images/"
train_dir_mask = "data/split/luna16/data934/image-format/train_data/masks/"
test_dir_image = "data/split/luna16/data934/image-format/test_data/images/"
test_dir_mask = "data/split/luna16/data934/image-format/test_data/masks/"
valid_dir_image = "data/split/luna16/data934/image-format/val_data/images/"
valid_dir_mask = "data/split/luna16/data934/image-format/val_data/masks/"

# FEMH
# train_dir_image = "data/split/femh/data474/image-format/train_data/images/"
# train_dir_mask = "data/split/femh/data474/image-format/train_data/masks/"
# test_dir_image = "data/split/femh/data474/image-format/test_data/images/"
# test_dir_mask = "data/split/femh/data474/image-format/test_data/masks/"
# valid_dir_image = "data/split/femh/data474/image-format/val_data/images/"
# valid_dir_mask = "data/split/femh/data474/image-format/val_data/masks/"

# Create directories if they don't exist
for directory in [train_dir_image, train_dir_mask, test_dir_image, test_dir_mask, valid_dir_image, valid_dir_mask]:
    os.makedirs(directory, exist_ok=True)

# Move or copy files to the respective directories
def move_files(source_files, source_path, destination_path):
    for file in source_files:
        source_file_path = os.path.join(source_path, file)
        destination_file_path = os.path.join(destination_path, file)
        shutil.copy(source_file_path, destination_file_path)

# Move training set files
move_files(image_train, image_path, train_dir_image)
move_files(mask_train, mask_path, train_dir_mask)

# Move testing set files
move_files(image_test, image_path, test_dir_image)
move_files(mask_test, mask_path, test_dir_mask)

# Move validation set files
move_files(image_valid, image_path, valid_dir_image)
move_files(mask_valid, mask_path, valid_dir_mask)


############## save new test data Luna16 and filter the filename #########
image_path = "data/split/luna16/data934/image-format/test_data/images/"
mask_path = "data/split/luna16/data934/image-format/test_data/masks/"
annotations_csv = "data/luna16/measurement/fix-mask-annotation.csv"

csv_filename = "filenames_without_extension.csv"
# Get a list of filenames without extensions in the directory
mask_filenames = [os.path.splitext(os.path.basename(filename))[0] for filename in os.listdir(mask_path)]
# Print the list of filenames without extensions
print(mask_filenames)
# Write the filenames to a CSV file
# with open(csv_filename, mode='w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(['Filenames Without Extension'])
#     csv_writer.writerows(map(lambda x: [x], mask_filenames))
# print(f"Filenames without extension saved to {csv_filename}")

# Read the full data CSV into a DataFrame
full_data = pd.read_csv(annotations_csv)

# Filter the DataFrame based on selected filenames
filtered_data = full_data[full_data['id'].isin(mask_filenames)]

# Print or do further operations with the filtered data
print(filtered_data)

# Save the filtered data to a new CSV file
filtered_data.to_csv("data/luna16/measurement/filtered_data.csv", index=False)
############## end #########