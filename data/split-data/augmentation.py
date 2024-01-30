from dataset import *
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Prepare dataset for training
# FEMH
# image_path = "data/femh/data474/roi/images/"
# mask_path = "data/femh/data474/roi/masks/"

# LUNA16
image_path = "data/luna16/output3/fix/img_cropped_ori/"
mask_path = "data/luna16/output3/fix/masks_morph/"

# convert image to numpy array
images_set, masks_set = read_dataset(image_path, mask_path)
images_set = np.array(images_set)
masks_set = np.array(masks_set)

# display samples of image and mask in gray scale
fig, ax = plt.subplots(1, 2)
ax[0].imshow(images_set[0].squeeze(), cmap='gray') 
ax[1].imshow(masks_set[0].squeeze(), cmap='gray')


# def split_train_test ():
#     # Assuming you have your data in X (features) and y (labels/masks)
#     x_train, x_val, y_train, y_val = train_test_split(images_set, masks_set, test_size=0.2, random_state=42)
#     return x_train, x_val, y_train, y_val

##################Split dataset into training, validation, and testing
# Split the dataset into training (70%) and temporary (30%) sets
x_train_temp, x_temp, y_train_temp, y_temp = train_test_split(images_set, masks_set, test_size=0.3, random_state=42)

# Split the temporary set into validation (50%) and testing (50%) sets
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
# Now, you have x_train_temp, y_train_temp for training (70%), x_val, y_val for validation (15%), and x_test, y_test for testing (15%).
# You can rename x_train_temp and y_train_temp to x_train and y_train if needed:
x_train = x_train_temp
y_train = y_train_temp

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

# # FEMH
# # # Create directories
# os.makedirs('data/split/femh/data474/original/train_data', exist_ok=True)
# os.makedirs('data/split/femh/data474/original/val_data', exist_ok=True)
# os.makedirs('data/split/femh/data474/original/test_data', exist_ok=True)

# # # Save data
# np.save('data/split/femh/data474/original/train_data/x_train.npy', x_train)
# np.save('data/split/femh/data474/original/train_data/y_train.npy', y_train)
# np.save('data/split/femh/data474/original/val_data/x_val.npy', x_val)
# np.save('data/split/femh/data474/original/val_data/y_val.npy', y_val)
# np.save('data/split/femh/data474/original/test_data/x_test.npy', x_test)
# np.save('data/split/femh/data474/original/test_data/y_test.npy', y_test)
# ##################################

#LUNA16
# # Create directories
os.makedirs('data/split/luna16/data934/original/train_data', exist_ok=True)
os.makedirs('data/split/luna16/data934/original/val_data', exist_ok=True)
os.makedirs('data/split/luna16/data934/original/test_data', exist_ok=True)

# # Save data
np.save('data/split/luna16/data934/original/train_data/x_train.npy', x_train)
np.save('data/split/luna16/data934/original/train_data/y_train.npy', y_train)
np.save('data/split/luna16/data934/original/val_data/x_val.npy', x_val)
np.save('data/split/luna16/data934/original/val_data/y_val.npy', y_val)
np.save('data/split/luna16/data934/original/test_data/x_test.npy', x_test)
np.save('data/split/luna16/data934/original/test_data/y_test.npy', y_test)
##################################
