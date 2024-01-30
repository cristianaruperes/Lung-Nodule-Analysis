from model import *
import numpy as np
import matplotlib.pyplot as plt
from evaluate import *
from dataset import *
from visualize import *
import pandas as pd
from scipy.stats import pearsonr
import os
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Prepare dataset for training
# image_path = "data/luna16/output2/img_cropped_ori/"
# mask_path = "data/luna16/output2/masks_morph/"
# annotations_csv = "data/luna16/fix-mask-annotation1.csv"


image_path = "../data-thesis/split/luna16/data934/image-format/test_data/images/"
mask_path = "../data-thesis/split/luna16/data934/image-format/test_data/masks/"
annotations_csv = "../data-thesis/split/luna16/data934/image-format/test_data/full-data.csv"


# Read the CSV file into a DataFrame
annotations_df = pd.read_csv(annotations_csv)

# convert image to numpy array
images_set, masks_set = read_dataset(image_path, mask_path)
images_set = np.array(images_set)
masks_set = np.array(masks_set)

print(len(images_set))

# display samples of image and mask in gray scale
fig, ax = plt.subplots(1, 2)
ax[0].imshow(images_set[2].squeeze(), cmap='gray') 
ax[1].imshow(masks_set[2].squeeze(), cmap='gray')

# Apply CLAHE to x_train
x_test_clahe = np.array([apply_clahe(image)[:, :, np.newaxis] for image in tqdm(images_set, desc="Applying CLAHE to x_train")])

# display_clahe(images_set, x_test_clahe, 3)

x_test = x_test_clahe
y_test = masks_set

# try:
#     loaded_model = keras.models.load_model("unet1.hdf5")
# except Exception as e:
#     print(f"An error occurred: {str(e)}")


# Clear the previous TensorFlow session and model
tf.keras.backend.clear_session()

# Load model
# ck_path = "models/mix/validation/unet/unet-fix-aug_fold2.hdf5"
# ck_path = "models/mix/validation/segnet/segnet-fix-aug_fold5.hdf5"
# ck_path = "models/mix/validation/nested/nested-fix-aug_fold2.hdf5"
# ck_path = "models/mix/validation/fcn/fcn-fix-aug_fold3.hdf5"
# ck_path = "models/mix/validation/ca-unet/ca-unet-fix-aug_fold5.hdf5"
ck_path = "models/mix/validation/sa-unet/sa-unet-fix-aug_fold2.hdf5"

input_shape = (64, 64, 1)

# UNET
# model = unet(input_shape)

#segnet
# model = segnet(input_shape, num_classes=1)

#Unet++
# model = nested_unet(input_size=(64, 64, 1), num_filters=32)

#Attention unet
# model = attention_unet(input_size=(64, 64, 1))

#FCN
# model = fcn_model(input_size=(64, 64, 1))

 # ca-unet
# model = ca_unet(input_shape)

# sa-unet
model = sa_unet(input_shape)

model.load_weights(ck_path)

# #Prediction
y_pred = model.predict(x_test)

print("mix fold 2 sa-unet")

# Choose an index for the image you want to visualize
image_index = 2  # Replace with the index of the image you want to visualize

# Get the original image, ground truth mask, and predicted mask for the selected index
original_image = x_test[image_index]
ground_truth_mask = y_test[image_index]
predicted_mask = y_pred[image_index]

display_image_comparison(original_image, ground_truth_mask, predicted_mask)

annotations_df.head()

for index, row in annotations_df.iterrows():
    v_diameter = row['diameter_mm']  # Assuming 'v_diameter' is the column name in your DataFrame
    pixel_size_mm = fix_array_from_mhd_file(row['space'])
    pixel_size_mm = pixel_size_mm[0]

    nodule_area_mm = calculate_nodule_area_in_mm(v_diameter)

    # Calculate nodule area in pixels for y_pred_image (modify as needed)
    y_pred_image = y_pred[index]
    y_test_image = y_test[index]
    nodule_area_mm_pred = calculate_nodule_area_in_mm_predict(y_pred_image, pixel_size_mm)
    nodule_area_mm_gt = calculate_nodule_area_in_mm_gt(y_test_image, pixel_size_mm)

    # You can do something with nodule_area_pixels here, such as adding it to a new column in the DataFrame
    annotations_df.at[index, 'nodule_area_mm'] = nodule_area_mm
    annotations_df.at[index, 'nodule_area_mm_pred'] = nodule_area_mm_pred
    annotations_df.at[index, 'nodule_area_mm_gt'] = nodule_area_mm_gt

# for index, row in annotations_df.iterrows():
#     # 1 pixel ? mm ?
#     pixel_size_gt_mm = row['v_diameters'] / row['nodule_area_pixels'] if row['nodule_area_pixels']!=0 else 0
#     pixel_size_pred_mm = row['v_diameters'] / row['nodule_area_pixels_pred'] if row['nodule_area_pixels_pred']!=0 else 0

#     # Convert the nodule area from square pixels to square millimeters
#     nodule_area_mm2_gt = row['nodule_area_pixels'] * (pixel_size_gt_mm ** 2)
#     nodule_area_mm2_pred = row['nodule_area_pixels_pred'] * (pixel_size_pred_mm ** 2)

#     annotations_df.at[index, 'nodule_area_milimeter_gt'] = nodule_area_mm2_gt
#     annotations_df.at[index, 'nodule_area_milimeter_pred'] = nodule_area_mm2_pred

#     print(f"Nodule area in square ground truth millimeters: {nodule_area_mm2_gt:.2f} mm^2")
#     print(f"Nodule area in square prediction millimeters: {nodule_area_mm2_pred:.2f} mm^2")


annotations_df.head()

#save csv file
# file_path = 'nodule_area7.csv'
file_path = 'file-measurement/mix/validation/saunet-fold2.csv'

# Check if the file exists
if os.path.exists(file_path):
    print(f"Warning: File '{file_path}' does exist.")
else:
    annotations_df.to_csv(file_path, index=False)     

load_noduleArea = annotations_df = pd.read_csv(file_path)
load_noduleArea.head()

print(load_noduleArea.shape)

#Ground truth

# Sample real ground truth and predicted values in millimeters (replace with your data)
real_ground_truth_mm_gt = load_noduleArea['nodule_area_mm_gt']  # Millimeters
predicted_values_mm_gt = load_noduleArea['nodule_area_mm_pred'] # Millimeters

# Calculate Pearson's correlation coefficient
correlation_coefficient_mm_gt, _ = pearsonr(real_ground_truth_mm_gt, predicted_values_mm_gt)

# Create a scatter plot
plt.scatter(real_ground_truth_mm_gt, predicted_values_mm_gt, color='b', label=f'Correlation: {correlation_coefficient_mm_gt:.2f} (mm²)')

# Add labels and legend
plt.xlabel('Real Ground Truth (mm²)')
plt.ylabel('Predicted Values (mm²)')

# Show the center line (perfect correlation)
plt.plot([min(real_ground_truth_mm_gt), max(real_ground_truth_mm_gt)],
         [min(real_ground_truth_mm_gt), max(real_ground_truth_mm_gt)],
         linestyle='--', color='gray', label='Center Line (Perfect Correlation)')

plt.legend()

# Show the plot
plt.title('Scatter Plot of Pearson\'s Correlation (mm²) with Center Line')

plt.show()

# Compare with diameter

# Sample real ground truth and predicted values in millimeters (replace with your data)
real_ground_truth_mm_dm = load_noduleArea['nodule_area_mm']  # Millimeters
predicted_values_mm_dm = load_noduleArea['nodule_area_mm_pred'] # Millimeters

# Calculate Pearson's correlation coefficient
correlation_coefficient_mm_dm, _ = pearsonr(real_ground_truth_mm_dm, predicted_values_mm_dm)

# Create a scatter plot
plt.scatter(real_ground_truth_mm_dm, predicted_values_mm_dm, color='b', label=f'Correlation: {correlation_coefficient_mm_dm:.2f} (mm²)')

# Add labels and legend
plt.xlabel('Real Ground Truth (mm²)')
plt.ylabel('Predicted Values (mm²)')

# Show the center line (perfect correlation)
plt.plot([min(real_ground_truth_mm_dm), max(real_ground_truth_mm_dm)],
         [min(real_ground_truth_mm_dm), max(real_ground_truth_mm_dm)],
         linestyle='--', color='gray', label='Center Line (Perfect Correlation)')

plt.legend()

# Show the plot
plt.title('Scatter Plot of Pearson\'s Correlation (mm) with Center Line (Diameter)')

# plt.show()

# Calculate pcc and rmse
pcc_gt = calculate_pcc(real_ground_truth_mm_gt, predicted_values_mm_gt)
rmse_gt = calculate_rmse(real_ground_truth_mm_gt, predicted_values_mm_gt)

print(f'(GT mask) Pearson Correlation Coefficient (PCC): {pcc_gt:.2f}')
print(f'(GT mask) Root Mean Square Error (RMSE): {rmse_gt:.2f}')

# Calculate pcc and rmse
pcc_dm = calculate_pcc(real_ground_truth_mm_dm, predicted_values_mm_dm)
rmse_dm = calculate_rmse(real_ground_truth_mm_dm, predicted_values_mm_dm)

print(f'(Diameter) Pearson Correlation Coefficient (PCC): {pcc_dm:.2f}')
print(f'(Diameter) Root Mean Square Error (RMSE): {rmse_dm:.2f}')

# display samples of image and predicted in gray scale
fig, ax = plt.subplots(1, 2)
ax[0].imshow(images_set[83].squeeze(), cmap='gray') 
ax[1].imshow(y_pred[83].squeeze(), cmap='gray')

threshold = 0.5
# Threshold the predicted mask for the current image

binary_mask = (y_pred[83] > threshold).astype(np.uint8)
# Count the number of nodule pixels in the binary mask
nodule_pixel_count = np.sum(binary_mask)
print(nodule_pixel_count)
print(pixel_size_mm)
# Calculate the area in square millimeters
area_mm2 = nodule_pixel_count * (pixel_size_mm ** 2)
print(area_mm2)