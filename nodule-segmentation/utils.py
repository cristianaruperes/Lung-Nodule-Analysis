import os
from skimage import io
import cv2
import pandas as pd
import numpy as np

def binary_mask(data, output_folder):
    nodules_masks = []

    for filename in os.listdir(data):
        # print(filename)
        nodules_mask = io.imread(os.path.join(data, filename))
        nodules_mask = np.where(nodules_mask > 0, 255, 0)  # Convert to binary mask (using 255 for foreground)
        nodules_masks.append(nodules_mask)
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_folder, output_filename)
        io.imsave(output_path, nodules_mask.astype(np.uint8))

    all_nodules_masks = np.concatenate(nodules_masks, axis=0)
    print("binary mask has been saved")

def save_image_mask_roi(image_dir, mask_dir, output_dir, output_dir_mask, file, size):
    csv_data = pd.read_csv(file)

    for _, row in csv_data.iterrows():
        image_filename = str(row['Index'])  # Convert to string
        x = row['X']
        y = row['Y']

        # Construct full paths to image and mask files
        image_path = os.path.join(image_dir, image_filename + ".jpg")
        mask_path = os.path.join(mask_dir, image_filename + ".png")

        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Calculate half of the desired ROI size
        half_roi_size = size // 2

        # Check if the extraction window exceeds the image boundaries
        if (
            x - half_roi_size < 0
            or x + half_roi_size >= image.shape[1]
            or y - half_roi_size < 0
            or y + half_roi_size >= image.shape[0]
        ):
            print(f"Warning: ROI extraction at ({image_filename}, {x}, {y}) is too close to the image boundary.")
            continue  # Skip this ROI if it's too close to the boundary

        # Extract the ROI from the image and mask
        roi_image = image[y - half_roi_size : y + half_roi_size, x - half_roi_size : x + half_roi_size]
        roi_mask = mask[y - half_roi_size : y + half_roi_size, x - half_roi_size : x + half_roi_size]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_mask, exist_ok=True)

        # Save the ROI and its corresponding mask to the output directory
        output_image_path = os.path.join(output_dir, f'{os.path.basename(image_filename + ".jpg")}')
        output_mask_path = os.path.join(output_dir_mask, f'{os.path.basename(image_filename + ".png")}')
        cv2.imwrite(output_image_path, roi_image)
        cv2.imwrite(output_mask_path, roi_mask)