from utils import *

# Prepare dataset for training
image_path = "data/femh/data474/images/"
mask_path = "data/femh/data474/masks/"
list_roi = "data/femh/data474/list-roi.csv"

binary_path = "data/femh/data474/roi/binary_mask/"
roi_images = 'data/femh/data474/roi/images/'
roi_masks = 'data/femh/data474/roi/masks/'

# ROI (dont use if data has saved)
desired_roi_size = 64  # Set to 64x64
binary_mask(mask_path, binary_path)
save_image_mask_roi(image_path, binary_path, roi_images, roi_masks, list_roi, desired_roi_size)
# save (optional)