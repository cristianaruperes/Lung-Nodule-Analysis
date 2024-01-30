from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os

IMG_SIZE = 64

def read_train_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = img / 255.0  # Scale pixel values to [0, 1]
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=-1)
    return img_arr

def read_train_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    # mask = cv2.resize(mask, (64, 64), cv2.INTER_AREA)
    mask = mask/255.0
    mask = np.array(mask)
    mask_arr = np.expand_dims(mask, axis=-1)
    return mask_arr

def read_test_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = img / 255.0  # Scale pixel values to [0, 1]
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=-1)
    return img_arr

def rotate_by_angle(image, mask, rotation_angle):
    # Rotate the image
    rotated_image = np.array(Image.fromarray(image.squeeze()).rotate(rotation_angle))
    # Rotate the mask (if using binary masks)
    rotated_mask = np.array(Image.fromarray(mask.squeeze()).rotate(rotation_angle))
    return rotated_image, rotated_mask

def rotate_image_and_mask(image, mask):
    rotated_x_data = []
    rotated_y_data = []
    rotation_angles = [90, 180, 270]  # You can adjust the angles as needed

    for i in range(image.shape[0]):
        original_image = image[i]
        original_mask = mask[i]

        for angle in rotation_angles:
            rotated_image, rotated_mask = rotate_by_angle(original_image, original_mask, angle)
            rotated_x_data.append(rotated_image)
            rotated_y_data.append(rotated_mask)

    # Convert lists to NumPy arrays
    rotated_x_data = np.array(rotated_x_data)
    rotated_y_data = np.array(rotated_y_data)
    rotated_x_data = np.expand_dims(rotated_x_data, axis=-1)
    rotated_y_data = np.expand_dims(rotated_y_data, axis=-1)

    return rotated_x_data, rotated_y_data

# Function to apply horizontal flip to an image and mask
def flip_image_and_mask_horizontal(image, mask):
    flipped_image = np.fliplr(image)
    flipped_mask = np.fliplr(mask)
    return flipped_image, flipped_mask

# Function to apply vertical flip to an image and mask
def flip_image_and_mask_vertical(image, mask):
    flipped_image = np.flipud(image)
    flipped_mask = np.flipud(mask)
    return flipped_image, flipped_mask

def flip_image(image, mask):
    # Augment the dataset with flips
    flipped_image = []
    flipped_mask = []

    for i in range(image.shape[0]):
        original_image = image[i]
        original_mask = mask[i]

        # Apply horizontal and vertical flips to the original image and mask
        flipped_horizontal_image, flipped_horizontal_mask = flip_image_and_mask_horizontal(original_image, original_mask)
        flipped_vertical_image, flipped_vertical_mask = flip_image_and_mask_vertical(original_image, original_mask)

        flipped_image.extend([flipped_horizontal_image, flipped_vertical_image])
        flipped_mask.extend([flipped_horizontal_mask, flipped_vertical_mask])
        
    # Convert lists to NumPy arrays
    flipped_image = np.array(flipped_image)
    flipped_mask = np.array(flipped_mask)

    return flipped_image, flipped_mask

def augmentation_data(images_set, masks_set, rotated_images, rotated_masks, flipped_images, flipped_masks):
    images_augmented = np.concatenate((images_set, rotated_images), axis=0)
    masks_augmented = np.concatenate((masks_set, rotated_masks), axis=0)

    images_augmented = np.concatenate((images_augmented, flipped_images), axis=0)
    masks_augmented = np.concatenate((masks_augmented, flipped_masks), axis=0)

    return images_augmented, masks_augmented

def augmentation_data_only_flip(images_set, masks_set, flipped_images, flipped_masks):
    images_augmented = np.concatenate((images_set, flipped_images), axis=0)
    masks_augmented = np.concatenate((masks_set, flipped_masks), axis=0)

    return images_augmented, masks_augmented

def display_images_augmented (images_set, masks_set):
    # Select a specific image from your dataset
    image_index = 1  # Change this to the index of the image you want to visualize
    rotation_angles = [90, 180, 270]  # You can adjust the angles as needed
    
    # Create subplots to display the rotated and flipped images
    plt.figure(figsize=(18, 24))

    for i, angle in enumerate(rotation_angles):
        rotated_image, rotated_mask = rotate_by_angle(images_set[image_index], masks_set[image_index], angle)
        
        plt.subplot(4, len(rotation_angles), i + 1)
        plt.title(f'Rotated Image {angle}°')
        plt.imshow(rotated_image.squeeze(), cmap='gray')

        plt.subplot(4, len(rotation_angles), i + len(rotation_angles) + 1)
        plt.title(f'Rotated Mask {angle}°')
        plt.imshow(rotated_mask.squeeze(), cmap='gray')

        flipped_horizontal_image, flipped_horizontal_mask = flip_image_and_mask_horizontal(rotated_image, rotated_mask)
        plt.subplot(4, len(rotation_angles), i + 2*len(rotation_angles) + 1)
        plt.title(f'Flipped Horizontal Image {angle}°')
        plt.imshow(flipped_horizontal_image.squeeze(), cmap='gray')

        plt.subplot(4, len(rotation_angles), i + 3*len(rotation_angles) + 1)
        plt.title(f'Flipped Horizontal Mask {angle}°')
        plt.imshow(flipped_horizontal_mask.squeeze(), cmap='gray')

        # flipped_vertical_image, flipped_vertical_mask = flip_image_and_mask_vertical(rotated_image, rotated_mask)
        # plt.subplot(4, len(rotation_angles), i + 2*len(rotation_angles) + 1)
        # plt.title(f'Flipped Vertical Image {angle}°')
        # plt.imshow(flipped_vertical_image.squeeze(), cmap='gray')

        # plt.subplot(4, len(rotation_angles), i + 3*len(rotation_angles) + 1)
        # plt.title(f'Flipped Vertical Mask {angle}°')
        # plt.imshow(flipped_vertical_mask.squeeze(), cmap='gray')

    plt.tight_layout()
    plt.show()

def read_dataset(image_path, mask_path):
    """
    image_path : path to stored images
    mask_path : path to stored masks
    """
    image_list = sorted(glob(image_path + "*"))
    mask_list = sorted(glob(mask_path + "*"))
    images = [] # list of training images
    masks = [] # list of training masks

    for img_file in tqdm(image_list):
        image = read_train_image(img_file)
        images.append(image)
    for mask_file in tqdm(mask_list):
        mask = read_train_mask(mask_file)
        masks.append(mask)
    return images, masks

def count_nodule_pixels(pred_mask, org_height, org_width):
        """
        # for binary mask only, with balck color is background and white color is nodule
        pred_mask: your prediction mask
        org_height: original height of image
        org_width: original width of your image
        """
        predicted_mask = pred_mask.flatten()>0.5
        num = np.sum(predicted_mask==True)
        num = num * (org_width*org_height/(512*512))  
        print("Predicted Number wound pixels in images: {}".format(num))

# def display_sample(display_list, title = ['Input Image', 'True Mask', 'Predicted Mask']):
#     """Show side-by-side an input image,
#     the ground truth and the prediction.
#     """
#     plt.figure(figsize=(15, 15))

#     cmaps = [None, plt.cm.binary_r]
#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i+1)
#         plt.title(title[i])
#         if i==0:
#             plt.imshow(display_list[i], cmap=cmaps[0])
#         else:
#             plt.imshow(display_list[i], cmap=cmaps[1])
#         plt.axis('off')
#     plt.show()

# def count_wound_size(wound_px, px_size):
#     """
#     wound_px: number of pixels belong to the wound in prediction
#     px_size : size of 1 pixel from finger nail model
    
#     """
#     return wound_px * px_size

def calculate_nodule_area_in_mm(v_diam):
    # Calculate the area of the nodule in square pixels
    nodule_area_mm = np.pi * ((v_diam / 2.0) ** 2)

    return nodule_area_mm

def calculate_nodule_area_in_mm_predict(y_pred_image, pixel_size_mm):
    threshold = 0.5
    # Threshold the predicted mask for the current image

    binary_mask = (y_pred_image > threshold).astype(np.uint8)
    # Count the number of nodule pixels in the binary mask
    nodule_pixel_count = np.sum(binary_mask)
    # Calculate the area in square millimeters
    area_mm2 = nodule_pixel_count * (pixel_size_mm ** 2)

    return area_mm2

def calculate_nodule_area_in_mm_gt(gt_image, pixel_size_mm):
    threshold = 0.5
    # Threshold the predicted mask for the current image

    binary_mask = (gt_image > threshold).astype(np.uint8)
    # Count the number of nodule pixels in the binary mask
    nodule_pixel_count = np.sum(binary_mask)
    # print(nodule_pixel_count)
    # print(pixel_size_mm)
    # Calculate the area in square millimeters
    area_mm2 = nodule_pixel_count * (pixel_size_mm ** 2)

    return area_mm2

def save_predicted_images(y_pred, output_dir):
    """
    Save a list of predicted images to a directory.

    Args:
        y_pred (list of numpy arrays): List of predicted images.
        output_dir (str): Directory where the images will be saved.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, predicted_image in enumerate(y_pred):
        file_name = f"prediction_{i}.png"
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, (predicted_image * 255).astype(np.uint8))

    print("Predicted images saved successfully to:", output_dir)

def fix_array_from_mhd_file(space_str ):
    # Remove the brackets and split the string by spaces
    space_str = space_str.strip('[]')
    space_values = space_str.split()

    # Convert the values to floating-point numbers
    space_values = [float(value) for value in space_values]

    return space_values

# Function to apply CLAHE to an image
def apply_clahe(image):
    # Define CLAHE parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Convert to grayscale if the image is in color
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the image is 8-bit
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Apply CLAHE directly to the single-channel (grayscale) image
    clahe_image = clahe.apply(image)

    return clahe_image
