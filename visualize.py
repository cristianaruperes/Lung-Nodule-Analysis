import numpy as np
import matplotlib.pyplot as plt

def display_image_comparison(img, gt_img, pred_img):

    # Calculate the absolute difference between the ground truth and predicted masks
    difference_map = np.abs(gt_img - pred_img)

    # Create a figure with four subplots for original image, ground truth, predicted mask, and difference map
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Plot the original image
    axes[0].imshow(np.squeeze(img), cmap='gray')
    axes[0].set_title('Original Image')

    # Plot the ground truth mask
    axes[1].imshow(np.squeeze(gt_img), cmap='gray')
    axes[1].set_title('Ground Truth Mask')

    # Plot the predicted mask
    axes[2].imshow(np.squeeze(pred_img), cmap='gray')
    axes[2].set_title('Predicted Mask')

    # Plot the difference map
    axes[3].imshow(np.squeeze(difference_map), cmap='viridis', vmax=1.0)
    axes[3].set_title('Difference Map')

    # Display the figure
    plt.tight_layout()
    plt.show()

def display_clahe (x_train, x_clahe, num_samples) :
    # Display a few examples before and after applying CLAHE

    for i in range(num_samples):
        # Original image
        original_image = x_train[i]

        # Apply CLAHE
        clahe_image = x_clahe[i]

        # Plot the images side by side
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(clahe_image, cmap='gray')
        plt.title('Image after CLAHE')

        plt.show()