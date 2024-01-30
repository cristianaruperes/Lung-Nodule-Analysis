from model import *
from evaluate import *
from utils import *
from dataset import *
from visualize import *
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

tf.config.list_physical_devices('GPU')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# dataset FEMH numpy
# x_train = np.load('data/split/femh/data474/original/train_data/x_train.npy')
# y_train = np.load('data/split/femh/data474/original/train_data/y_train.npy')
# x_val = np.load('data/split/femh/data474/original/val_data/x_val.npy')
# y_val = np.load('data/split/femh/data474/original/val_data/y_val.npy')

# FEMH raw
# image_path_train_femh = "../data-thesis/split/femh/data474/image-format/train_data/images/"
# mask_path_train_femh = "../data-thesis/split/femh/data474/image-format/train_data/masks/"
# image_path_val_femh = "../data-thesis/split/femh/data474/image-format/val_data/images/"
# mask_path_val_femh = "../data-thesis/split/femh/data474/image-format/val_data/masks/"

# LUNA16 raw
image_path_train_luna = "../data-thesis/split/luna16/data934/image-format/train_data/images/"
mask_path_train_luna = "../data-thesis/split/luna16/data934/image-format/train_data/masks/"
image_path_val_luna = "../data-thesis/split/luna16/data934/image-format/val_data/images/"
mask_path_val_luna = "../data-thesis/split/luna16/data934/image-format/val_data/masks/"

# convert image to numpy array
images_set_train_luna, masks_set_train_luna= read_dataset(image_path_train_luna, mask_path_train_luna)

# images_set_train_femh, masks_set_train_femh= read_dataset(image_path_train_femh, mask_path_train_femh)

images_set_val_luna, masks_set_val_luna= read_dataset(image_path_val_luna, mask_path_val_luna)

# images_set_val_femh, masks_set_val_femh= read_dataset(image_path_val_femh, mask_path_val_femh)

x_train_luna = np.array(images_set_train_luna)
y_train_luna = np.array(masks_set_train_luna)
x_val_luna = np.array(images_set_val_luna)
y_val_luna = np.array(masks_set_val_luna)

# x_train_femh = np.array(images_set_train_femh)
# y_train_femh = np.array(masks_set_train_femh)
# x_val_femh = np.array(images_set_val_femh)
# y_val_femh = np.array(masks_set_val_femh)

############ MIX ##########################
# Concatenate the two datasets along the first axis (axis=0)
# concatenated_data = np.concatenate([x_train_luna, x_train_femh], axis=0)

# x_train = np.concatenate([x_train_luna, x_train_femh], axis=0)
# y_train = np.concatenate([y_train_luna, y_train_femh], axis=0)
# x_val = np.concatenate([x_val_luna, x_val_femh], axis=0)
# y_val = np.concatenate([y_val_luna, y_val_femh], axis=0)

# FEMH
# x_train = x_train_femh
# y_train = y_train_femh
# x_val = x_val_femh
# y_val = y_val_femh

# LUNA
x_train = x_train_luna
y_train = y_train_luna
x_val = x_val_luna
y_val = y_val_luna

# Print or do further operations with the concatenated data
print(len(x_train))
print(len(y_train))
print(len(x_val))
print(len(y_val))

print(x_train.shape)
############ MIX ##########################

# display samples of image and mask in gray scale
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(x_train[1].squeeze(), cmap='gray') 
# ax[1].imshow(y_train[1].squeeze(), cmap='gray')

# Combine your training and validation sets for k-fold cross-validation
X_train_val = np.concatenate((x_train, x_val), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)

print(len(X_train_val))
print(len(y_train_val))

def apply_test_augmentation(x_train, y_train, x_val, y_val):

    ################### Data augmentation
    # Rotate
    rotated_images, rotated_masks = rotate_image_and_mask(x_train, y_train)
    # Flip image
    Flipped_images, Flipped_masks = flip_image(x_train, y_train)
    # Combine
    images_augmented, masks_augmented = augmentation_data(x_train, y_train, rotated_images, rotated_masks, Flipped_images, Flipped_masks)

    #flip only
    # images_augmented, masks_augmented = augmentation_data_only_flip(x_train, y_train, Flipped_images, Flipped_masks)

    ###################

    print("Image augmented shape:", images_augmented.shape)
    print("Mask augmentted shape:", masks_augmented.shape)

    # Display image augmentation
    # display_images_augmented(images_augmented, masks_augmented)

    # Apply CLAHE to x_train
    x_train_clahe = np.array([apply_clahe(image)[:, :, np.newaxis] for image in tqdm(images_augmented, desc="Applying CLAHE to x_train")])
    # Apply CLAHE to x_val
    x_val_clahe = np.array([apply_clahe(image)[:, :, np.newaxis] for image in tqdm(x_val, desc="Applying CLAHE to x_val")])

    # display_clahe(x_train, x_train_clahe, 3)

    #inputs to the model
    x_train_fix = x_train_clahe
    y_train_fix = masks_augmented
    x_val_fix = x_val_clahe
    y_val_fix = y_val

    # x_train = np.repeat(x_train, 3, axis=-1)
    # y_train = np.repeat(y_train, 3, axis=-1)
    # x_val = np.repeat(x_val, 3, axis=-1)
    # y_val = np.repeat(y_val, 3, axis=-1)

    return x_train_fix, y_train_fix, x_val_fix, y_val_fix

# Define the number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
dsc_scores = []
iou_scores = []

# Iterate through folds
for fold, (train_index, val_index) in enumerate(kf.split(X_train_val)):
    
    print(f"\n Training for LUNA SA-unet Fold {fold + 1} START.\n")
    
    X_fold_train, X_fold_val = X_train_val[train_index], X_train_val[val_index]
    y_fold_train, y_fold_val = y_train_val[train_index], y_train_val[val_index]


    # Apply data augmentation to the training set only
    X_fold_train_aug, y_fold_train_aug, X_fold_val_aug, y_fold_val_aug = apply_test_augmentation(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    # Train your model on X_fold_train, y_fold_train
    ck_path = f"models/luna/validation/sa-unet/sa-unet-fix-aug_fold{fold + 1}.hdf5"
    reduces = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, mode='auto', verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(ck_path, 
                                    monitor='val_loss',
                                    mode = "min",
                                    verbose=1, 
                                    save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=100, verbose=1)
    callback_list = [model_checkpoint]

    # Define Model
    IMG_HEIGHT = X_fold_train_aug.shape[1]
    IMG_WIDTH  = X_fold_train_aug.shape[2]
    IMG_CHANNELS = X_fold_train_aug.shape[3]

    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # UNET
    # model = unet(input_shape)

    #segnet
    # model = segnet(input_shape, num_classes=1)

    #Unet++
    # model = nested_unet(input_size=(64, 64, 1), num_filters=32)

    #FCN
    # model = fcn_model(input_size=(64, 64, 1))
    
    # ca-unet
    # model = ca_unet(input_shape)

    # sa-unet
    model = sa_unet(input_shape)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss = dice_coef_loss,
              metrics = ['accuracy', dice_coef, iou_score])
    
    batch_size = 16
    # Training 
    history = model.fit(X_fold_train_aug, y_fold_train_aug,
        epochs=50,
        batch_size = batch_size,
        steps_per_epoch = len(X_fold_train_aug)/batch_size,
        validation_data = (X_fold_val_aug, y_fold_val_aug),
        validation_steps = len(X_fold_val_aug)/batch_size,
        callbacks=callback_list)

    # Validate your model on X_fold_val, y_fold_val
    # Visualize Metrics
    # Evaluate the model on the validation data
    loss_and_metrics = model.evaluate(X_fold_val_aug, y_fold_val_aug)
    
    print(f"Mean IoU for Fold {fold + 1}: {loss_and_metrics[3]}")
    print(f"Dice Coefficient for Fold {fold + 1}: {loss_and_metrics[2]}")

    # predict segmentation
    preds = model.predict(X_fold_val_aug)

    # show results for the first 20 rows
    num_rows_to_display = 5
    fig, ax = plt.subplots(num_rows_to_display, 3, figsize=(10, 20))
    for i in range(num_rows_to_display):
        ax[i, 0].imshow(X_fold_val_aug[i].squeeze(), cmap='gray')
        ax[i, 1].imshow(y_fold_val_aug[i].squeeze(), cmap='gray')
        ax[i, 2].imshow(preds[i].squeeze(), cmap='gray')
    plt.show()

    visualize_all(history)
    
    dsc_scores.append(loss_and_metrics[2])
    iou_scores.append(loss_and_metrics[3])
    
    print(f"Training for Fold {fold + 1} completed.\n")


# Iterate through the list and print the scores for each fold
for fold, dsc_score in enumerate(dsc_scores, start=1):
    print(f"Fold {fold}: Dice Coefficient = {dsc_score}")

# Iterate through the list and print the scores for each fold
for fold, iou_score in enumerate(iou_scores, start=1):
    print(f"Fold {fold}: IoU Scores = {iou_score}")
     
# Calculate mean and standard deviation of accuracy scores
dsc_mean_accuracy = np.mean(dsc_scores)
dsc_std_accuracy = np.std(dsc_scores)

iou_mean_accuracy = np.mean(iou_scores)
iou_std_accuracy = np.std(iou_scores)

# Print the results
print(f"DSC Mean Accuracy: {dsc_mean_accuracy}")
print(f"DSC Standard Deviation of Accuracy: {dsc_std_accuracy}")

print(f"IoU Mean Accuracy: {iou_mean_accuracy}")
print(f"IoU Standard Deviation of Accuracy: {iou_std_accuracy}")
