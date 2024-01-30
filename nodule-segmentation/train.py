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

tf.config.list_physical_devices('GPU')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# dataset FEMH numpy
# x_train = np.load('data/split/femh/data474/original/train_data/x_train.npy')
# y_train = np.load('data/split/femh/data474/original/train_data/y_train.npy')
# x_val = np.load('data/split/femh/data474/original/val_data/x_val.npy')
# y_val = np.load('data/split/femh/data474/original/val_data/y_val.npy')

# FEMH raw
image_path_train_femh = "../data-thesis/split/femh/data474/image-format/train_data/images/"
mask_path_train_femh = "../data-thesis/split/femh/data474/image-format/train_data/masks/"
image_path_val_femh = "../data-thesis/split/femh/data474/image-format/val_data/images/"
mask_path_val_femh = "../data-thesis/split/femh/data474/image-format/val_data/masks/"

# LUNA16 raw
image_path_train_luna = "../data-thesis/split/luna16/data934/image-format/train_data/images/"
mask_path_train_luna = "../data-thesis/split/luna16/data934/image-format/train_data/masks/"
image_path_val_luna = "../data-thesis/split/luna16/data934/image-format/val_data/images/"
mask_path_val_luna = "../data-thesis/split/luna16/data934/image-format/val_data/masks/"

# convert image to numpy array
images_set_train_luna, masks_set_train_luna= read_dataset(image_path_train_luna, mask_path_train_luna)

images_set_train_femh, masks_set_train_femh= read_dataset(image_path_train_femh, mask_path_train_femh)

images_set_val_luna, masks_set_val_luna= read_dataset(image_path_val_luna, mask_path_val_luna)

images_set_val_femh, masks_set_val_femh= read_dataset(image_path_val_femh, mask_path_val_femh)

x_train_luna = np.array(images_set_train_luna)
y_train_luna = np.array(masks_set_train_luna)
x_val_luna = np.array(images_set_val_luna)
y_val_luna = np.array(masks_set_val_luna)

x_train_femh = np.array(images_set_train_femh)
y_train_femh = np.array(masks_set_train_femh)
x_val_femh = np.array(images_set_val_femh)
y_val_femh = np.array(masks_set_val_femh)

############ MIX ##########################
# Concatenate the two datasets along the first axis (axis=0)
# concatenated_data = np.concatenate([x_train_luna, x_train_femh], axis=0)

x_train = np.concatenate([x_train_luna, x_train_femh], axis=0)
y_train = np.concatenate([y_train_luna, y_train_femh], axis=0)
x_val = np.concatenate([x_val_luna, x_val_femh], axis=0)
y_val = np.concatenate([y_val_luna, y_val_femh], axis=0)

# x_train = x_train_femh
# y_train = y_train_femh
# x_val = x_val_femh
# y_val = y_val_femh

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

# Run Model

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
x_train = x_train_clahe
y_train = masks_augmented
x_val = x_val_clahe
y_val = y_val

# x_train = np.repeat(x_train, 3, axis=-1)
# y_train = np.repeat(y_train, 3, axis=-1)
# x_val = np.repeat(x_val, 3, axis=-1)
# y_val = np.repeat(y_val, 3, axis=-1)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)


# change in every train
model_name = "models/mix/sa-unet/sa-unet-full-50-aug.hdf5"
model_plot = "figures/mix/sa-unet/sa-unet-full-50-aug.png"
print("========= test sa-unet 50 full augmentation ===========")

# Define callbacks lists
ck_path = model_name
reduces = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, mode='auto', verbose=1)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(ck_path, 
                                   monitor='val_loss',
                                   mode = "min",
                                   verbose=1, 
                                   save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(patience=100, verbose=1)
callback_list = [model_checkpoint]

# Define Model
IMG_HEIGHT = x_train.shape[1]
IMG_WIDTH  = x_train.shape[2]
IMG_CHANNELS = x_train.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# UNET
# model = unet(input_shape)

# ca-unet
# model = ca_unet(input_shape)

# sa-unet
model = sa_unet(input_shape)

#segnet
# model = segnet(input_shape, num_classes=1)

#Unet++
# model = nested_unet(input_size=(64, 64, 1), num_filters=32)

#FCN
# model = fcn_model(input_size=(64, 64, 1))

#DeepLab v3
# model = deeplabv3_binary(input_shape)

#Attention unet
# model = attention_unet(input_size=(64, 64, 1))

model.summary()

# # Save the summary to a file
# with open('model_summary.txt', 'w') as f:
#     model.summary(print_fn=lambda x: f.write(x + '\n'))

# # Open the file to view the complete details
# with open('model_summary.txt', 'r') as f:
#     print(f.read())


# # Count total number of layers
# total_layers = len(model.layers)
# print(f'Total number of layers: {total_layers}')

# tf.keras.utils.plot_model(model, show_dtype=True, show_layer_names=True, show_shapes=True, to_file=model_plot)

# # Specify the file path for saving the model architecture image
# model_plot = 'unet_architecture.png'

# # Plot the model architecture
# plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True)

# # Save the figure
# plt.savefig(model_plot, bbox_inches='tight', format='png')

# print(f"Model architecture image saved at {model_plot}")


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss = dice_coef_loss,
              metrics = ['accuracy', dice_coef, iou_score])

batch_size = 16
# Training 
history = model.fit(x_train, y_train,
        epochs=50,
        batch_size = batch_size,
        steps_per_epoch = len(x_train)/batch_size,
        validation_data = (x_val, y_val),
        validation_steps = len(x_val)/batch_size,
        callbacks=callback_list)

# Visualize Metrics
# Evaluate the model on the validation data
loss_and_metrics = model.evaluate(x_val, y_val)
print("Mean IoU:", loss_and_metrics[3])  # Or appropriate index for accuracy
print("Dice_coefficient:", loss_and_metrics[2])  # Or appropriate index for accuracy

# predict segmentation
preds = model.predict(x_val)

# show results for the first 20 rows
num_rows_to_display = 5
fig, ax = plt.subplots(num_rows_to_display, 3, figsize=(10, 20))
for i in range(num_rows_to_display):
    ax[i, 0].imshow(x_val[i].squeeze(), cmap='gray')
    ax[i, 1].imshow(y_val[i].squeeze(), cmap='gray')
    ax[i, 2].imshow(preds[i].squeeze(), cmap='gray')
plt.show()

visualize_all(history)