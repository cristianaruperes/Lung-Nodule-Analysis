from model import *
from evaluate import *
from utils import *
from dataset import *
from visualize import *
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# # FEMH numpy
# x_test = np.load('data/split/femh/data474/original/test_data/x_test.npy')
# y_test = np.load('data/split/femh/data474/original/test_data/y_test.npy')

# FEMH raw
<<<<<<< Updated upstream
x_test_femh = 'data/split/femh/data474/image-format/test_data/images/'
y_test_femh = 'data/split/femh/data474/image-format/test_data/masks/'

# LUNA16 raw
x_test_luna = 'data/split/luna16/data934/image-format/test_data/images/'
y_test_luna = 'data/split/luna16/data934/image-format/test_data/masks/'

x_test_femh, y_test_femh = read_dataset(x_test_femh, y_test_femh)
x_test_luna, y_test_luna = read_dataset(x_test_luna, y_test_luna)

x_test_femh = np.array(x_test_femh)
y_test_femh = np.array(y_test_femh)
=======
# x_test_femh = '../data-thesis/split/femh/data474/image-format/test_data/images/'
# y_test_femh = '../data-thesis/split/femh/data474/image-format/test_data/masks/'

# LUNA16 raw
x_test_luna = '../data-thesis/split/luna16/data934/image-format/test_data/images/'
y_test_luna = '../data-thesis/split/luna16/data934/image-format/test_data/masks/'

# x_test_femh, y_test_femh = read_dataset(x_test_femh, y_test_femh)
x_test_luna, y_test_luna = read_dataset(x_test_luna, y_test_luna)

# x_test_femh = np.array(x_test_femh)
# y_test_femh = np.array(y_test_femh)
>>>>>>> Stashed changes
x_test_luna = np.array(x_test_luna)
y_test_luna = np.array(y_test_luna)

############ MIX ##########################
# Concatenate the two datasets along the first axis (axis=0)
# concatenated_data = np.concatenate([x_train_luna, x_train_femh], axis=0)

<<<<<<< Updated upstream
x_test = np.concatenate([x_test_femh, x_test_luna], axis=0)
y_test = np.concatenate([y_test_femh, y_test_luna], axis=0)
=======
# x_test = np.concatenate([x_test_femh, x_test_luna], axis=0)
# y_test = np.concatenate([y_test_femh, y_test_luna], axis=0)

#FEMH
# x_test = x_test_femh
# y_test = y_test_femh

#LUNA
x_test = x_test_luna
y_test = y_test_luna
>>>>>>> Stashed changes

# Print or do further operations with the concatenated data
print(len(x_test))
print(len(y_test))
############ MIX ##########################

# print(len(x_test))

# display samples of image and mask in gray scale
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(x_test[6].squeeze(), cmap='gray') 
# ax[1].imshow(y_test[6].squeeze(), cmap='gray')

# Apply CLAHE to x_train
x_test_clahe = np.array([apply_clahe(image)[:, :, np.newaxis] for image in tqdm(x_test, desc="Applying CLAHE to x_train")])

# display_clahe(x_test, x_test_clahe, 3)

x_test = x_test_clahe

# Clear the previous TensorFlow session and model
tf.keras.backend.clear_session()

# Load model
<<<<<<< Updated upstream
ck_path = "models/mix/unet-aug.hdf5"
ck_path1 = "models/mix/segnet-aug.hdf5"
ck_path2 = "models/mix/nested-aug.hdf5"
ck_path3 = "models/mix/fcn-aug.hdf5"
=======
ck_path = "models/luna/validation/sa-unet/sa-unet-fix-aug_fold1.hdf5"
ck_path1 = "models/luna/validation/sa-unet/sa-unet-fix-aug_fold2.hdf5"
ck_path2 = "models/luna/validation/sa-unet/sa-unet-fix-aug_fold3.hdf5"
ck_path3 = "models/luna/validation/sa-unet/sa-unet-fix-aug_fold4.hdf5"
ck_path4 = "models/luna/validation/sa-unet/sa-unet-fix-aug_fold5.hdf5"
>>>>>>> Stashed changes

print("===========test all aug============")
input_shape = (64, 64, 1)

# UNET
<<<<<<< Updated upstream
model = unet(input_shape)

#segnet
model1 = segnet(input_shape, num_classes=1)

#Unet++
model2 = nested_unet(input_size=(64, 64, 1), num_filters=32)
=======
# model = unet(input_shape)

# ca unet
# model = ca_unet(input_shape)

# sa-unet
# model = sa_unet(input_shape)

# #segnet
# model1 = segnet(input_shape, num_classes=1)

# #Unet++
# model2 = nested_unet(input_size=(64, 64, 1), num_filters=32)
>>>>>>> Stashed changes

#Attention unet
# model = attention_unet(input_size=(64, 64, 1))

#FCN
<<<<<<< Updated upstream
model3 = fcn_model(input_size=(64, 64, 1))
=======
# model3 = fcn_model(input_size=(64, 64, 1))

#MIX
model = sa_unet(input_shape)
model1 = sa_unet(input_shape)
model2 = sa_unet(input_shape)
model3 = sa_unet(input_shape)
model4 = sa_unet(input_shape)

>>>>>>> Stashed changes

model.load_weights(ck_path)
model1.load_weights(ck_path1)
model2.load_weights(ck_path2)
model3.load_weights(ck_path3)
<<<<<<< Updated upstream
=======
model4.load_weights(ck_path4)

>>>>>>> Stashed changes

# #Prediction
y_pred = model.predict(x_test)
y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)
<<<<<<< Updated upstream
=======
y_pred4 = model4.predict(x_test)
>>>>>>> Stashed changes


# Assuming y_test and y_pred are your ground truth and predicted masks
# Threshold the predicted masks
# y_pred_thresholded = (y_pred > 0.5).astype(np.uint8)
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
y_pred_thresholded = (y_pred > 0.5).astype(np.float32)  # Cast to float32
y_pred_thresholded1 = (y_pred1 > 0.5).astype(np.float32)  # Cast to float32
y_pred_thresholded2 = (y_pred2 > 0.5).astype(np.float32)  # Cast to float32
y_pred_thresholded3 = (y_pred3 > 0.5).astype(np.float32)  # Cast to float32
<<<<<<< Updated upstream
=======
y_pred_thresholded4 = (y_pred4 > 0.5).astype(np.float32)  # Cast to float32
>>>>>>> Stashed changes

# Cast y_test to float32 if it's not already
y_test = y_test.astype(np.float32)

# Calculate Dice coefficient and IoU
dice = dice_coef(y_test, y_pred_thresholded)
iou = iou_score(y_test, y_pred_thresholded)

dice1 = dice_coef(y_test, y_pred_thresholded1)
iou1 = iou_score(y_test, y_pred_thresholded1)

dice2 = dice_coef(y_test, y_pred_thresholded2)
iou2 = iou_score(y_test, y_pred_thresholded2)

dice3 = dice_coef(y_test, y_pred_thresholded3)
iou3 = iou_score(y_test, y_pred_thresholded3)

<<<<<<< Updated upstream
print("===========test all aug============")
print(f'Dice Coefficient (Unet): {dice:.4f}')
print(f'IoU (Unet): {iou:.4f}')

print(f'Dice Coefficient (Segnet): {dice1:.4f}')
print(f'IoU (segnet): {iou1:.4f}')

print(f'Dice Coefficient (Nested): {dice2:.4f}')
print(f'IoU (Nested): {iou2:.4f}')

print(f'Dice Coefficient (FCN): {dice3:.4f}')
print(f'IoU (FCN): {iou3:.4f}')
=======
dice4 = dice_coef(y_test, y_pred_thresholded4)
iou4 = iou_score(y_test, y_pred_thresholded4)

print("===========test all aug============")
print(f'Dice Coefficient (fold 1): {dice:.4f}')
print(f'IoU (fold 1): {iou:.4f}')

print(f'Dice Coefficient (fold 2): {dice1:.4f}')
print(f'IoU (fold 2): {iou1:.4f}')

print(f'Dice Coefficient (fold 3): {dice2:.4f}')
print(f'IoU (fold 3): {iou2:.4f}')

print(f'Dice Coefficient (fold 4): {dice3:.4f}')
print(f'IoU (fold 4): {iou3:.4f}')

print(f'Dice Coefficient (fold 5): {dice4:.4f}')
print(f'IoU (fold 5): {iou4:.4f}')
>>>>>>> Stashed changes

# show results for the first 20 rows
# num_rows_to_display = 5
# fig, ax = plt.subplots(num_rows_to_display, 3, figsize=(10, 20))
# for i in range(num_rows_to_display):
#     ax[i, 0].imshow(x_test[i].squeeze(), cmap='gray')
#     ax[i, 1].imshow(y_test[i].squeeze(), cmap='gray')
#     ax[i, 2].imshow(y_pred[i].squeeze(), cmap='gray')
# plt.show()

# Assuming you have y_pred1, y_pred2, y_pred3, y_pred4

<<<<<<< Updated upstream
# Show results for the first 5 rows without padding
num_rows_to_display = 5

for i in range(num_rows_to_display):
    # Ensure the shapes of the images are compatible for concatenation
    x_image = x_test[i].squeeze()
    y_true_image = y_test[i].squeeze()
    y_pred_image = y_pred[i].squeeze()
    y_pred1_image = y_pred1[i].squeeze()
    y_pred2_image = y_pred2[i].squeeze()
    y_pred3_image = y_pred3[i].squeeze()

    # Display the individual images without padding
    fig, ax = plt.subplots(1, 6, figsize=(25, 5), tight_layout=True)
    ax[0].imshow(x_image, cmap='gray')
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(y_true_image, cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')

    ax[2].imshow(y_pred_image, cmap='gray')
    ax[2].set_title('Unet')
    ax[2].axis('off')

    ax[3].imshow(y_pred1_image, cmap='gray')
    ax[3].set_title('Segnet')
    ax[3].axis('off')

    ax[4].imshow(y_pred2_image, cmap='gray')
    ax[4].set_title('Nested')
    ax[4].axis('off')

    ax[5].imshow(y_pred3_image, cmap='gray')
    ax[5].set_title('FCN')
    ax[5].axis('off')

    plt.show()
=======
# # Show results for the first 5 rows without padding
# num_rows_to_display = 5

# for i in range(num_rows_to_display):
#     # Ensure the shapes of the images are compatible for concatenation
#     x_image = x_test[i].squeeze()
#     y_true_image = y_test[i].squeeze()
#     y_pred_image = y_pred[i].squeeze()
#     y_pred1_image = y_pred1[i].squeeze()
#     y_pred2_image = y_pred2[i].squeeze()
#     y_pred3_image = y_pred3[i].squeeze()

#     # Display the individual images without padding
#     fig, ax = plt.subplots(1, 6, figsize=(25, 5), tight_layout=True)
#     ax[0].imshow(x_image, cmap='gray')
#     ax[0].set_title('Input Image')
#     ax[0].axis('off')

#     ax[1].imshow(y_true_image, cmap='gray')
#     ax[1].set_title('Ground Truth')
#     ax[1].axis('off')

#     ax[2].imshow(y_pred_image, cmap='gray')
#     ax[2].set_title('Unet')
#     ax[2].axis('off')

#     ax[3].imshow(y_pred1_image, cmap='gray')
#     ax[3].set_title('Segnet')
#     ax[3].axis('off')

#     ax[4].imshow(y_pred2_image, cmap='gray')
#     ax[4].set_title('Nested')
#     ax[4].axis('off')

#     ax[5].imshow(y_pred3_image, cmap='gray')
#     ax[5].set_title('FCN')
#     ax[5].axis('off')

#     plt.show()
>>>>>>> Stashed changes
