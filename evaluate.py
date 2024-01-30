import keras
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def dice_coef(y_true, y_pred):
    y_true_f = keras.layers.Flatten()(y_true)
    y_pred_f = keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-15) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-15)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# def dice_coef(y_true, y_pred, smooth=1):
#     intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
#     union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
#     dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
#     return dice

# def dice_coef_loss(y_true, y_pred):
#     return 1 - dice_coef(y_true, y_pred)

smooth = 1e-15
def iou_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def iou_loss(y_true, y_pred):
    return -iou_score(y_true, y_pred)

# def calculate_pcc(y_true, y_pred):
#     # Calculate Pearson Correlation Coefficient (PCC)
#     pcc, _ = pearsonr(y_true, y_pred)
#     return pcc

# def dice_coef(y_true, y_pred, smooth=1e-5):
#     intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
#     union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
#     dice = (2.0 * intersection + smooth) / (union + smooth)
#     return tf.reduce_mean(dice)

# def dice_coef_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# def iou_score(y_true, y_pred, smooth=1e-5):
#     intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
#     union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
#     iou = (intersection + smooth) / (union + smooth)
#     return tf.reduce_mean(iou)

def calculate_pcc(y_true, y_pred):
    # Flatten y_true and y_pred to 1D arrays using tf.reshape
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    # Calculate Pearson Correlation Coefficient (PCC)
    pcc, _ = pearsonr(y_true_flat.numpy(), y_pred_flat.numpy())
    
    return pcc

def calculate_rmse(y_true, y_pred):
    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

# from tensorflow.keras import backend as K
# def class_tversky(y_true, y_pred):
#     smooth = 1

#     #y_true = K.permute_dimensions(y_true, (3,1,2,0))
#     #y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

#     y_true_pos = K.batch_flatten(y_true)
#     y_pred_pos = K.batch_flatten(y_pred)
#     true_pos = K.sum(y_true_pos * y_pred_pos, 1)
#     false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
#     false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
#     alpha = 0.7
#     return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

# def focal_tversky_loss(y_true,y_pred):
#     pt_1 = class_tversky(y_true, y_pred)
#     gamma = 0.75
#     return K.sum(K.pow((1-pt_1), gamma))

def visualize_all(history):
    # Create a figure with four subplots
    plt.figure(figsize=(16, 4))

    # Plot training & validation loss values
    plt.subplot(141)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot training & validation accuracy values
    plt.subplot(142)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation IoU values
    plt.subplot(143)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation Dice coefficient values
    plt.subplot(144)
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Display the plots
    plt.tight_layout()
    plt.show()