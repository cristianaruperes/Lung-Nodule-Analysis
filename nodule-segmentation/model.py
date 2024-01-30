from tensorflow.keras.layers import *
from keras.models import *

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def unet(input_size=(64,64,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.25)(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(drop3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.25)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])


def segnet(input_shape=(64, 64, 1), num_classes=1):
    model = Sequential()

    # Encoder
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Decoder
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same'))

    return model

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    return x

def nested_unet(input_size=(64, 64, 1), num_filters=32):
    inputs = Input(input_size)
    
    # Contracting Path
    conv1_1 = conv_block(inputs, num_filters)
    pool1 = MaxPooling2D((2, 2))(conv1_1)

    conv2_1 = conv_block(pool1, num_filters * 2)
    pool2 = MaxPooling2D((2, 2))(conv2_1)

    conv3_1 = conv_block(pool2, num_filters * 4)
    pool3 = MaxPooling2D((2, 2))(conv3_1)

    conv4_1 = conv_block(pool3, num_filters * 8)
    pool4 = MaxPooling2D((2, 2))(conv4_1)

    # Bottleneck
    conv5_1 = conv_block(pool4, num_filters * 16)

    # Expanding Path
    up4_2 = UpSampling2D((2, 2))(conv5_1)
    conv4_2 = conv_block(Concatenate(axis=-1)([up4_2, conv4_1]), num_filters * 8)

    up3_3 = UpSampling2D((2, 2))(conv4_2)
    conv3_3 = conv_block(Concatenate(axis=-1)([up3_3, conv3_1]), num_filters * 4)

    up2_4 = UpSampling2D((2, 2))(conv3_3)
    conv2_4 = conv_block(Concatenate(axis=-1)([up2_4, conv2_1]), num_filters * 2)

    up1_5 = UpSampling2D((2, 2))(conv2_4)
    conv1_5 = conv_block(Concatenate(axis=-1)([up1_5, conv1_1]), num_filters)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv1_5)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# def attention_block(x, g, inter_channel):
#     theta_x = Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(x)
#     phi_g = Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(g)

#     theta_x_gap = GlobalAveragePooling2D()(theta_x)
#     phi_g_gap = GlobalAveragePooling2D()(phi_g)

#     theta_x_reshape = Reshape((1, 1, inter_channel))(theta_x_gap)
#     phi_g_reshape = Reshape((1, 1, inter_channel))(phi_g_gap)

#     concat = Concatenate(axis=-1)([theta_x_reshape, phi_g_reshape])

#     f = Dense(inter_channel, activation='relu')(concat)

#     psi_f = Dense(1, activation='sigmoid')(f)

#     rate = Reshape((1, 1, 1))(psi_f)
#     att_x = Multiply()([x, rate])

#     return att_x

# def attention_unet(input_size=(64, 64, 1), attention_inter_channel=8):
#     inputs = Input(input_size)

#     # Encoder
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
#     att1 = attention_block(conv1, conv1, attention_inter_channel)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(att1)

#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
#     att2 = attention_block(conv2, conv1, attention_inter_channel)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(att2)

#     conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
#     att3 = attention_block(conv3, conv2, attention_inter_channel)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(att3)

#     # Continue with additional encoder blocks...

#     # Decoder
#     up6 = UpSampling2D(size=(2, 2))(conv3)
#     att6 = attention_block(up6, conv2, attention_inter_channel)
#     merge6 = Concatenate(axis=3)([att6, conv2])
#     conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
#     conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

#     up7 = UpSampling2D(size=(2, 2))(conv6)
#     att7 = attention_block(up7, conv1, attention_inter_channel)
#     merge7 = Concatenate(axis=3)([att7, conv1])
#     conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
#     conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

#     outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

#     return Model(inputs=inputs, outputs=outputs)

def fcn_model(input_size=(64, 64, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)

    # Decoder
    up4 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    # Output layer
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=output)

    return model

# def deeplabv3_binary(input_shape):
#     # Load the MobileNetV2 model as the base
#     base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

#     # Use the output from the last convolutional layer
#     base_output = base_model.get_layer('out_relu').output

#     # Upsample the feature map
#     x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(base_output)
#     x = concatenate([x, base_model.get_layer('block_16_project_BN').output], axis=-1)

#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
#     x = concatenate([x, base_model.get_layer('block_13_project_BN').output], axis=-1)

#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
#     x = concatenate([x, base_model.get_layer('block_6_project_BN').output], axis=-1)

#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

#     # Final convolutional layer for binary segmentation with sigmoid activation
#     output = Conv2D(1, (1, 1), activation='sigmoid')(x)

#     # Create the model
#     model = Model(inputs=base_model.input, outputs=output)

#     return model


def swin_transformer_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    # Swin Transformer block
    conv1 = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(filters, kernel_size, strides=strides, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    return conv2

def channel_attention_module(x, ratio=8):
    # Channel attention module
    channels = int(x.shape[-1])
    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    concat = concatenate([avg_pool, max_pool], axis=-1)
    f = Conv2D(channels // ratio, 1, activation='relu', padding='same')(concat)
    f = Conv2D(channels, 1, activation='sigmoid', padding='same')(f)
    return f * x

def ca_unet(input_size=(64, 64, 1)):
    inputs = Input(input_size)

    conv1 = swin_transformer_block(inputs, 32)
    conv1 = swin_transformer_block(conv1, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = swin_transformer_block(pool1, 64)
    conv2 = swin_transformer_block(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = swin_transformer_block(pool2, 128)
    conv3 = swin_transformer_block(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.25)(pool3)

    conv4 = swin_transformer_block(drop3, 256)
    conv4 = swin_transformer_block(conv4, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = swin_transformer_block(pool4, 512)
    conv5 = swin_transformer_block(conv5, 512)
    drop5 = Dropout(0.25)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5), conv4], axis=3)
    conv6 = swin_transformer_block(up6, 256)
    conv6 = channel_attention_module(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = swin_transformer_block(up7, 128)
    conv7 = channel_attention_module(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = swin_transformer_block(up8, 64)
    conv8 = channel_attention_module(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = swin_transformer_block(up9, 32)
    conv9 = channel_attention_module(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])


# SAtUNET

def atrous_conv_block(x, filters, dilation_rate=(1, 1)):
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(x)
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same', dilation_rate=dilation_rate)(conv)
    return conv

def sa_unet(input_size=(64, 64, 1)):
    inputs = Input(input_size)
    
    conv1 = atrous_conv_block(inputs, 32)
    conv1 = atrous_conv_block(conv1, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = atrous_conv_block(pool1, 64)
    conv2 = atrous_conv_block(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = atrous_conv_block(pool2, 128)
    conv3 = atrous_conv_block(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.25)(pool3)

    conv4 = atrous_conv_block(drop3, 256)
    conv4 = atrous_conv_block(conv4, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = atrous_conv_block(pool4, 512)
    conv5 = atrous_conv_block(conv5, 512)
    drop5 = Dropout(0.25)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5), conv4], axis=3)
    conv6 = atrous_conv_block(up6, 256)
    conv6 = atrous_conv_block(conv6, 256)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = atrous_conv_block(up7, 128)
    conv7 = atrous_conv_block(conv7, 128)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = atrous_conv_block(up8, 64)
    conv8 = atrous_conv_block(conv8, 64)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = atrous_conv_block(up9, 32)
    conv9 = atrous_conv_block(conv9, 32)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
