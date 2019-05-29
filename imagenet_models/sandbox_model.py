import tensorflow.keras as keras
from tensorflow.keras import layers

def conv_bn_relu(inputs, ch, kernel=3, strides=1):
    x = layers.Conv2D(ch, kernel, strides=strides, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def SandboxModelA():
    input = layers.Input((224, 224, 3))
    x = conv_bn_relu(input, 64, 13, 7)
    for i in range(3):
        x = conv_bn_relu(x, 128)
    x = layers.AveragePooling2D(2)(x)
    for i in range(3):
        x = conv_bn_relu(x, 256)
    x = layers.AveragePooling2D(2)(x)
    for i in range(3):
        x = conv_bn_relu(x, 512)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation="softmax")(x)
    return keras.models.Model(input, x)
