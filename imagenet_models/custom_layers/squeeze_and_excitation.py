"""Squeeze and excitation layer for Keras.
# Reference:
- [Squeeze-and-Excitation Networks](
    https://arxiv.org/abs/1709.01507) (CVPR 2018)
"""

import tensorflow.keras.layers as layers

def se_block(input_tensor, filters, conv_name_base, channel_axis, r=16):
    if channel_axis == 1:
        reshape_target = (filters, 1, 1)
    elif channel_axis == 3:
        reshape_target = (1, 1, filters)
    else:
        raise ValueError("channel_axis should be 1 or 3.")

    x = layers.GlobalAveragePooling2D(name=conv_name_base+"_se_pool")(input_tensor)
    x = layers.Dense(filters//r, activation="relu", 
                     kernel_initializer='he_normal',
                     name=conv_name_base+"_se_reduction")(x)
    x = layers.Dense(filters, activation="sigmoid", 
                     kernel_initializer='he_normal',
                     name=conv_name_base+"_se_sigmoid")(x)
    x = layers.Reshape(reshape_target, name=conv_name_base+"_se_reshape")(x)
    x = layers.Multiply(name=conv_name_base+"_se_scale")([input_tensor, x])
    return x