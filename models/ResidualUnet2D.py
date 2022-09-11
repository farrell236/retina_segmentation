# Keras 2D Residual Unet Model
# https://keras.io/examples/generative/ddim/#network-architecture

import tensorflow as tf


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[-1]
        if input_width == width:
            residual = x
        else:
            residual = tf.keras.layers.Conv2D(width, kernel_size=1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            width, kernel_size=3, padding="same", activation=tf.keras.activations.swish
        )(x)
        x = tf.keras.layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = tf.keras.layers.UpSampling2D(size=2)(x)
        for _ in range(block_depth):
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(input_shape, widths, block_depth, n_classes=10, activation='softmax'):

    image_input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(widths[0], kernel_size=1)(image_input)

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = tf.keras.layers.Conv2D(n_classes, 1, activation=activation)(x)

    return tf.keras.Model([image_input], x, name="residual_unet")


if __name__ == '__main__':

    a=1

    widths = [32, 64, 96, 128]
    block_depth = 4

    a=1

    unet = get_network((256, 256, 1), widths, block_depth, n_classes=10)

    a=1

    import numpy as np
    x = np.random.rand(2, 256, 256, 1).astype('float32')
    out = unet(x)



