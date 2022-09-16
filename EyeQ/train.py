import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa


data_root = '/vol/biomedic3/bh1511/retina/EyePACS'
train_df = pd.read_csv('Label_EyeQ_train.csv', index_col=[0])
train_df['image'] = data_root + '/train/' + train_df['image']
test_df = pd.read_csv('Label_EyeQ_test.csv', index_col=[0])
test_df['image'] = data_root + '/test/' + test_df['image']

a=1

##### Tensorflow Dataloader ############################################################################################


def parse_function(filename, label):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)

    # Resize image with padding to 512x512
    image = tf.image.resize(image, [512, 512])

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


def augmentation_fn(image, label):
    # Random left-right flip the image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random rotation
    # degree = tf.random.normal([]) * 360
    # image = tfa.image.rotate(image, degree * np.pi / 180., interpolation='nearest')

    # Random brightness, saturation and contrast shifting
    # image = tf.image.random_brightness(image, 0.2)
    # image = tf.image.random_hue(image, 0.08)
    # image = tf.image.random_saturation(image, 0.6, 1.6)
    # image = tf.image.random_contrast(image, 0.7, 1.3)

    # Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def load_image_train(image, label):
    image, label = parse_function(image, label)
    image, label = augmentation_fn(image, label)
    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_df['image'], train_df['quality']))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=8)
train_dataset = train_dataset.batch(16)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices((test_df['image'], test_df['quality']))
valid_dataset = valid_dataset.map(parse_function, num_parallel_calls=8)
valid_dataset = valid_dataset.batch(8)


##### Network Definition ###############################################################################################

model = tf.keras.Sequential([
    tf.keras.applications.ResNet50V2(include_top=False, weights=None, pooling='avg'),
    tf.keras.layers.Dense(1024, activation='swish'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='ResNetV2-EyeQ-QA.tf',
    monitor='val_accuracy', verbose=1, save_best_only=True)

# model.fit(train_dataset, epochs=20, callbacks=[checkpoint])
model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=100,
          callbacks=[checkpoint])


########################################################################################################################
