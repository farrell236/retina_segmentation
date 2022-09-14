import os
os.environ['WANDB_CACHE_DIR'] = '/vol/medic01/users/bh1511/wandb'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import cv2
import math
import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from matplotlib import pyplot as plt

from models.DeeplabV3Plus import DeeplabV3Plus
from sklearn.model_selection import KFold

import wandb
from wandb.keras import WandbCallback



a=1


root_dir = '/vol/biomedic3/bh1511/retina/DRIVE/preprocessed'
train_df = pd.read_csv(os.path.join(root_dir, 'train_list.csv'))
train_df = root_dir + '/DRIVE_train/' + train_df[['image', 'label']]


a=1

images = []
labels = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
for idx, row in tqdm.tqdm(train_df.iterrows()):
    image = cv2.imread(row['image'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    images.append(image)
    labels.append(cv2.imread(row['label']))

images = np.stack(images, axis=0)
labels = np.stack(labels, axis=0)


def load(image, label):
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image)
    label = tf.io.read_file(label)
    label = tf.image.decode_png(label)
    return image, label


def random_rotate(image, label):
    degree = tf.random.normal([]) * 360
    image = tfa.image.rotate(image, degree * math.pi / 180., interpolation='nearest')
    label = tfa.image.rotate(label, degree * math.pi / 180., interpolation='nearest')
    return image, label


def colour_augmentation(image, label):
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    return image, label


def random_crop(image, label, height=512, width=512):
    stacked_image = tf.stack([image, label], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, height, width, 3])
    return cropped_image[0], cropped_image[1]


def random_flip(image, label):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    return image, label


def normalize(image, label):
    # normalizing the images to [-1, 1]
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = label[..., 0][..., None]
    return image, label


def load_image_train(image, label):
    # image, label = load(image, label)
    image, label = random_rotate(image, label)
    image, label = random_flip(image, label)
    image, label = colour_augmentation(image, label)
    # image, label = random_crop(image, label)
    image, label = normalize(image, label)
    return image, label


def load_image_test(image, label):
    # image, label = load(image, label)
    image, label = normalize(image, label)
    return image, label


def dice_coef(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for binary class labels
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, dtype=tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersect = tf.keras.backend.sum(y_true_f * y_pred_f)
    denom = tf.keras.backend.sum(y_true_f + y_pred_f)
    return tf.keras.backend.mean((2. * intersect / (denom + smooth)))


def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred, alpha=0.5):
    bce = tf.keras.losses.BinaryCrossentropy()
    return (1 - alpha) * bce(y_true, y_pred) + alpha * dice_coef_loss(y_true, y_pred)

a=1

kf = KFold()

for idx, (train_index, test_index) in enumerate(kf.split(images)):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = images[train_index], images[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    wandb.init(project='DR-Segmentation', entity="farrell236", group="x-val", name=f'fold_{idx}')

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(2)

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    valid_dataset = valid_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(2)


    model = DeeplabV3Plus((1024, 1024, 1), 1, activation='sigmoid')


    a=1

    # Fine tune with dice loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[dice_coef])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'CI/DeeplabV3Plus_DRIVE_{idx}.tf',
        monitor='val_dice_coef', mode='max', verbose=1, save_best_only=True)
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        steps_per_epoch=len(train_dataset),
        validation_steps=len(valid_dataset),
        epochs=2000,
        callbacks=[checkpoint, WandbCallback()]
    )


    a=1

    del model
    del train_dataset
    del valid_dataset

    wandb.join()
