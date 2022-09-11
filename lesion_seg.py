import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import tqdm
import math

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from models.DeeplabV3Plus import DeeplabV3Plus


import wandb
from wandb.keras import WandbCallback

wandb.init(project='DR-Segmentation', entity="farrell236")


a=1

# IDRiD Dataset
# root_dir = '/vol/biomedic3/bh1511/retina/IDRID/segmentation/preprocessed'
# images = [f'{root_dir}/0_Original_Images/IDRiD_{i:02d}.jpg' for i in range(1, 82)]
# labels = [f'{root_dir}/onehot_masks/IDRiD_{i:02d}_mask.png' for i in range(1, 82)]

# FGADR Dataset
from patient_ids import patient_ids
root_dir = '/vol/biomedic3/bh1511/retina/FGADR/Seg-set'
images = [f'{root_dir}/Original_Images/{i}.png' for i in patient_ids]
labels = [f'{root_dir}/idrid_labels/{i}.png' for i in patient_ids]


train_images = []
train_labels = []
test_images = []
test_labels = []


a=1

res = 1024
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

for idx in tqdm.tqdm(range(0, 1400)):
    image = cv2.imread(images[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = cv2.resize(image, (res, res))
    train_images.append(image)
    label = cv2.resize(cv2.imread(labels[idx]), (res, res), interpolation=cv2.INTER_NEAREST)
    train_labels.append(label)

for idx in tqdm.tqdm(range(1400, 1494)):
    image = cv2.imread(images[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = cv2.resize(image, (res, res))
    test_images.append(image)
    label = cv2.resize(cv2.imread(labels[idx]), (res, res), interpolation=cv2.INTER_NEAREST)
    test_labels.append(label)

train_images = np.stack(train_images, axis=0)
train_labels = np.stack(train_labels, axis=0)
test_images = np.stack(test_images, axis=0)
test_labels = np.stack(test_labels, axis=0)



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


def normalize(image, label):
    # normalizing the images to [-1, 1]
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = label[..., 0][..., None]
    return image, label


def load_image_train(image, label):
    # image, label = load(image, label)
    image, label = random_rotate(image, label)
    image, label = colour_augmentation(image, label)
    # image, label = random_crop(image, label)
    image, label = normalize(image, label)
    return image, label


def load_image_test(image, label):
    # image, label = load(image, label)
    image, label = normalize(image, label)
    return image, label


# train_dataset = tf.data.Dataset.from_tensor_slices((train_df['image'], train_df['label']))
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(2)

# valid_dataset = tf.data.Dataset.from_tensor_slices((test_df['image'], test_df['label']))
valid_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
valid_dataset = valid_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(2)


def dice_coef(y_true, y_pred, num_classes=6, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.keras.backend.flatten(tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)[..., 1:])
    y_pred_f = tf.keras.backend.flatten(y_pred[..., 1:])
    intersect = tf.keras.backend.sum(y_true_f * y_pred_f)
    denom = tf.keras.backend.sum(y_true_f + y_pred_f)
    return tf.keras.backend.mean((2. * intersect / (denom + smooth)))


def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred, alpha=0.5):
    sBCE = tf.keras.losses.SparseCategoricalCrossentropy()
    return (1 - alpha) * sBCE(y_true, y_pred) + alpha * dice_coef_loss(y_true, y_pred)


model = DeeplabV3Plus((1024, 1024, 3), 6, activation='softmax')  # IDRiD
# model = DeeplabV3Plus((1280, 1280, 3), 6, activation='softmax')  # FGADR


a=1

# Fine tune with dice loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=[dice_coef])
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'DeeplabV3Plus_FGADR.tf',
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

exit(0)

