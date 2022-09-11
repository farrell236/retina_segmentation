import cv2
import tqdm

import tensorflow as tf


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


def normalize(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label[..., 0], depth=6)
    return image, label


METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    dice_coef,
]  # Metric calculation ignores Background class


# model = tf.keras.models.load_model('checkpoints/DeeplabV3Plus_IDRiD.tf', compile=False)
model = tf.keras.models.load_model('checkpoints/DeeplabV3Plus_FGADR.tf', compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=METRICS)


class_name = {  # For Reference
    1: 'Microaneurysms',
    2: 'Hemohedge',
    3: 'Hard Exudate',
    4: 'Soft Exudate',
    5: 'Optical Disk',
}


########## IDRiD #######################################################################################################

root_dir = '/vol/biomedic3/bh1511/retina/IDRID/segmentation/preprocessed'
images = [f'{root_dir}/0_Original_Images/IDRiD_{i:02d}.jpg' for i in range(1, 82)]
labels = [f'{root_dir}/onehot_masks/IDRiD_{i:02d}_mask.png' for i in range(1, 82)]

test_images = []
test_labels = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
for idx in tqdm.tqdm(range(54, 81)):
    image = cv2.imread(images[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    test_images.append(image)
    test_labels.append(cv2.imread(labels[idx]))

idrid_test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
idrid_test_ds = idrid_test_ds.map(normalize)
idrid_test_ds = idrid_test_ds.batch(1)

y_pred_all = []
y_true_all = []
for image, label in tqdm.tqdm(idrid_test_ds):
    y_pred_all.append(model(image))
    y_true_all.append(label)
y_pred_all = tf.concat(y_pred_all, axis=0)
y_true_all = tf.concat(y_true_all, axis=0)

results = [dice_coef(y_true_all[..., 1:], y_pred_all[..., 1:])]
for metric in METRICS[:-1]:
    results.append(metric(y_true_all[..., 1:], y_pred_all[..., 1:]))

print(f'--- Results IDRiD (Model: IDRiD) ---')
print(f'Dice: {results[0]}')  # 0.74666
print(f'Sensitivity: {results[1] / (results[1] + results[4])}')  # 0.47974
print(f'Specificity: {results[3] / (results[2] + results[3])}')  # 0.99633
print(f'AUC: {results[8]}')  # 0.11045

print(f'--- Results IDRiD (Model: FGADR) ---')
print(f'Dice: {results[0]}')  # 0.26545
print(f'Sensitivity: {results[1] / (results[1] + results[4])}')  # 0.27606
print(f'Specificity: {results[3] / (results[2] + results[3])}')  # 0.99370
print(f'AUC: {results[8]}')  # 0.10063


########## FGADR #######################################################################################################

from patient_ids import patient_ids
root_dir = '/vol/biomedic3/bh1511/retina/FGADR/Seg-set'
images = [f'{root_dir}/Original_Images/{i}.png' for i in patient_ids]
labels = [f'{root_dir}/idrid_labels/{i}.png' for i in patient_ids]

test_images = []
test_labels = []

res = 1024
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
for idx in tqdm.tqdm(range(1400, 1494)):
    image = cv2.imread(images[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = cv2.resize(image, (res, res))
    test_images.append(image)
    label = cv2.resize(cv2.imread(labels[idx]), (res, res), interpolation=cv2.INTER_NEAREST)
    test_labels.append(label)

FGADR_test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
FGADR_test_ds = FGADR_test_ds.map(normalize)
FGADR_test_ds = FGADR_test_ds.batch(1)

y_pred_all = []
y_true_all = []
for image, label in tqdm.tqdm(FGADR_test_ds):
    y_pred_all.append(model(image))
    y_true_all.append(label)
y_pred_all = tf.concat(y_pred_all, axis=0)
y_true_all = tf.concat(y_true_all, axis=0)

results = [dice_coef(y_true_all[..., 1:], y_pred_all[..., 1:])]
for metric in METRICS[:-1]:
    results.append(metric(y_true_all[..., 1:], y_pred_all[..., 1:]))

print(f'--- Results FGADR (Model: IDRiD) ---')
print(f'Dice: {results[0]}')  # 0.58443
print(f'Sensitivity: {results[1] / (results[1] + results[4])}')  # 0.44908
print(f'Specificity: {results[3] / (results[2] + results[3])}')  # 0.99963
print(f'AUC: {results[8]}')  # 0.02949

print(f'--- Results FGADR (Model: FGADR) ---')
print(f'Dice: {results[0]}')  # 0.82566
print(f'Sensitivity: {results[1] / (results[1] + results[4])}')  # 0.75133
print(f'Specificity: {results[3] / (results[2] + results[3])}')  # 0.99971
print(f'AUC: {results[8]}')  # 0.03387
