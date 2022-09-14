import cv2
import os
import tqdm

import numpy as np
import pandas as pd
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
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = label[..., 0][..., None]
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
]


model = tf.keras.models.load_model('checkpoints/DeeplabV3Plus_DRIVE.tf', compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=METRICS)


########## DRIVE #######################################################################################################

root_dir = '/vol/biomedic3/bh1511/retina/DRIVE/preprocessed'
test_df = pd.read_csv(os.path.join(root_dir, 'test_list.csv'))
test_df = root_dir + '/DRIVE_test/' + test_df[['image', 'label']]  # use 'label1' for 2ndHO

test_images = []
test_labels = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
for idx, row in tqdm.tqdm(test_df.iterrows()):
    image = cv2.imread(row['image'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    test_images.append(image)
    test_labels.append(cv2.imread(row['label']))  # use 'label1' for 2ndHO

drive_test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
drive_test_ds = drive_test_ds.map(normalize)
drive_test_ds = drive_test_ds.batch(1)

results = model.evaluate(drive_test_ds)

print(f'--- Results DRIVE ---')
print(f'Dice: {results[10]}')  # 0.76198, 0.77084
print(f'Sensitivity: {results[1] / (results[1] + results[4])}')  # 0.78165, 0.80767
print(f'Specificity: {results[3] / (results[2] + results[3])}')  # 0.97600, 0.97565
print(f'AUC: {results[8]}')  # 0.95197, 0.95944


########## CHASE_DB1 ###################################################################################################

root_dir = '/vol/biomedic3/bh1511/retina/CHASE_DB1/images_processed'
file_id = ['Image_01L', 'Image_01R', 'Image_02L', 'Image_02R', 'Image_03L',
           'Image_03R', 'Image_04L', 'Image_04R', 'Image_05L', 'Image_05R',
           'Image_06L', 'Image_06R', 'Image_07L', 'Image_07R', 'Image_08L',
           'Image_08R', 'Image_09L', 'Image_09R', 'Image_10L', 'Image_10R',
           'Image_11L', 'Image_11R', 'Image_12L', 'Image_12R', 'Image_13L',
           'Image_13R', 'Image_14L', 'Image_14R']
images = [f'{root_dir}/{fid}.jpg' for fid in file_id]
label1 = [f'{root_dir}/{fid}_1stHO.png' for fid in file_id]
label2 = [f'{root_dir}/{fid}_2ndHO.png' for fid in file_id]

chasedb1_images = []
chasedb1_label1 = []
chasedb1_label2 = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
for img_file, lbl1_file, lbl2_file in tqdm.tqdm(zip(images, label1, label2), total=len(images)):
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    chasedb1_images.append(image)
    chasedb1_label1.append(np.clip(cv2.imread(lbl1_file), 0, 1))
    chasedb1_label2.append(np.clip(cv2.imread(lbl2_file), 0, 1))

chasedb1_1stHO_ds = tf.data.Dataset.from_tensor_slices((chasedb1_images, chasedb1_label1))
chasedb1_1stHO_ds = chasedb1_1stHO_ds.map(normalize)
chasedb1_1stHO_ds = chasedb1_1stHO_ds.batch(1)

chasedb1_2ndHO_ds = tf.data.Dataset.from_tensor_slices((chasedb1_images, chasedb1_label2))
chasedb1_2ndHO_ds = chasedb1_2ndHO_ds.map(normalize)
chasedb1_2ndHO_ds = chasedb1_2ndHO_ds.batch(1)

results1 = model.evaluate(chasedb1_1stHO_ds)
results2 = model.evaluate(chasedb1_2ndHO_ds)

print(f'--- Results CHASE_DB1 (1stHO) ---')
print(f'Dice: {results1[10]}')  # 0.72111
print(f'Sensitivity: {results1[1] / (results1[1] + results1[4])}')  # 0.80264
print(f'Specificity: {results1[3] / (results1[2] + results1[3])}')  # 0.97074
print(f'AUC: {results1[8]}')  # 0.95793

print(f'--- Results CHASE_DB1 (2ndHO) ---')
print(f'Dice: {results2[10]}')  # 0.71033
print(f'Sensitivity: {results2[1] / (results2[1] + results2[4])}')  # 0.80751
print(f'Specificity: {results2[3] / (results2[2] + results2[3])}')  # 0.96838
print(f'AUC: {results2[8]}')  # 0.96093


########## STARE #######################################################################################################

root_dir = '/vol/biomedic3/bh1511/retina/STARE'
file_id = ['im0001', 'im0002', 'im0003', 'im0004', 'im0005',
           'im0044', 'im0077', 'im0081', 'im0082', 'im0139',
           'im0162', 'im0163', 'im0235', 'im0236', 'im0239',
           'im0240', 'im0255', 'im0291', 'im0319', 'im0324']

images = [f'{root_dir}/preprocessed/{fid}.png' for fid in file_id]
label1 = [f'{root_dir}/preprocessed/{fid}.ah.png' for fid in file_id]
label2 = [f'{root_dir}/preprocessed/{fid}.vk.png' for fid in file_id]

stare_images = []
stare_label1 = []
stare_label2 = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
for img_file, lbl1_file, lbl2_file in tqdm.tqdm(zip(images, label1, label2), total=len(images)):
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    stare_images.append(image)
    stare_label1.append(np.clip(cv2.imread(lbl1_file), 0, 1))
    stare_label2.append(np.clip(cv2.imread(lbl2_file), 0, 1))

stare_ah_ds = tf.data.Dataset.from_tensor_slices((stare_images, stare_label1))
stare_ah_ds = stare_ah_ds.map(normalize)
stare_ah_ds = stare_ah_ds.batch(1)

stare_vk_ds = tf.data.Dataset.from_tensor_slices((stare_images, stare_label2))
stare_vk_ds = stare_vk_ds.map(normalize)
stare_vk_ds = stare_vk_ds.batch(1)

results1 = model.evaluate(stare_ah_ds)
results2 = model.evaluate(stare_vk_ds)

print(f'--- Results STARE (ah) ---')
print(f'Dice: {results1[10]}')  # 0.73415
print(f'Sensitivity: {results1[1] / (results1[1] + results1[4])}')  # 0.83960
print(f'Specificity: {results1[3] / (results1[2] + results1[3])}')  # 0.97117
print(f'AUC: {results1[8]}')  # 0.96797

print(f'--- Results STARE (vk) ---')
print(f'Dice: {results2[10]}')  # 0.71110
print(f'Sensitivity: {results2[1] / (results2[1] + results2[4])}')  # 0.67829
print(f'Specificity: {results2[3] / (results2[2] + results2[3])}')  # 0.98133
print(f'AUC: {results2[8]}')  # 0.91613
