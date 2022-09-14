import cv2
import imageio
import os
import tqdm

import numpy as np

from utils import _pad_to_square, _get_retina_bb

a=1

root_dir = '/vol/biomedic3/bh1511/retina/DRIVE/DRIVE'

train_images = [f'{root_dir}/training/images/{x}_training.tif' for x in range(21, 41)]
train_labels = [f'{root_dir}/training/1st_manual/{x}_manual1.gif' for x in range(21, 41)]

test_images = [f'{root_dir}/test/images/{x:02d}_test.tif' for x in range(1, 21)]
test_labels1 = [f'{root_dir}/test/1st_manual/{x:02d}_manual1.gif' for x in range(1, 21)]
test_labels2 = [f'{root_dir}/test/2nd_manual/{x:02d}_manual2.gif' for x in range(1, 21)]


res = 1024


for img_file, lbl_file in tqdm.tqdm(zip(train_images, train_labels)):

    img, lbl = cv2.imread(img_file), imageio.imread(lbl_file)[..., None]

    x, y, w, h, _ = _get_retina_bb(img)
    cropped_img = img[y:y + h, x:x + w, :]
    cropped_img = _pad_to_square(cropped_img, border=0)
    cropped_img = cv2.resize(cropped_img, (res, res))

    cropped_lbl = lbl[y:y + h, x:x + w]
    cropped_lbl = _pad_to_square(cropped_lbl, border=0)
    cropped_lbl = cv2.resize(cropped_lbl, (res, res), interpolation=cv2.INTER_NEAREST)
    cropped_lbl = np.clip(cropped_lbl, 0, 1)

    cv2.imwrite('DRIVE_train/' + os.path.basename(img_file.replace('tif', 'png')), cropped_img)
    cv2.imwrite('DRIVE_train/' + os.path.basename(lbl_file.replace('gif', 'png')), cropped_lbl)


for img_file, lbl1_file, lbl2_file in tqdm.tqdm(zip(test_images, test_labels1, test_labels2)):

    img, lbl1, lbl2 = cv2.imread(img_file), imageio.imread(lbl1_file)[..., None], imageio.imread(lbl2_file)[..., [0]]

    x, y, w, h, _ = _get_retina_bb(img)
    cropped_img = img[y:y + h, x:x + w, :]
    cropped_img = _pad_to_square(cropped_img, border=0)
    cropped_img = cv2.resize(cropped_img, (res, res))

    cropped_lbl1 = lbl1[y:y + h, x:x + w]
    cropped_lbl1 = _pad_to_square(cropped_lbl1, border=0)
    cropped_lbl1 = cv2.resize(cropped_lbl1, (res, res), interpolation=cv2.INTER_NEAREST)
    cropped_lbl1 = np.clip(cropped_lbl1, 0, 1)

    cropped_lbl2 = lbl2[y:y + h, x:x + w]
    cropped_lbl2 = _pad_to_square(cropped_lbl2, border=0)
    cropped_lbl2 = cv2.resize(cropped_lbl2, (res, res), interpolation=cv2.INTER_NEAREST)
    cropped_lbl2 = np.uint8(cropped_lbl2 > 128)  # This label is [4, 252] instead of [0, 255]

    cv2.imwrite('DRIVE_test/' + os.path.basename(img_file.replace('tif', 'png')), cropped_img)
    cv2.imwrite('DRIVE_test/' + os.path.basename(lbl1_file.replace('gif', 'png')), cropped_lbl1)
    cv2.imwrite('DRIVE_test/' + os.path.basename(lbl2_file.replace('gif', 'png')), cropped_lbl2)
