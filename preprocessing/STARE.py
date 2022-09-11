import cv2
import os
import tqdm

from utils import _pad_to_square, _get_retina_bb


root_dir = '/vol/biomedic3/bh1511/retina/STARE'
file_id = ['im0001', 'im0002', 'im0003', 'im0004', 'im0005',
           'im0044', 'im0077', 'im0081', 'im0082', 'im0139',
           'im0162', 'im0163', 'im0235', 'im0236', 'im0239',
           'im0240', 'im0255', 'im0291', 'im0319', 'im0324']

images = [f'{root_dir}/all-images-1/{fid}.png' for fid in file_id]
label1 = [f'{root_dir}/segmentation/labels-ah/{fid}.ah.ppm' for fid in file_id]
label2 = [f'{root_dir}/segmentation/labels-vk/{fid}.vk.ppm' for fid in file_id]


# images are RGB (605, 700, 3)
# labels are (605, 700) in dimension and between (0,255) intensity

res = 1024

import matplotlib.pyplot as plt

for img_file, lbl1_file, lbl2_file in tqdm.tqdm(zip(images, label1, label2)):

    img, lbl1, lbl2 = cv2.imread(img_file), cv2.imread(lbl1_file), cv2.imread(lbl2_file)

    x, y, w, h, _ = _get_retina_bb(img)
    cropped_img = img[y:y + h, x:x + w, :]
    cropped_img = _pad_to_square(cropped_img, border=0)
    cropped_img = cv2.resize(cropped_img, (res, res))

    cropped_lbl1 = lbl1[y:y + h, x:x + w]
    cropped_lbl1 = _pad_to_square(cropped_lbl1, border=0)
    cropped_lbl1 = cv2.resize(cropped_lbl1, (res, res), interpolation=cv2.INTER_NEAREST)

    cropped_lbl2 = lbl2[y:y + h, x:x + w]
    cropped_lbl2 = _pad_to_square(cropped_lbl2, border=0)
    cropped_lbl2 = cv2.resize(cropped_lbl2, (res, res), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite('output/' + os.path.basename(img_file), cropped_img)
    cv2.imwrite('output/' + os.path.basename(lbl1_file.replace('ppm', 'png')), cropped_lbl1)
    cv2.imwrite('output/' + os.path.basename(lbl2_file.replace('ppm', 'png')), cropped_lbl2)
