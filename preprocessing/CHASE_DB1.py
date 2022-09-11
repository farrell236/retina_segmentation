import cv2
import os
import tqdm

from utils import _pad_to_square, _get_retina_bb


root_dir = '/vol/biomedic3/bh1511/retina/CHASE_DB1/images'
file_id = ['Image_01L', 'Image_01R', 'Image_02L', 'Image_02R', 'Image_03L',
           'Image_03R', 'Image_04L', 'Image_04R', 'Image_05L', 'Image_05R',
           'Image_06L', 'Image_06R', 'Image_07L', 'Image_07R', 'Image_08L',
           'Image_08R', 'Image_09L', 'Image_09R', 'Image_10L', 'Image_10R',
           'Image_11L', 'Image_11R', 'Image_12L', 'Image_12R', 'Image_13L',
           'Image_13R', 'Image_14L', 'Image_14R']
images = [f'{root_dir}/{fid}.jpg' for fid in file_id]
label1 = [f'{root_dir}/{fid}_1stHO.png' for fid in file_id]
label2 = [f'{root_dir}/{fid}_2ndHO.png' for fid in file_id]

res = 1024


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
    cv2.imwrite('output/' + os.path.basename(lbl1_file), cropped_lbl1)
    cv2.imwrite('output/' + os.path.basename(lbl2_file), cropped_lbl2)
