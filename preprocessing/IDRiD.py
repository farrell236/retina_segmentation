import cv2
import os
import tqdm

import numpy as np

from utils import _pad_to_square, _get_retina_bb


root_dir = '/vol/biomedic3/bh1511/retina/IDRID/segmentation/original'

images = [f'{root_dir}/0_Original_Images/IDRiD_{i:02d}.jpg' for i in range(1, 82)]
MAs = [f'{root_dir}/1_Microaneurysms/IDRiD_{i:02d}_MA.tif' for i in range(1, 82)]
HEs = [f'{root_dir}/2_Haemorrhages/IDRiD_{i:02d}_HE.tif' for i in range(1, 82)]
EXs = [f'{root_dir}/3_Hard_Exudates/IDRiD_{i:02d}_EX.tif' for i in range(1, 82)]
SEs = [f'{root_dir}/4_Soft_Exudates/IDRiD_{i:02d}_SE.tif' for i in range(1, 82)]
ODs = [f'{root_dir}/5_Optic_Disc/IDRiD_{i:02d}_OD.tif' for i in range(1, 82)]


res = 1024

t = tqdm.tqdm(zip(images, MAs, HEs, EXs, SEs, ODs), total=len(images))
for img_file, MA_file, HE_file, EX_file, SE_file, OD_file in t:

    img = cv2.imread(img_file)
    MA = np.clip(cv2.imread(MA_file, 0), 0, 1)
    HE = np.clip(cv2.imread(HE_file, 0), 0, 1)
    EX = np.clip(cv2.imread(EX_file, 0), 0, 1)
    SE = np.clip(cv2.imread(SE_file, 0), 0, 1)
    OD = np.clip(cv2.imread(OD_file, 0), 0, 1)
    masks = np.stack([1-(MA+HE+EX+SE+OD), MA, HE, EX, SE, OD], axis=-1)
    masks = np.argmax(masks, axis=-1)[..., None]

    x, y, w, h, _ = _get_retina_bb(img)
    cropped_img = img[y:y + h, x:x + w, :]
    cropped_img = _pad_to_square(cropped_img, border=0)
    cropped_img = cv2.resize(cropped_img, (res, res))

    cropped_mask = masks[y:y + h, x:x + w, :]
    cropped_mask = _pad_to_square(cropped_mask, border=0)
    cropped_mask = cv2.resize(cropped_mask, (res, res), interpolation=cv2.INTER_NEAREST)
    cropped_mask_onehot = np.eye(6)[cropped_mask].astype('uint8')

    cv2.imwrite('output/0_Original_Images/' + os.path.basename(img_file), cropped_img)
    cv2.imwrite('output/1_Microaneurysms/' + os.path.basename(MA_file).replace('.tif', '.png'), 255 * cropped_mask_onehot[..., 1])
    cv2.imwrite('output/2_Haemorrhages/' + os.path.basename(HE_file).replace('.tif', '.png'), 255 * cropped_mask_onehot[..., 2])
    cv2.imwrite('output/3_Hard_Exudates/' + os.path.basename(EX_file).replace('.tif', '.png'), 255 * cropped_mask_onehot[..., 3])
    cv2.imwrite('output/4_Soft_Exudates/' + os.path.basename(SE_file).replace('.tif', '.png'), 255 * cropped_mask_onehot[..., 4])
    cv2.imwrite('output/5_Optic_Disc/' + os.path.basename(OD_file).replace('.tif', '.png'), 255 * cropped_mask_onehot[..., 5])
    cv2.imwrite('output/onehot_masks/' + os.path.basename(img_file).replace('.jpg', '_mask.png'), cropped_mask)
