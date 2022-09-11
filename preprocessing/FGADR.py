import cv2
import glob
import os
import tqdm

import numpy as np


root_dir = '/vol/biomedic3/bh1511/retina/FGADR/Seg-set'

files = glob.glob(os.path.join(root_dir, 'Original_Images/*.png'))
files.sort()

patient_ids = [os.path.basename(x).split('.')[0] for x in files]

images = [f'{root_dir}/Original_Images/{pid}.png' for pid in patient_ids]
MAs = [f'{root_dir}/Microaneurysms_Masks/{pid}.png' for pid in patient_ids]
HEs = [f'{root_dir}/Hemohedge_Masks/{pid}.png' for pid in patient_ids]
EXs = [f'{root_dir}/HardExudate_Masks/{pid}.png' for pid in patient_ids]
SEs = [f'{root_dir}/SoftExudate_Masks/{pid}.png' for pid in patient_ids]
ODs = [f'{root_dir}/OpticalDisk_Masks/{pid}.png' for pid in patient_ids]


t = tqdm.tqdm(zip(images, MAs, HEs, EXs, SEs, ODs), total=len(images))
for img_file, MA_file, HE_file, EX_file, SE_file, OD_file in t:

    # img = cv2.imread(img_file)
    MA = np.clip(cv2.imread(MA_file, 0), 0, 1)
    HE = np.clip(cv2.imread(HE_file, 0), 0, 1)
    EX = np.clip(cv2.imread(EX_file, 0), 0, 1)
    SE = np.clip(cv2.imread(SE_file, 0), 0, 1)
    OD = np.clip(cv2.imread(OD_file, 0), 0, 1)
    masks = np.stack([1-(MA+HE+EX+SE+OD), MA, HE, EX, SE, OD], axis=-1)
    masks = np.argmax(masks, axis=-1)

    cv2.imwrite(f'idrid_labels/{os.path.basename(img_file).split(".")[0]}.png', masks)
