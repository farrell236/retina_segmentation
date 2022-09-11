import cv2
import glob
import os
import tqdm

from utils import _pad_to_square, _get_retina_bb, rgb_clahe


root_dir = '/vol/vipdata/data/retina/kaggle-diabetic-retinopathy-detection'
image_files = sorted(glob.glob(os.path.join(root_dir, 'train/*')))

# remove broken/corrupt images
image_files.remove(f'{root_dir}/train/492_right.jpeg')
# image_files.remove(f'{root_dir}/test/27096_right.jpeg')
# image_files.remove(f'{root_dir}/test/25313_right.jpeg')

for image_file in tqdm.tqdm(image_files):

    # Load Image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Localise and center retina image
    x, y, w, h, _ = _get_retina_bb(image)
    image = image[y:y + h, x:x + w, :]
    image = _pad_to_square(image, border=0)
    image = cv2.resize(image, (1024, 1024))

    # Apply CLAHE pre-processing
    image = rgb_clahe(image)

    # Save image
    cv2.imwrite(f'train_1024x1024/{os.path.basename(image_file)}', image)
