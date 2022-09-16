import argparse
import cv2
import os

import numpy as np
import tensorflow as tf


def _pad_to_square(image, long_side=None, border=0):
    h, w, _ = image.shape

    if long_side == None: long_side = max(h, w)

    l_pad = (long_side - w) // 2 + border
    r_pad = (long_side - w) // 2 + border
    t_pad = (long_side - h) // 2 + border
    b_pad = (long_side - h) // 2 + border
    if w % 2 != 0: r_pad = r_pad + 1
    if h % 2 != 0: b_pad = b_pad + 1

    image = np.pad(
        image,
        ((t_pad, b_pad),
         (l_pad, r_pad),
         (0, 0)),
        'constant')

    return image, l_pad, r_pad, t_pad, b_pad


def _get_retina_bb(image):

    # make image greyscale and normalise
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # calculate threshold perform threshold
    threshold = np.mean(image)/3-7
    ret, thresh = cv2.threshold(image, max(0, threshold), 255, cv2.THRESH_BINARY)

    # median filter, erode and dilate to remove noise and holes
    thresh = cv2.medianBlur(thresh, 25)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find mask contour
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # Get bounding box from mask contour
    x, y, w, h = cv2.boundingRect(c)

    # Get mask from contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

    return x, y, w, h, mask


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retina Vessel Segmentation')
    parser.add_argument('-i', '--input_fn', help='Input Retina Image')
    parser.add_argument('-o', '--output_fn', help='Output Segmentation')
    parser.add_argument('-m', '--model_fn', help='Trained Model')
    parser.add_argument('-v', '--verbose', help='Verbose Output', action='store_true', default=False)
    args = vars(parser.parse_args())

    # Debug Overrides
    # args['input_fn'] = '/vol/biomedic3/bh1511/retina/STARE/all-images/im0372.ppm'
    # args['model_fn'] = 'checkpoints/DeeplabV3Plus_DRIVE.tf'

    # Load Image
    image = original_image = cv2.imread(args['input_fn'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Localise and center retina image
    if args['verbose']: print(f'Localising Retina...')
    x, y, w, h, _ = _get_retina_bb(image)
    image = image[y:y + h, x:x + w, :]
    image, l_pad, r_pad, t_pad, b_pad = _pad_to_square(image, border=0)
    image_shape = image.shape
    image = cv2.resize(image, (1024, 1024))

    # Apply CLAHE pre-processing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Load and run model on image
    if args['verbose']: print(f'Predicting Vessels...')
    model = tf.keras.models.load_model(args['model_fn'], compile=False)
    y_pred = model(image[None, ..., None])[0].numpy()

    # Overlay segmentation on original image
    y_pred = cv2.resize(y_pred, image_shape[:-1], interpolation=cv2.INTER_NEAREST)
    y_pred = y_pred[t_pad:y_pred.shape[0]-b_pad, l_pad:y_pred.shape[1]-r_pad]
    canvas = np.zeros(original_image.shape[:-1])
    canvas[y:y + y_pred.shape[0], x:x + y_pred.shape[1]] = y_pred

    # Save predicted segmentation mask to disk
    if args['output_fn'] is None:
        filename, extension = os.path.basename(args['input_fn']).split('.', 1)
        fn = filename + '_segmentation.' + extension
    else:
        fn = args['output_fn']

    if args['verbose']: print(f'Saving Mask File: {fn}')
    cv2.imwrite(fn, canvas*255)
