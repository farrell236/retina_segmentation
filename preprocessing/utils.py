import cv2

import matplotlib.pyplot as plt
import numpy as np


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

    return image


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


def _get_retina_bb2(image, skips=4):
    '''
    Experimental Retina Bounding Box detector based on Convexity Defect Points
    '''

    # make image greyscale and normalise
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # calculate threshold perform threshold
    threshold = np.mean(image)/3-7
    ret, thresh = cv2.threshold(image, max(0, threshold), 255, cv2.THRESH_BINARY)

    # median filter, erode and dilate to remove noise and holes
    thresh = cv2.medianBlur(thresh, 25)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    # Get convexity defect points
    hull = cv2.convexHull(cnt, returnPoints=False)
    hull[::-1].sort(axis=0)
    defects = cv2.convexityDefects(cnt, hull)

    ConvexityDefectPoint = []
    for i in range(0, defects.shape[0], skips):
        s, e, f, d = defects[i, 0]
        ConvexityDefectPoint.append(tuple(cnt[f][0]))

    # Get minimum enclosing circle as retina estimate
    (x, y), radius = cv2.minEnclosingCircle(np.array(ConvexityDefectPoint))

    # Get mask from contour
    mask = np.zeros_like(image)
    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)

    # return (x, y, w, h) bounding box
    return int(x - radius), int(y - radius), int(2 * radius - 1), int(2 * radius - 1), mask


def rgb_clahe(image, clipLimit=2.0, tileGridSize=(16, 16)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


if __name__ == '__main__':

    # image_file = '/vol/biomedic3/bh1511/retina/IDRID/segmentation/0_Original_Images/IDRiD_65.jpg'
    # image_file = '/vol/biomedic3/bh1511/retina/CHASE_DB1/images/Image_08R.jpg'
    # image_file = '/vol/vipdata/data/retina/kaggle-diabetic-retinopathy-detection/train/16_right.jpeg'
    # image_file = '/vol/vipdata/data/retina/IDRID/a_segmentation/images/train/IDRiD_01.jpg'
    image_file = 'preprocessing/Image_10L.png'

    # Load Image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image); plt.show()

    # Localise and center retina image
    x, y, w, h, _ = _get_retina_bb(image)
    image = image[y:y + h, x:x + w, :]
    image = _pad_to_square(image, border=0)
    image = cv2.resize(image, (1024, 1024))

    # Apply CLAHE pre-processing
    image = rgb_clahe(image)

    # Display or save image
    plt.imshow(image); plt.show()
    # cv2.imwrite('processed_image.png', image)
