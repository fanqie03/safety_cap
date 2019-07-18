import os
import numpy as np
from scipy import misc


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def crop_with_margin(img, det, expands, image_size):
    img_size = np.asarray(img.shape)[0:2]
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - det[2] * expands, 0)
    bb[1] = np.maximum(det[1] - det[3] * expands, 0)
    bb[2] = np.minimum(det[2] + det[0] + det[2] * expands, img_size[1])
    bb[3] = np.minimum(det[3] + det[1] + det[3] * expands, img_size[0])
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    return scaled
