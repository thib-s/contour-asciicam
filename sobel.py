import cv2
import numpy as np


def compute_gradients(img1, kersize=3):
    img_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=kersize)
    img_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=kersize)
    return img_x, img_y


def get_sobel(img_x, img_y):
    vfunc = np.vectorize(lambda t: t ** 2)

    img = np.vectorize(np.math.sqrt)(np.add(
        vfunc(img_x),
        vfunc(img_y)
    )
    )
    return img


def normalize(img):
    mini = np.min(img)
    maxi = np.max(img)
    return np.vectorize(lambda x: int(255 * (x - mini)/(maxi - mini)))(img)