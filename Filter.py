import numpy as np
from collections import Counter
import cv2


# Set the link between method name and function
def select_method(method_name):
    if method_name == 'mean':
        return cv2.mean
    elif method_name == 'hsv':
        return hsv_dominant
    elif method_name == 'rgb':
        return bgr_dominant
    elif method_name == 'kmean':
        return kmean_dominant


# Return the dominant using the kmean method
def kmean_dominant(frame):
    return generate_kmean_image(frame, 8, True)


# Return the dominant using hsv histogram
def hsv_dominant(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [180, 256, 256], [0, 180, 0, 256, 0, 256])

    # Get max position
    i, j, k = np.unravel_index(hist.argmax(), hist.shape)
    color = np.uint8([[[i, j, k]]])

    return cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]


# Return the dominant using bgr histogram
def bgr_dominant(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [256, 256, 256], [0, 255, 0, 255, 0, 255])

    # Get max position
    i, j, k = np.unravel_index(hist.argmax(), hist.shape)
    color = np.uint8([[[i, j, k]]])

    return color[0][0]


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


# Return the kmean image or the dominant color
def generate_kmean_image(frame, k, dominant=False, display=False):
    z = frame.reshape((-1, 3))
    z = np.float32(z)

    # K-Mean
    criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(frame.shape)

    if display:
        cv2.imshow('res2', res2)

    if dominant:
        return center[most_common(label[0])]

    return res2