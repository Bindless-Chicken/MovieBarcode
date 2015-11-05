import numpy as np
from collections import Counter
import cv2


def get_dominant_color(frame):
    # Get K-mean image
    color = color_quantization(frame, 4)
    # print color
    return color


# Return the frame with Color Quantization
def color_quantization(frame, k):
    z = frame.reshape((-1, 3))
    z = np.float32(z)

    # K-Mean
    criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    dominant = most_common(label[0])

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(frame.shape)

    cv2.imshow('res2', res2)

    return center[dominant]


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]
