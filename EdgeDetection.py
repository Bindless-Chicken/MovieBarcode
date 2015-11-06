import numpy as np
import cv2


def black_border_detect(frame, width, height):
    # Switch to greyscale
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny filter to get the edges
    frame_grey = cv2.Canny(frame_grey, 50, 150, apertureSize=3)

    # cv2.imshow("canny", frame_grey)

    # Get the vertical limits
    y1 = iterate_vertical(0, height/2, width, frame_grey, 0)
    y2 = iterate_vertical(height/2, height, width, frame_grey, height)

    # Get the horizontal limits
    x1 = iterate_horizontal(0, width/2, height, frame_grey, 0)
    x2 = iterate_horizontal(width/2, width, height, frame_grey, width)

    return y1, y2, x1, x2


def iterate_vertical(dim1, dim2, width, frame, default):
    max_sum = 0
    y = 0
    for i in range(dim1, dim2):
        temp_sum = 0

        for j in range(0, width):
            temp_sum += frame[i][j]

        if temp_sum > max_sum:
            max_sum = temp_sum
            y = i

    # Only if we have a line long enough
    if max_sum/255 > width/2:
        return y
    else:
        return default


def iterate_horizontal(dim1, dim2, height, frame, default):
    max_sum = 0
    x = 0
    for i in range(dim1, dim2):
        temp_sum = 0

        for j in range(0, height):
            temp_sum += frame[j][i]

        if temp_sum > max_sum:
            max_sum = temp_sum
            x = i

    # Only if we have a line long enough
    if max_sum/255 > height/2:
        return x
    else:
        return default
