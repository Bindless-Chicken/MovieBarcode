import numpy
import cv2
import sys
import Filter

if len(sys.argv) > 1:
    filename = str(sys.argv[1])
else:
    filename = "input.mp4"

cap = cv2.VideoCapture(filename)

count = 0

frameColor = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow('window-name', frame)
    # fil.color_quantization(frame, 4)
    frameColor.append(Filter.hsv_dominant(frame))

    # cv2.imwrite("frame%d.jpg" % count, frame)
    count += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

img = numpy.zeros([800, count, 3])

for i in range(0, len(frameColor)):
    img[:, i, 0] = frameColor[i][0]
    img[:, i, 1] = frameColor[i][1]
    img[:, i, 2] = frameColor[i][2]

r, g, b = cv2.split(img)
img_bgr = cv2.merge([r, g, b])

cv2.imwrite('color_img.jpg', img_bgr)

cap.release()
