import numpy
import cv2
import sys
import Filter
import EdgeDetection as ED
import argparse
from tqdm import tqdm

# Create arg parser
parser = argparse.ArgumentParser(description='Transform a video into its barcode')
parser.add_argument('-f', '--filename', help='Video path (default: input.mp4)', default='input.mp4')
parser.add_argument('-m', '--method', help='Method used to generate the barcode (default: hsv)',
                    choices=['mean', 'hsv', 'rgb', 'kmean'], default='hsv')
parser.add_argument('-t', '--timestamp', help='Timestamp at which to start the analysing(default: 0)',
                    default='0', type=int)
args = parser.parse_args()

# Read video file
cap = cv2.VideoCapture(args.filename)
if not cap.isOpened():
    sys.exit('File ' + args.filename + ' not found!')

# Retrieve various data about the video
nb_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare the final image
final_image = numpy.zeros([800, nb_of_frames, 3], dtype=numpy.uint8)

# Sample color image
sample_image = numpy.zeros([200, 200, 3], dtype=numpy.uint8)

# Edge Detection
x1, y1, x2, y2 = ED.detect_black_edges(cap)

# Set to the correct frame according to the timestamp
fpms = fps/1000
cap.set(cv2.CAP_PROP_POS_FRAMES, args.timestamp*fpms)

# For each Frame
for i in tqdm(range(nb_of_frames)):
    ret, frame = cap.read()

    # If this is an empty frame aka. last one
    if not ret:
        break

    # Crop the frame
    frame = frame[y1:y2, x1:x2]

    cv2.imshow('Direct Video', frame)

    # Get color
    frame_color = Filter.select_method(args.method)(frame)
    sample_image[:, :] = frame_color
    cv2.imshow('Current Dominant Color', sample_image)

    # Fill final image
    final_image[:, i] = frame_color

    # Force quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.imwrite('color_img.jpg', final_image)

cap.release()
