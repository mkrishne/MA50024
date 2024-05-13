import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.settrace

import cv2
# draw the bounding boxes for face detection
def draw_bbox(bounding_boxes, image):
    for i in range(len(bounding_boxes)):
        x1, y1, x2, y2 = bounding_boxes[i]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 0, 255), 2)
    
    return image

# plot the facial landmarks
def plot_landmarks(landmarks, image):
    for i in range(len(landmarks)):
        for p in range(landmarks[i].shape[0]):
            cv2.circle(image, 
                      (int(landmarks[i][p, 0]), int(landmarks[i][p, 1])),
                      2, (0, 0, 255), -1, cv2.LINE_AA)
    return image


import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

# computation device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device_cpu = torch.device('cpu')
# create the MTCNN model, `keep_all=True` returns all the detected faces 
mtcnn = MTCNN(keep_all=True, device=device_cpu)


# read the image 
image = Image.open('train_small/0.jpg').convert('RGB')
# create an image array copy so that we can use OpenCV functions on it
image_array = np.array(image, dtype=np.float32)
# cv2 image color conversion
image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

# the detection module returns the bounding box coordinates and confidence ...
# ... by default, to get the facial landmarks, we have to provide ...
# ... `landmarks=True`
bounding_boxes, conf, landmarks = mtcnn.detect(image, landmarks=True)
# print(f"Bounding boxes shape: {bounding_boxes.shape}")
# print(f"Landmarks shape: {landmarks.shape}")


# draw the bounding boxes around the faces
image_array = draw_bbox(bounding_boxes, image_array)
# plot the facial landmarks
image_array = plot_landmarks(landmarks, image_array)

# set the save path
save_path = 'output_facenet/0.jpg'
# save image
cv2.imwrite(save_path, image_array)
# shoe the image
cv2.imshow('Image', image_array/255.0)
cv2.waitKey(0)