import argparse
import cv2
from emnist import extract_training_samples, extract_test_samples
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import ensemble

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
           "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

"""
Load EMNIST letters data for handwriting recognition. 
If specified, plots a few samples & corresponding labels.
"""
def load_data(plot=True):
    # extract data from EMNIST [letters]
    images_train, labels_train = extract_training_samples('letters')
    images_test, labels_test = extract_test_samples('letters')

    if plot:
        # randomly plot 25 letters
        f, axarr = plt.subplots(5, 5)
        indices, ctr = random.sample(range(labels_train.shape[0]), 25), 0
        for i in range(5):
            for j in range(5):
                idx = indices[ctr]
                axarr[i, j].imshow(images_train[idx], cmap="gray")
                axarr[i, j].set_title(f"{letters[labels_train[idx] - 1]}")
                ctr += 1
        plt.show()

    # flatten last two dimensions to be (N, 784,)
    return images_train.reshape((images_train.shape[0], images_train.shape[1] * images_train.shape[2])), images_test.reshape((images_test.shape[0], images_test.shape[1] * images_test.shape[2])), labels_train, labels_test

""" 
Load model from saved .joblib
"""
def load(model_path):
    return joblib.load(model_path)

"""
Detect and remove lines from image. (lined paper)
"""
def remove_lines(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # detect lines
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]

    # create a mask of lines
    mask = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    for element in lines:
        if (abs(int(element[0][0]) - int(element[0][2])) > 130 or 
            abs(int(element[0][1]) - int(element[0][3])) > 130):
            cv2.line(mask, (int(element[0][0]), int(element[0][1])), 
                           (int(element[0][2]), int(element[0][3])), (255, 255, 255), 15)

    # remove mask from image
    mask = 255 - mask
    removed = cv2.bitwise_and(im, mask)
    removed[mask == 0] = 255 
    removed = cv2.threshold(cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)[1]
    return removed

""" 
Selects best candidate bounding box from overlapping bounding boxes.
Adapted from pyimagesearch.
"""
def non_max_suppression_fast(boxes, overlapThresh):
   if len(boxes) == 0:
      return []

   # convert bounding boxes to floats
   if boxes.dtype.kind == "i":
      boxes = boxes.astype("float")

   pick = [] # picked list

   x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] # coordinates of the bounding boxes

   area = (x2 - x1 + 1) * (y2 - y1 + 1) # area of bounding boxes
   idxs = np.argsort(y2) # sort bounding boxes by bottom-right y-coordinate

   while len(idxs) > 0:
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)

      # largest (x, y) for top right corner of bounding box
      xx1, yy1 = np.maximum(x1[i], x1[idxs[:last]]), np.maximum(y1[i], y1[idxs[:last]])

      # smallest(x, y) for bottom left corner of bounding box
      xx2, yy2 = np.minimum(x2[i], x2[idxs[:last]]), np.minimum(y2[i], y2[idxs[:last]])

      # compute the width and height of the bounding box
      w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)

      # compute the ratio of overlap
      overlap = (w * h) / area[idxs[:last]]

      # delete all indexes from the index list that have overlap > overlapThresh
      idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

   return boxes[pick].astype("int")

"""
Sorts contours from left-to-right.
"""
def sort_contours(contours):
    return sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

"""
Order letters given bounding rectangles.
"""
def order_letters(bounding_rects):
    ordered = []
    pixel_bound = 60
    # find topmost box
    while (len(bounding_rects) != 0):
        bounding_rects = bounding_rects[bounding_rects[:,1].argsort()]
        top = bounding_rects[0][1]
        current_row = bounding_rects[np.abs(bounding_rects[:,1] - top) < pixel_bound]
        current_row_sorted = current_row[current_row[:,0].argsort()]
        ordered.extend(list(map(lambda box: list(box), current_row_sorted)))
        bounding_rects = np.delete(bounding_rects, np.abs(bounding_rects[:, 1] - top) < pixel_bound, 0)
    return ordered

"""
Get bounding boxes.
"""
def get_bounding_rects(im):
    # get contours
    im_blur = cv2.GaussianBlur(im, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(im_blur, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours & filter by area
    contours = sort_contours(contours)
    contours = filter(lambda cnt: cv2.contourArea(cnt) > 300, contours)

    # find best candidate bounding boxes from contours
    bounding_rects = map(lambda cnt: (cv2.boundingRect(cnt)[0], cv2.boundingRect(cnt)[1], 
                                      cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2], 
                                      cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3]), contours)
    bounding_rects = non_max_suppression_fast(np.array(list(bounding_rects)), 0.3)

    # order letters
    bounding_rects = order_letters(bounding_rects)

    return bounding_rects
  
"""
Crop image given bounding box.
"""
def get_roi(im, rect, pad=5):
    # compute ROI from provided bounding rectangle
    x, y, x2, y2 = rect
    roi = im[y - pad:y2 + pad, x - pad:x2 + pad]
    roi = cv2.resize(roi, (28, 28))
    roi = ~roi # invert to match dataset coloring
    return roi


"""
Draw bounding box.
"""
def draw_bounding_box(im, rect, pad=5):
  x, y, x2, y2 = rect
  cv2.rectangle(im, (x - pad, y - pad), (x2 + pad, y2 + pad), (0, 0, 255), 2)
  return im
