import argparse
import cv2
from emnist import extract_training_samples, extract_test_samples
from emnist_net import EmnistNet
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import ensemble
import torch


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
Train a random forest on train data, evaluate on test data, and save, if specified.
"""
def train(x_train, x_test, y_train, y_test, model, save=True, batch_size=32, num_epochs=50, log_interval=100):
    # set labels to be 0-indexed
    y_train = y_train - 1
    y_test = y_test - 1

    if model == "random_forest":
        # fit a classifier
        classifier = ensemble.RandomForestClassifier(n_estimators=100)
        classifier.fit(x_train, y_train)

        # print score on test data
        scores = classifier.score(x_test, y_test)
        print(f"Classification Accuracy (Test): {scores}")

        # save models
        if save:
            joblib.dump(classifier, "emnist_random_forest.joblib")
        
        return classifier
    elif model == "neural_net":
        # create model
        model = EmnistNet()
        model.train()

        # create train batches
        train_idx = np.array(list(range(len(x_train))))
        np.random.shuffle(train_idx)
        train_batches = np.split(train_idx, batch_size)

        # train model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_batches):
                ims, labels = x_train[batch], y_train[batch]
                ims = ims.reshape(ims.shape[:-1] + (1, 28, 28))
                ims, labels = torch.Tensor(ims), torch.Tensor(labels).type(torch.LongTensor)
                optimizer.zero_grad()
                outputs = model(ims)
                loss = torch.nn.functional.nll_loss(outputs, labels)
                optimizer.step()

                # log
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                          epoch, batch_idx * len(ims), len(x_train),
                          100. * batch_idx / len(x_train), loss.item()))

        # create test batches
        test_idx = np.array(list(range(len(x_test))))
        np.random.shuffle(test_idx)
        test_batches = np.split(test_idx, batch_size)

        # test model
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch in test_batches:
                ims, labels = x_test[batch], y_test[batch]
                ims = ims.reshape(ims.shape[:-1] + (1, 28, 28))
                ims, labels = torch.Tensor(ims), torch.Tensor(labels).type(torch.LongTensor)
                outputs = model(ims)

                # sum up batch loss
                test_loss += torch.nn.functional.nll_loss(outputs, labels, reduction='sum').item()

                # get the index of the max log-probability
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                
        test_loss /= len(x_test)

        # log test results
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
              test_loss, correct, len(x_test),
              100. * correct / len(x_test)))
            
        if save:
            torch.save(model.state_dict(), "emnist_neural_net.pt")
    else:
        raise Exception(f"Invalid model type: {model}. Please choose from 'random_forest' or 'neural_net'.")

""" 
Load model from saved path
"""
def load(model_path):
    if model_path == "emnist_random_forest.joblib":
        return joblib.load(model_path)
    elif model_path == "emnist_neural_net.pt":
        model = EmnistNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    else:
        raise Exception(f"Cannot load model from provided model path: {model_path}.")

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
Return text translation of provided image.
"""
def predict(im_path, model, pad=5):
    im = cv2.imread(im_path)

    # remove lines
    im = remove_lines(im)
    copy = im.copy()

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

    pred = ""
    for i, rect in enumerate(bounding_rects, 0):
        # compute ROI from provided bounding rectangle
        x, y, x2, y2 = rect
        roi = copy[y - pad:y2 + pad, x - pad:x2 + pad]
        roi = cv2.resize(roi, (28, 28))
        roi = ~roi # invert to match dataset coloring

        cv2.imwrite(f"segmented/roi{i}.jpg", roi) # write ROI to folder for debugging

        # predict letter for ROI
        pred += letters[model.predict([roi.flatten()])[0]]

        # draw bounding box
        cv2.rectangle(im, (x - pad, y - pad), (x2 + pad, y2 + pad), (0, 0, 255), 2)

    cv2.imwrite("bounding_boxes.jpg", im) # write bounding box for debugging
    
    return pred

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Train & use a model to translate handwriting to text.")
    parser.add_argument("--train", action="store", type=str, choices=["random_forest", "neural_net"], required=False)
    parser.add_argument("--plot-data", action="store_true", required=False)
    parser.add_argument("--input", action="store", type=str, required=False)
    parser.add_argument("--classifier", action="store", type=str, choices=["emnist_random_forest.joblib", "emnist_neural_net.pt"], required=False)
    args = parser.parse_args()

    # train on emnist
    if args.train:
        x_train, x_test, y_train, y_test = load_data(plot=args.plot_data)
        classifier = train(x_train, x_test, y_train, y_test, args.train, save=True)

    # predict provided input
    if args.input:
        if not args.train:
            classifier = load(args.classifier)
        pred = predict(args.input, classifier)
        print(f"Prediction: {pred}")
