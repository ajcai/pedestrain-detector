from __future__ import division
from __future__ import print_function
xrange=range
import re
import glob
import os
import cv2
import random
import string
import numpy as np
import argparse as ap
from matplotlib import pyplot as plt
from skimage.transform import resize,pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
from config import *
from annotationutil import get_all_person,normlize_locs
from nms import nms


def sliding_window(image, window_size, step_size):
    for y in xrange(0, image.shape[0]-window_size[0], step_size[0]):
        for x in xrange(0, image.shape[1]-window_size[1], step_size[1]):
            yield (x, y, image[y:y + window_size[0], x:x + window_size[1]])

def detect_an_image(img_path,visualize_det=False):
    #return detections of an image,
    #[[xmin,ymin,xmax,ymax,confidence]],coordinates are normalized
    # Read the image
    img_id=os.path.split(img_path)[-1].split(".")[0]
    im = imread(img_path, as_grey=True)
    im = resize(im,(300,300))
    #min_wdw_sz = (100,40)
    #step_size = (10, 10)
    downscale = 1.25

    # Load the classifier
    clf = joblib.load(model_path)
    # list to store all ditections
    detections=[]
    # List to store the predictions
    predictions = []
    # Downscale the image and iterate
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        cd = []
        if im_scaled.shape[0] < min_wdw_sz[0] or im_scaled.shape[1] < min_wdw_sz[1]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[0] or im_window.shape[1] != min_wdw_sz[1]:
                continue
            # Calculate the HOG features
            fd = hog(im_window, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                cells_per_block=cells_per_block, block_norm=normalize, visualise=visualize)
            fd=fd.reshape(1, -1)
            pred = clf.predict(fd)
            norm_wnd=(x/im_scaled.shape[1], y/im_scaled.shape[0], 
                    (x+min_wdw_sz[1])/im_scaled.shape[1],
                    (y+min_wdw_sz[0])/im_scaled.shape[0])
            detections.append((norm_wnd[0],norm_wnd[1],norm_wnd[2],norm_wnd[3],clf.decision_function(fd)[0]) )
            if pred == 1:
                #print("Detection:: Location -> ({}, {})".format(x, y))
                predictions.append((norm_wnd[0],norm_wnd[1],norm_wnd[2],norm_wnd[3],clf.decision_function(fd)[0]) )
                cd.append(predictions[-1])
            if visualize_det:
                clone = im_scaled.copy()
                for x1, y1, x2, y2, _  in cd:
                    # Draw the predictions at this scale
                    cv2.rectangle(clone, (int(x1*im_scaled.shape[1]), int(y1*im_scaled.shape[0])), 
                        (int(x2*im_scaled.shape[1]), int(y2*im_scaled.shape[0])), (0, 0, 0), thickness=2)
                cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                    im_window.shape[0]), (255, 255, 255), thickness=2)
                cv2.imshow("Sliding Window in Progress", clone)
                cv2.waitKey(20)
    if visualize_det:
        # Display the results before performing NMS
        clone = im.copy()
        box_persons(im,predictions,"Before NMS") #Raw predictions before NMS
        # Perform Non Maxima Suppression
        predictions = nms(predictions, iou_threshold)
        # Display the results after performing NMS
        box_persons(clone,predictions,"After NMS")
        if cv2.waitKey(0) & 0xFF==ord('q'):
             cv2.destroyAllWindows()
    return detections

def detect_dataset(detections_path=detections_path):
    if not os.path.isdir(os.path.split(detections_path)[0]):
        os.makedirs(os.path.split(detections_path)[0])
    dataset_dects={}
    #test detection on positive 
    print("testing on positive images")
    for img_path in glob.glob(os.path.join(test_pos_path,'*')):
        img_id=os.path.split(img_path)[-1].split(".")[0]
        dataset_dects[img_id]=detect_an_image(img_path)
    #test detection on negative 
    print("testing on negative images")
    for img_path in glob.glob(os.path.join(test_neg_path,'*')):
        img_id=os.path.split(img_path)[-1].split(".")[0]
        dataset_dects[img_id]=detect_an_image(img_path)

    joblib.dump(dataset_dects, detections_path)
    print("detections saved to {}".format(detections_path))
    return dataset_dects


def box_persons(im,boxes,title='title'):
    for box in boxes:
        x_tl=box[0]
        y_tl=box[1]
        x_br=box[2]
        y_br=box[3]
        # Draw the predictions
        cv2.rectangle(im, (int(x_tl*im.shape[1]), int(y_tl*im.shape[0])),
             (int(x_br*im.shape[1]), int(y_br*im.shape[0])), (0, 0, 0), thickness=2)
    cv2.imshow(title, im)
    cv2.waitKey()


if __name__=="__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--det_path", help="Path to the detections directory", default=detections_path)
    args = vars(parser.parse_args())
    det_path = args["det_path"]
    detect_dataset(det_path)

