'''
Set the config variable.
'''

import configparser as cp
import json

config = cp.RawConfigParser()
config.read('../data/config/config.cfg')

min_wdw_sz = json.loads(config.get("hog","min_wdw_sz"))
step_size = json.loads(config.get("hog", "step_size"))
orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
normalize = config.get("hog", "normalize")
pos_path=config.get("train","pos_path")
neg_path=config.get("train","neg_path")
annotation_path=config.get("train","annotation_path")
person_path=config.get("train","person_path")
background_path=config.get("train","background_path")
pos_feat_ph = config.get("paths", "pos_feat_ph")
neg_feat_ph = config.get("paths", "neg_feat_ph")
model_path = config.get("paths", "model_path")
iou_threshold = config.getfloat("iou", "threshold")
detections_path = config.get("test","detections_path")
test_pos_path = config.get("test","test_pos_path")
test_neg_path = config.get("test","test_neg_path")