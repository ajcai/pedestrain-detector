import configparser as cp
import os
import shutil
from annotationutil import perpare_person,perpare_noperson
from feature_extractor import extract_features
from trainer import train_classifier
from retrainer import retrain_classifier
from tester import detect_dataset
config = cp.RawConfigParser()
config.read('../data/config/config.cfg')
person_path=config.get("train","person_path")
background_path=config.get("train","background_path")
pos_feat_ph = config.get("paths", "pos_feat_ph")
neg_feat_ph = config.get("paths", "neg_feat_ph")

#sample image patches
if not os.path.exists(person_path):
    perpare_person()
if not os.path.exists(background_path):
    perpare_noperson()
orient=[3,4,6,9]
for i in orient:
    #modify configuration file
    config.set('hog','orientations',i)
    config.write(open('../data/config/config.cfg'.format(i),'w'))
    #extract features
    shutil.rmtree(pos_feat_ph)
    shutil.rmtree(neg_feat_ph)
    extract_features()
    #train
    train_classifier()
    #retrain
    retrain_classifier()
    #save detections
    detect_dataset('../data/detections/detections_orient_{}.data'.format(i))
