from __future__ import division
from __future__ import print_function
import glob
import os
import random
import string
from skimage.transform import resize
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
from config import *
from trainer import train_classifier
xrange=range


def sliding_window(image, window_size=min_wdw_sz, step_size=step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0]-window_size[0], step_size[0]):
        for x in xrange(0, image.shape[1]-window_size[1], step_size[1]):
            yield image[y:y + window_size[0], x:x + window_size[1]]
def collect_hard_example():
    #This method collect false positive samples in negetive collection.
    # Load the classifier
    clf = joblib.load(model_path)
    print('Colleting hard examples ...')
    counter=0
    for img_path in glob.glob(os.path.join(neg_path,'*')):
        counter+=1
        print('processing {}/{}'.format(counter,len(glob.glob(os.path.join(neg_path,'*')))) )
        img_name=os.path.split(img_path)[-1].split(".")[0]
        img = imread(img_path, as_grey=True)
        img = resize(img,(240,320))
        for im_window in sliding_window(img, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[0] or im_window.shape[1] != min_wdw_sz[1]:
                continue
            # Calculate the HOG features
            fd = hog(im_window, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                cells_per_block=cells_per_block, block_norm=normalize, visualise=visualize)
            fd=fd.reshape(1, -1)
            pred = clf.predict(fd)
            if pred == 1:
                random_name=''.join(random.sample(string.ascii_letters+string.digits, 10))
                fd_path='{}/{}_hard.feat'.format(neg_feat_ph,random_name)
                joblib.dump(fd.flatten(), fd_path)
    print('Collet hard examples finished!')
def retrain_classifier():
    collect_hard_example()
    train_classifier(pos_feat_ph,neg_feat_ph)
if __name__ =='__main__':
    retrain_classifier()
