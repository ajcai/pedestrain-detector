from __future__ import division
from __future__ import print_function
import re
import glob
import os
import cv2
import random
import string
import numpy as np
from config import *
xrange=range


def get_obj(filename):
    #get person position from a picture annotation
    #one position is represented by a tuple:[Xmin, Ymin,Xmax,Ymax]
    #return a list of those positions
    namerule=r'Image filename : "(.*)"'
    sizerule=r'Image size.*: (\d+) x (\d+) x 3'
    numrule=r'Objects with ground truth : (\d+)'
    corrule=r'"PASperson" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)'
    correg=re.compile(corrule)
    ff=open(filename,'rt')
    objnum=0
    coordinates=[]
    for line in ff:
        line=line.strip()
        namelist=re.findall(namerule,line)
        sizelist=re.findall(sizerule,line)
        numlist=re.findall(numrule,line)
        corlist=re.findall(correg,line)
        if len(namelist)>0:
            imgname=os.path.split(namelist[0])[-1].split(".")[0]
        if len(sizelist)>0:
            imgsize=[int(x) for x in sizelist[0]]
        if len(numlist)>0:
            objnum=int(numlist[0])
        if len(corlist)>0:
            corlist=[int(x) for x in corlist[0]]
            coordinates.append(corlist)
    if objnum != len(coordinates):
        print("error occured in file %s"% filename)
        return {}
    return {imgname:{'imgsize':imgsize,'locs':coordinates}}
def get_all_person(annotation_path):
    obj_pos={}
    for txt_path in glob.glob(os.path.join(annotation_path,'*.txt')):
        obj_pos=dict(get_obj(txt_path),**obj_pos)
    return obj_pos
def normlize_locs(obj_pos):
    #return normlized postion of person
    #{imgid:[[xmin,ymin,xmax,ymax]]}
    norm_locs={}
    for k,v in obj_pos.items():
        locs=[]
        for ll in v['locs']:
            locs.append([ll[0]/v['imgsize'][0],
                ll[1]/v['imgsize'][1],
                ll[2]/v['imgsize'][0],
                ll[3]/v['imgsize'][1]  ])
        norm_locs[k]=locs
    return norm_locs
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

def flip_left_right(img):
    dstimg=np.zeros(img.shape,dtype=img.dtype)
    for c in range(img.shape[1]):
        dstimg[:,c,...]=img[:,img.shape[1]-1-c,...]
    return dstimg

def perpare_person():
    # If image directories don't exist, create them
    if not os.path.isdir(person_path):
        os.makedirs(person_path)
    obj_pos=get_all_person(annotation_path)
    #print obj_pos
    #aspect_ratio=[]
    for img_path in glob.glob(os.path.join(pos_path,'*.png')):
        img=cv2.imread(img_path)
        img_id=os.path.split(img_path)[-1].split(".")[0]
        coordinates=obj_pos[img_id]['locs']
        for i in range(len(coordinates)):
            one_pos=coordinates[i]
            #width=one_pos[0]-one_pos[2]
            #height=one_pos[1]-one_pos[3]
            #aspect_ratio.append(width/height)
            imgroi=img[one_pos[1]:one_pos[3],one_pos[0]:one_pos[2]]
            imgroi=cv2.resize(imgroi,(min_wdw_sz[1],min_wdw_sz[0]))
            flip_img=flip_left_right(imgroi)
            random_name=''.join(random.sample(string.ascii_letters+string.digits, 10))
            cv2.imwrite('%s/%s_person_L.png'%(person_path,img_id),imgroi)
            cv2.imwrite('%s/%s_person_R.png'%(person_path,img_id),flip_img)
    #print aspect_ratio
    #print np.mean(aspect_ratio)

def perpare_noperson():
    # If image directories don't exist, create them
    if not os.path.isdir(background_path):
        os.makedirs(background_path)
    wnd=min_wdw_sz
    for img_path in glob.glob(os.path.join(neg_path,'*')):
        img_name=os.path.split(img_path)[-1].split(".")[0]
        img = cv2.imread(img_path)
        img=cv2.resize(img,(int(img.shape[1]//2),int(img.shape[0]//2)))
        for i in range(10):     #get 10 random patches
            tl_y=int(np.random.uniform(1,img.shape[0]-wnd[0]-1))
            tl_x=int(np.random.uniform(1,img.shape[1]-wnd[1]-1))
            dstimg = img[tl_y:tl_y+wnd[0],tl_x:tl_x+wnd[1]]
            random_name=''.join(random.sample(string.ascii_letters+string.digits, 10))
            cv2.imwrite('%s/%s_%d_background.png'%(background_path,img_name,i),dstimg)
if __name__ == "__main__":
    perpare_person()
    perpare_noperson()

