import cv2
import os
import glob

# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
# To read file names
import argparse as ap
import glob
import os
from config import *

def img_resize(im):
    h, w, c = im.shape  
    if h > 200 or w > 200:
        (newh, neww) = (100, w * 100 // h) if h < w else (h * 100 //w, 100)
        return cv2.resize(im, (neww, newh))
    return im

def resize(impath = '/new_home/kevin/aj/102675/HOG-SVM-python/data/dataset/plant/neg/'):
    newpath = impath.replace('neg', 'resizeneg')
    if os.path.exists(newpath):
        os.system('rm -rf {}'.format(newpath))
    os.mkdir(newpath)
    for i in glob.glob(impath + '*.png'):
        im = cv2.imread(i)
        im = img_resize(im)
        cv2.imwrite(i.replace('neg', 'resizeneg'),im)

def resize(impath = '/new_home/kevin/aj/102675/HOG-SVM-python/data/dataset/plant/neg/', ori = 'neg', newp = 'resizeneg'):
    newpath = impath.replace(ori, newp)
    if os.path.exists(newpath):
        os.system('rm -rf {}'.format(newpath))
    os.mkdir(newpath)
    for i in glob.glob(impath + '*.png'):
        im = cv2.imread(i)
        im = cv2.resize(im,(100,100))
        cv2.imwrite(i.replace(ori, newp),im)

def extract_test():

    pos_im_path = "../data/dataset/plant/resizepos"
    neg_im_path = "../data/dataset/plant/resizeneg"
	

    print("Calculating the descriptors for the positive samples and saving them")
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        im = imread(im_path, as_gray=True)
        
        fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize=visualize,transform_sqrt=transform_sqrt)
        print(fd)
        print(fd.shape)
        break

if __name__ == "__main__":
    # resize('/new_home/kevin/aj/102675/HOG-SVM-python/data/dataset/plant/neg/', 'neg', 'resizeneg')
    # resize('/new_home/kevin/aj/102675/HOG-SVM-python/data/dataset/plant/pos/', 'pos', 'resizepos')

    extract_test()
