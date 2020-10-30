# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
from config import *
import glob
import numpy as np

def auto_compute():
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--posfeat", help="Path to the positive features directory", required=True)
    parser.add_argument('-n', "--negfeat", help="Path to the negative features directory", required=True)
    args = vars(parser.parse_args())

    pos_feat_path =  args["posfeat"]
    neg_feat_path = args["negfeat"]
    testpath = '/new_home/kevin/aj/102675/HOG-SVM-python/data/dataset/plant/Ara2013-RPi'

    clf = joblib.load(model_path)

    tp = 0
    fn = 0
    print('test pos')
    for im_scaled in glob.glob(pos_feat_path + '/*.png'):
        im = imread(im_scaled, as_gray=False)
        fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=transform_sqrt)
        # print(fd.shape)
        pred = clf.predict(fd.reshape(1, -1))
        # print(clf.predict_proba(fd.reshape(1, -1)))
        if pred == 1:
            tp += 1
        else:
            print('wrong pos predict:', im_scaled)
            fn += 1
    fp = 0
    tn = 0
    print('test neg')
    for im_scaled in glob.glob(neg_feat_path + '/*.png'):
        im = imread(im_scaled, as_gray=False)
        fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=transform_sqrt)
        # print(fd.shape)
        pred = clf.predict(fd.reshape(1, -1))
        # print(clf.predict_proba(fd.reshape(1, -1)))
        if pred == 1:
            fp += 1
            print('wrong pos predict:', im_scaled)
        else:
            tn += 1
    print('{} pos samples, {} correct, {} wrong, correct rate is {}'.format(tp + fn, tp, fn, tp*1.0/(tp + fn)))
    print('{} neg samples, {} correct, {} wrong, correct rate is {}'.format(fp + tn, tn, fp, tn*1.0/(fp + tn)))
    print('precision = {}'.format(tp*1.0/(tp + fp)))
    print('recall = {}'.format(tp*1.0/(tp +fn)))

def auc():
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--posfeat", help="Path to the positive features directory", required=True)
    parser.add_argument('-n', "--negfeat", help="Path to the negative features directory", required=True)
    args = vars(parser.parse_args())

    pos_feat_path =  args["posfeat"]
    neg_feat_path = args["negfeat"]
    testpath = '/new_home/kevin/aj/102675/HOG-SVM-python/data/dataset/plant/Ara2013-RPi'

    clf = joblib.load(model_path)

    resp = []
    resn = []
    print('test pos')
    for im_scaled in glob.glob(pos_feat_path + '/*.png'):
        im = imread(im_scaled, as_gray=False)
        fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=transform_sqrt)
        # print(fd.shape)
        resp.append(clf.predict_proba(fd.reshape(1, -1))[0][1])
        
    print('test neg')
    for im_scaled in glob.glob(neg_feat_path + '/*.png'):
        im = imread(im_scaled, as_gray=False)
        fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=transform_sqrt)
        # print(fd.shape)
        resn.append(clf.predict_proba(fd.reshape(1, -1))[0][1])

    resp = np.asarray(resp)
    resn = np.asarray(resn)
    for th in np.arange(0.5,1,0.1):
        # print(resp)
        a = resp > th
        tp = sum(a)
        fn = len(a) - sum(a)

        # print(resn)
        b = resn <= th
        tn = sum(a)
        fp = len(a) - sum(a)

        print('{} pos samples, {} correct, {} wrong, correct rate is {}'.format(tp + fn, tp, fn, tp*1.0/(tp + fn)))
        print('{} neg samples, {} correct, {} wrong, correct rate is {}'.format(fp + tn, tn, fp, tn*1.0/(fp + tn)))
        print('precision = {}'.format(tp*1.0/(tp + fp)))
        print('recall = {}'.format(tp*1.0/(tp +fn)))

if __name__ == "__main__":
    auc()