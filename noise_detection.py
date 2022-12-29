from tqdm import tqdm
from math import log10, sqrt
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
image=tf.keras.preprocessing.image
from dataset_utils import *
import cv2
from skimage import filters
from skimage.measure import label

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def get_body_mask(img):
    val = filters.threshold_otsu(img)

    mask = (img > val) * 1.0

    mask = getLargestCC(mask)

#    inv_mask = np.zeros((len(mask), len(mask)), dtype=int)
 #   for x in range(len(inv_mask)):
  #      for y in range(len(inv_mask)):
   #         if mask[x, y] == 0:
    #            inv_mask[x, y] = 1

#    inv_mask[0:255, :] = getLargestCC(inv_mask[0:255, :])
 #   inv_mask[255:511, :] = getLargestCC(inv_mask[255:511, :])

#    for x in range(len(inv_mask)):
 #       for y in range(len(inv_mask)):
  #          if inv_mask[x, y] == 0:
   #             mask[x, y] = 1
    #        elif inv_mask[x, y] == 1:
     #           mask[x, y] = 0
    return mask

if __name__ == '__main__':
    seed = 1
    classes = 3
    batch_size = 1
    data = pd.read_csv('dataset/labels_train.csv', sep=',')

    patients = []

    for i in range(len(data)):
        patients.append(data.iloc[i, 0])
    count_inverted=0
    count_normal=0
    count_sp=0
    count_gaussian=0
    for i in range(len(patients)):
        img = cv2.imread('dataset/'+patients[i], 0)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

        val = filters.threshold_otsu(img)
        mask = (img > val) * 1.0

        histBR, x = np.histogram(mask[230:256, 230:256], bins=2, range=(0, 1))
        histTL, x = np.histogram(mask[0:25, 0:25], bins=2, range=(0, 1))
        histTR, x = np.histogram(mask[0:25, 230:256], bins=2, range=(0, 1))
        histBL, x = np.histogram(mask[230:256, 0:25], bins=2, range=(0, 1))
        hist = histTL+histTR+histBL+histBR
        labels = label(mask)

        if len(np.unique(labels)) >= 100:
            print(patients[i], '\tGAUSSIAN (MASK)')
            count_gaussian += 1
        else:
            if hist[0] > hist[1]:
                print(patients[i], '\tNORMAL (MASK)')
                count_normal += 1
            else:
                print(patients[i], '\tINVERTED (MASK)')
                count_inverted += 1
