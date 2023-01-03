import pickle

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import hog
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from tqdm import tqdm

from dataset_utils import make_list_of_patients, test_split, stratified_cross_validation_splits, dataframe2lists, \
    TUBERCULOSIS, invert_image

# Trains a SVM with 5-fold cross validation. Train, validation, and test sets are the same used for the DL models

patients = make_list_of_patients()

patients_train, patients_test = test_split(data=patients)
folds = 5
X_train_folds, y_train_folds, X_val_folds, y_val_folds = stratified_cross_validation_splits(data=patients_train,
                                                                                            fold=folds)
X_test, y_test = dataframe2lists(patients_test)

img_size = (256, 256)

ex_img = cv2.imread('dataset/' + X_train_folds[0][0], 0)
ex_img = cv2.resize(ex_img, img_size, interpolation=cv2.INTER_CUBIC)

orientations = 18
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
filtering = True
invert_black_bg = True

print("Hyperparams:")
print(img_size)
print(orientations)
print(pixels_per_cell)
print(cells_per_block)
print(filtering)

fd = hog(ex_img, orientations=orientations, pixels_per_cell=pixels_per_cell,
         cells_per_block=cells_per_block, visualize=False, channel_axis=None)

print(fd.shape)

for fold in range(folds):
    x_train_fold = X_train_folds[fold]
    y_train_fold = y_train_folds[fold]

    x_val_fold = X_val_folds[fold]
    y_val_fold = y_val_folds[fold]

    X = np.zeros(shape=(len(x_train_fold),) + fd.shape)

    for i in tqdm(range(len(x_train_fold))):
        img = cv2.imread('dataset/' + x_train_fold[i], 0)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)

        if invert_black_bg:
            img = invert_image(img)

        if filtering:
            img = cv2.medianBlur(img, ksize=5)
            img = ndimage.uniform_filter(img, size=3)

        fd = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block, visualize=False, channel_axis=None)
        X[i] = fd

    classes = len(np.unique(y_train_fold))
    cl_w = {}
    for c in range(classes):
        cl_w.update({c: 1})

    cl_w.update({TUBERCULOSIS: 19})
    clf = SVC(class_weight=cl_w, probability=True, C=0.7)

    y = np.asarray(y_train_fold)
    clf.fit(X, y)

    y_pred = clf.predict(X)

    print(f'Fold {fold} train set results')
    print(classification_report(y, y_pred, digits=4,
                                output_dict=False))

    X = np.zeros(shape=(len(x_val_fold),) + fd.shape)

    for i in tqdm(range(len(x_val_fold))):
        img = cv2.imread('dataset/' + x_val_fold[i], 0)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)

        if invert_black_bg:
            img = invert_image(img)

        if filtering:
            img = cv2.medianBlur(img, ksize=5)
            img = ndimage.uniform_filter(img, size=3)

        fd = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block, visualize=False, channel_axis=None)
        X[i] = fd

    y = np.asarray(y_val_fold)
    y_pred = clf.predict(X)

    print(f'Fold {fold} validation set results')
    print(classification_report(y, y_pred, digits=4,
                                output_dict=False))

    pickle.dump(clf, open(f'explainedModels/fold{fold}-svm.pkl', 'wb'))
