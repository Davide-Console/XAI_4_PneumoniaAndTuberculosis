import cv2
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.svm import SVC
from skimage.feature import hog

from dataset_utils import make_list_of_patients, test_split, dataframe2lists, TUBERCULOSIS

patients = make_list_of_patients()
patients_train, patients_test = test_split(data=patients)
X_train, y_train = dataframe2lists(patients_train)
X_test, y_test = dataframe2lists(patients_test)

img_size = (400, 400)

ex_img = cv2.imread('dataset/' + X_train[0], 0)
ex_img = cv2.resize(ex_img, img_size, interpolation=cv2.INTER_CUBIC)

fd = hog(ex_img, orientations=9, pixels_per_cell=(16, 16),
         cells_per_block=(2, 2), visualize=False, channel_axis=None)

X = np.zeros(shape=(len(X_train),) + fd.shape)

for i in tqdm(range(len(X_train))):
    img = cv2.imread('dataset/' + X_train[i], 0)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    fd = hog(img, orientations=9, pixels_per_cell=(16, 16),
             cells_per_block=(2, 2), visualize=False, channel_axis=None)
    X[i] = fd

classes = len(np.unique(y_train))
cl_w = {}
for c in range(classes):
    cl_w.update({c: 1})

cl_w.update({TUBERCULOSIS: 39})
clf = SVC(class_weight=cl_w)

y = np.asarray(y_train)
clf.fit(X, y)

X = np.zeros(shape=(len(X_test),) + fd.shape)

for i in tqdm(range(len(X_test))):
    img = cv2.imread('dataset/' + X_test[i], 0)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    fd = hog(img, orientations=9, pixels_per_cell=(16, 16),
             cells_per_block=(2, 2), visualize=False, channel_axis=None)
    X[i] = fd

y_pred = clf.predict(X)

print(classification_report(y_test, y_pred, digits=4,
                            output_dict=False))
