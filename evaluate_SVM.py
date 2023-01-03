import pickle

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import hog
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset_utils import make_list_of_patients, test_split, dataframe2lists, invert_image

model = pickle.load(open('float_model/fold1-svm.pkl', 'rb'))

patients = make_list_of_patients()
patients_train, patients_test = test_split(data=patients)
X_test, y_test = dataframe2lists(patients_test)

img_size = (256, 256)

ex_img = cv2.imread('dataset/' + X_test[0], 0)
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

X = np.zeros(shape=(len(X_test),) + fd.shape)

for i in tqdm(range(len(X_test))):
    img = cv2.imread('dataset/' + X_test[i], 0)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)

    if invert_black_bg:
        img = invert_image(img)

    if filtering:
        img = cv2.medianBlur(img, ksize=5)
        img = ndimage.uniform_filter(img, size=3)

    fd = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
             cells_per_block=cells_per_block, visualize=False, channel_axis=None)
    X[i] = fd

y_pred = model.predict(X)

print(classification_report(y_test, y_pred, digits=4,
                            output_dict=False))
