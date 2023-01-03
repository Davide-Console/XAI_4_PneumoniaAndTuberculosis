import tensorflow as tf
from tqdm import tqdm

from skimage.measure import label as lfoo
from dataset_utils import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

image = tf.keras.preprocessing.image
LABELS = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]


def get_x(datagen):
    """
        Extract the images from a data generator and return as a single array.

        Parameters
        ----------
        datagen : generator
            A generator that yields image-label pairs.

        Returns
        -------
        X : numpy array
            A single array containing all the images from the generator. The images are squeezed along the last axis.
    """
    x = []
    for index in range(datagen.__len__()):
        img, lbl = datagen.__getitem__(index)
        x.append(img)
    X = np.concatenate(x)
    return np.squeeze(X, axis=-1)


if __name__ == '__main__':
    # plots train-val-test sets distribution and the histograms of ROIs of some sample images
    # prints noise distribution

    seed = 1
    classes = 3
    batch_size = 32

    images_list = make_list_of_patients()

    patients_train, patients_test = test_split(data=images_list)
    X_train_folds, y_train_folds, X_val_folds, y_val_folds = stratified_cross_validation_splits(data=patients_train)
    X_test, y_test = dataframe2lists(patients_test)

    x_train_fold0 = X_train_folds[0]
    y_train_fold0 = y_train_folds[0]

    x_val_fold0 = X_val_folds[0]
    y_val_fold0 = y_val_folds[0]

    dg_train0 = DataGen(batch_size, (256, 256), x_train_fold0, y_train_fold0)
    dg_val0 = DataGen(batch_size, (256, 256), x_val_fold0, y_val_fold0)
    dg_test = DataGen(batch_size, (256, 256), X_test, y_test)

    # DATA EXPLORATION - Stratification
    # plots the training-validation split for each class. The proportions in the graphs are respected for each one of
    # the folds
    group_train = pd.DataFrame(list(zip(dg_train0.input_img_paths, dg_train0.target)),
                               columns=['count', 'label'])
    group_val = pd.DataFrame(list(zip(dg_val0.input_img_paths, dg_val0.target)),
                             columns=['count', 'label'])
    group_test = pd.DataFrame(list(zip(dg_test.input_img_paths, dg_test.target)),
                              columns=['count', 'label'])
    label_group_t = group_train[['count', 'label']].groupby(['label']).count().reset_index()
    label_group_t['dataset'] = 'training'
    label_group_v = group_val[['count', 'label']].groupby(['label']).count().reset_index()
    label_group_v['dataset'] = 'validation'
    label_group = pd.merge(left=label_group_t, right=label_group_v, how='outer')

    label_group_test = group_test[['count', 'label']].groupby(['label']).count().reset_index()
    label_group_test['dataset'] = 'test'
    label_group = pd.merge(left=label_group, right=label_group_test, how='outer')

    ax = sns.barplot(x='label', y='count', hue='dataset', data=label_group)
    ax.set_xticklabels(LABELS)
    plt.title('Number of images per class')
    plt.show()

    # DATA EXPLORATION - ROI Analysis
    # selects significant images (representative of different kind of conditions) and plots the image, the ROI selected
    # and the histogram f the ROI
    image_indexes = [1, 12, 18, 19, 124]
    start_pointx = 0
    end_pointx = 25
    start_pointy = 231
    end_pointy = 256
    i = 0
    images, labels = get_images(image_indexes)

    row = 0
    for image, label in zip(images, labels):
        plt.suptitle('Data Exploration for image n.' + str(image_indexes[row]))
        image = image[0, :, :, 0]

        label = label[0]
        roi = image[start_pointy:end_pointy, start_pointx:end_pointx]
        image_rect = cv2.rectangle(gray2rgb(image) / 255, (start_pointx, start_pointy), (end_pointx, end_pointy),
                                   (255, 0, 0), 2)

        plt.subplot(2, 2, 1)
        plt.imshow(image_rect, cmap='gray')
        plt.axis('off')
        plt.title('Image')

        plt.subplot(2, 2, 3)
        plt.imshow(roi / 255, cmap='gray')
        titleroi = "Selected ROI"
        plt.axis('off')
        plt.title(titleroi)

        plt.subplot(1, 2, 2)
        plt.hist(roi.ravel(), 256, [0, 256])
        plt.ylim([0, 50])
        title = "ROI histogram"
        plt.title(title)

        row += 1
        plt.show()

    # DATA EXPLORATION - Noise detection
    # scans all images and classifies them in normal contrast images, complementary contrast images, and noisy/corrupted
    # images
    data = pd.read_csv('dataset/labels_train.csv', sep=',')

    images_list = []

    for i in range(len(data)):
        images_list.append(data.iloc[i, 0])

    count_inverted = 0
    count_normal = 0
    count_noise = 0
    for i in tqdm(range(len(images_list))):
        img = cv2.imread('dataset/' + images_list[i], 0)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

        val = filters.threshold_otsu(img)
        mask = (img > val) * 1.0
        histBR, x1 = np.histogram(mask[230:256, 230:256], bins=2, range=(0, 1))
        histTL, x2 = np.histogram(mask[0:25, 0:25], bins=2, range=(0, 1))
        histTR, x3 = np.histogram(mask[0:25, 230:256], bins=2, range=(0, 1))
        histBL, x4 = np.histogram(mask[230:256, 0:25], bins=2, range=(0, 1))
        hist = histTL + histTR + histBL + histBR
        labels = lfoo(mask)

        if len(np.unique(labels)) >= 100:
            count_noise += 1
        else:
            if hist[0] > hist[1]:
                count_normal += 1
            else:
                count_inverted += 1

    print('Normal contrast images:\t', round(count_normal / len(images_list), 4) * 100, '%')
    print('Complementary contrast images:\t', round(count_inverted / len(images_list), 4) * 100, '%')
    print('Noisy/Corrupted images:\t', round(count_noise / len(images_list), 4) * 100, '%')

    noise_labels = ['Normal Contrast', 'Complementary Contrast', 'Noisy/Corrupted']
    data = [round(count_normal / len(images_list), 4), round(count_inverted / len(images_list), 4), round(count_noise / len(images_list), 4)]
    plt.pie(data, labels=noise_labels)
    plt.title('Distribution of normal, inverted contrast, and noisy or corrupted images')
    plt.show()
