import tensorflow as tf
from tqdm import tqdm

image = tf.keras.preprocessing.image
from skimage.measure import label as l
from dataset_utils import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

LABELS = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

def get_x(datagen):
    x = []
    for index in range(datagen.__len__()):
        img, lbl = datagen.__getitem__(index)
        x.append(img)
    X = np.concatenate(x)
    return np.squeeze(X, axis=-1)


if __name__ == '__main__':
    seed = 1
    classes = 3
    batch_size = 32
    
    images_list = make_list_of_patients()

    patients_train, patients_test = test_split(data=images_list)
    X_train_folds, y_train_folds, X_val_folds, y_val_folds = stratified_cross_validation_splits(data=patients_train)
    X_test, y_test = dataframe2lists(patients_test)
    folds = 5
    x_train_fold0 = X_train_folds[0]
    y_train_fold0 = y_train_folds[0]

    x_val_fold0 = X_val_folds[0]
    y_val_fold0 = y_val_folds[0]

    dg_train0 = DataGen(batch_size, (256, 256), x_train_fold0, y_train_fold0)
    dg_val0 = DataGen(batch_size, (256, 256), x_val_fold0, y_val_fold0)

    # DATA EXPLORATION
    group_train = pd.DataFrame(list(zip(dg_train0.input_img_paths, dg_train0.target)),
                               columns=['count', 'label'])
    group_val = pd.DataFrame(list(zip(dg_val0.input_img_paths, dg_val0.target)),
                             columns=['count', 'label'])
    label_group_t = group_train[['count', 'label']].groupby(['label']).count().reset_index()
    label_group_t['dataset'] = 'training'
    label_group_v = group_val[['count', 'label']].groupby(['label']).count().reset_index()
    label_group_v['dataset'] = 'validation'
    label_group = pd.merge(left=label_group_t, right=label_group_v, how='outer')
    sns.barplot(x='label', y='count', hue='dataset', data=label_group)
    plt.title('Number of images per class')
    plt.show()
    
    image_indexes = [1, 12, 18, 19, 124]
    start_pointx = 0
    end_pointx = 25
    start_pointy = 231
    end_pointy = 256
    i = 0
    images, labels = get_images(image_indexes)

    row = 0
    for image, label in zip(images, labels):
        plt.suptitle('Data Exploration for image n.' + str(image_indexes[row]) + '\nLabel: ' + LABELS[np.argmax(label)] )
        image = image[0, :, :, 0]

        label = label[0]
        roi = image[start_pointy:end_pointy, start_pointx:end_pointx]
        image_rect = cv2.rectangle(image, (start_pointx, start_pointy), (end_pointx, end_pointy), 2)

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

    # NOISE DETECTION
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
        histBR, x = np.histogram(mask[230:256, 230:256], bins=2, range=(0, 1))
        histTL, x = np.histogram(mask[0:25, 0:25], bins=2, range=(0, 1))
        histTR, x = np.histogram(mask[0:25, 230:256], bins=2, range=(0, 1))
        histBL, x = np.histogram(mask[230:256, 0:25], bins=2, range=(0, 1))
        hist = histTL+histTR+histBL+histBR
        labels = l(mask)

        if len(np.unique(labels)) >= 100:
            count_noise += 1
        else:
            if hist[0] > hist[1]:
                count_normal += 1
            else:
                count_inverted += 1

    print('Normal images:\t', round(count_normal / len(images_list), 4)*100, '%')
    print('Inverted images:\t', round(count_inverted / len(images_list), 4)*100, '%')
    print('Noisy/Corrupted images:\t', round(count_noise / len(images_list), 4)*100, '%')

