import tensorflow as tf
image=tf.keras.preprocessing.image
from dataset_utils import *
import cv2
from skimage import filters
from skimage.measure import label


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
    count_uniform=0
    for i in range(len(patients)):
        img = cv2.imread('dataset/'+patients[i], 0)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

        val = filters.threshold_otsu(img)
        mask = (img > val) * 1.0
        total_hist, x = np.histogram(img, bins=256, range=(0, 255))
        histBR, x = np.histogram(mask[230:256, 230:256], bins=2, range=(0, 1))
        histTL, x = np.histogram(mask[0:25, 0:25], bins=2, range=(0, 1))
        histTR, x = np.histogram(mask[0:25, 230:256], bins=2, range=(0, 1))
        histBL, x = np.histogram(mask[230:256, 0:25], bins=2, range=(0, 1))
        hist = histTL+histTR+histBL+histBR
        labels = label(mask)

        if len(np.unique(labels)) >= 100:
            uniform=True
            for j in range(254):
                if total_hist[j + 1] >= sum(total_hist[1:255]) / 254 * 1.5 or \
                        total_hist[j + 1] <= sum(total_hist[1:255]) / 254 * 0.5:
                    uniform = False
            if uniform is True:
                print(patients[i], '\tUNIFORM NOISE')
                count_uniform += 1
            elif uniform is False:
                print(patients[i], '\tS&P NOISE')
                count_sp += 1
        else:
            if hist[0] > hist[1]:
                print(patients[i], '\tNORMAL')
                count_normal += 1
            else:
                print(patients[i], '\tINVERTED')
                count_inverted += 1

    print('Normal images:\t', round(count_normal/len(patients), 2), '%')
    print('Inverted images:\t', round(count_inverted/len(patients), 2), '%')
    print('Uniform images:\t', round(count_uniform/len(patients), 2), '%')
    print('S&P images:\t', round(count_sp/len(patients), 2), '%')
