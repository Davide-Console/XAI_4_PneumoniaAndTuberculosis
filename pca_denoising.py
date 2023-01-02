import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import cv2
from dataset_utils import *


def pca_denoising(path):
    image_raw = cv2.imread(path, 0)
    print(image_raw.shape)

    image_bw = image_raw

    pca = PCA()
    pca.fit(image_bw)

    # Getting the cumulative variance
    var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100

    # How many PCs explain 95% of the variance?
    k = np.argmax(var_cumu > 95)
    print("Number of components explaining 95% variance: " + str(k))

    def plot_at_k(k):
        ipca = IncrementalPCA(n_components=k)
        image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
        plt.imshow(image_recon, cmap=plt.cm.gray)

    ks = [10, 25, 50, 100, 150, 250]

    plt.figure(figsize=[15, 9])

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plot_at_k(ks[i])
        plt.title("Components: " + str(ks[i]))

    plt.subplots_adjust(wspace=0.2, hspace=0.0)
    plt.show()


def get_x(datagen):
    x = []
    y=[]
    for index in range(datagen.__len__()):
        img, lbl = datagen.__getitem__(index)
        x.append(img)
        y.append(lbl)
    # X = np.concatenate(x)
    return np.squeeze(x, axis=-1), y


def batch_pca(x, n_batch):
    batch = x[n_batch]
    X = batch.reshape(len(batch), 256 * 256)
    n_comp = 12
    pca = PCA(n_components=n_comp)
    pca.fit(X)

    x_reconstructed_pca = pca.inverse_transform(pca.transform(X))
    x_reconstructed_pca = x_reconstructed_pca.reshape(batch.shape)

    if n_batch == 0:
        i = 7
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        axs[0].imshow(batch[i], cmap="gray")
        axs[0].set_title("Original image " + str(i))
        axs[1].imshow(x_reconstructed_pca[i], cmap="gray")
        axs[1].set_title("Reconstruction with " + str(n_comp) + " components")
        plt.show()
    return x_reconstructed_pca

def reconstructed_pca():
    seed = 1
    classes = 3
    batch_size = 32

    patients = make_list_of_patients()

    patients_train, patients_test = test_split(data=patients)
    X_train_folds, y_train_folds, X_val_folds, y_val_folds = stratified_cross_validation_splits(data=patients_train)

    x_train_fold0 = X_train_folds[0]
    y_train_fold0 = y_train_folds[0]

    dg_train0 = DataGen(batch_size, (256, 256), x_train_fold0, y_train_fold0)

    x_train0, y_train0 = get_x(dg_train0)

    x_rec = []
    for k in range(x_train0.shape[0]):
        x_rec.append(batch_pca(x_train0, k))

    return np.array(x_rec)

if __name__ == '__main__':
    # pca_denoising("dataset/P00001_1.png")
    # pca_denoising("dataset/P00024_1.jpeg")
    x_reconstructed = reconstructed_pca()
