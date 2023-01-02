import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import cv2


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


if __name__ == '__main__':
    pca_denoising("dataset/P00001_1.png")
    pca_denoising("dataset/P00024_1.jpeg")

