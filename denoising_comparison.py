from dataset_utils import *

LABELS = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

if __name__ == '__main__':
    # Plots a visual comparison of noisy image with the processed images using different methods:
    # median+mean filtering
    # PCA
    # Denoising Autoencoder

    seed = 1
    classes = 3
    batch_size = 1
    batch_size_pca = 32
    image_indexes = [7, 19, 21]
    img_path = ["dataset/P11642_1.jpeg", "dataset/P04618_1.jpeg", "dataset/P08347_1.jpeg"]
    patients = make_list_of_patients()

    patients_train, patients_test = test_split(data=patients)
    X_train_folds, y_train_folds, X_val_folds, y_val_folds = stratified_cross_validation_splits(data=patients_train)
    X_test, y_test = dataframe2lists(patients_test)

    fold = 0
    x_train_fold0 = X_train_folds[fold]
    y_train_fold0 = y_train_folds[fold]

    x_val_fold0 = X_val_folds[fold]
    y_val_fold0 = y_val_folds[fold]

    dg_train_autoenc = DataGen(batch_size, (256, 256), x_train_fold0, y_train_fold0,
                               autoencoder='AE_model/0.0103f_model.h5')
    dg_train_pca = DataGen(batch_size_pca, (256, 256), x_train_fold0, y_train_fold0, pca_denoising=True)
    dg_train_filt=DataGen(batch_size, (256, 256), x_train_fold0, y_train_fold0, filtering=True)

    images_origin = []
    images_filtered = []
    images_autoenc = []
    images_pca = []
    lbls = []
    k = 0
    images_batch_pca, labels = dg_train_pca.__getitem__(0)
    for index in image_indexes:
        image_raw = cv2.imread(img_path[k], 0)
        image_raw= cv2.resize(image_raw, (256, 256), interpolation=cv2.INTER_CUBIC)
        k = k + 1
        # Filter
        img_filtered, lab= dg_train_filt.__getitem__(index)

        # Autoencoder
        img_auto, lbl1 = dg_train_autoenc.__getitem__(index)

        # PCA
        img_pca = images_batch_pca[index]

        lbls.append(lbl1)
        images_origin.append(image_raw)
        images_filtered.append(img_filtered)
        images_autoenc.append(img_auto)
        images_pca.append(img_pca)

    i = 0
    for img_origin, img_filt, img_autoenc, img_denoise_pca, label in zip(images_origin, images_filtered, images_autoenc,
                                                                         images_pca, lbls):

        image_filt = img_filt[0, :, :, 0]
        image_autoencoder = img_autoenc[0, :, :, 0]
        image_pca = img_denoise_pca[:, :, 0]

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),constrained_layout=True)
        fig.suptitle("Image: " + str(image_indexes[i]) + "\n Label: " + LABELS[np.argmax(label)])
        axs[0,0].imshow(img_origin, cmap="gray")
        axs[0,0].set_title("Original")

        axs[0,1].imshow(image_filt, cmap="gray")
        axs[0,1].set_title("Median+Mean filter")

        axs[1,0].imshow(image_autoencoder, cmap="gray")
        axs[1,0].set_title("Autoencoder")

        axs[1,1].imshow(image_pca, cmap="gray")
        axs[1,1].set_title("PCA reconstruction")

        name_fig="denoise_compare_img"+str(image_indexes[i])+".png"
        plt.savefig(name_fig)
        plt.show()
        i = i + 1
