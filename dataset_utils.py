import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
import cv2
from skimage.color import gray2rgb
from skimage.measure import label as label_fn
from scipy import ndimage
import tensorflow as tf


NORMAL = 0
PNEUMONIA = 1
TUBERCULOSIS = 2


def invert_image(img):
    """
        Invert the image if the number of unique labels in the image is less than 100 and the number of white pixels is
        greater than the number of black pixels in the ROI.

        The function first calculates a threshold value using Otsu's method and creates a binary mask of the image. It then
        counts the number of white pixels in the mask in the top left, top right, bottom left, and bottom right quadrants.
        The function also calculates the number of unique labels in the image using the 'label_fn()' function. If the
        number of unique labels is less than 100 and the number of white pixels is greater than the number of black pixels,
        the function returns the inverted image, otherwise it returns the original image.

        Parameters
        ----------
        img : numpy array
            A 2D image array.

        Returns
        -------
        img : numpy array
            The original image if the number of unique labels is greater than or equal to 100 or the number of black pixels
            is greater than the number of white pixels, or the inverted image if the number of unique labels is less than
            100 and the number of white pixels is greater than the number of black pixels.
        """
    val = filters.threshold_otsu(img)
    mask = (img > val) * 1.0
    histBR, x = np.histogram(mask[230:256, 230:256], bins=2, range=(0, 1))
    histTL, x = np.histogram(mask[0:25, 0:25], bins=2, range=(0, 1))
    histTR, x = np.histogram(mask[0:25, 230:256], bins=2, range=(0, 1))
    histBL, x = np.histogram(mask[230:256, 0:25], bins=2, range=(0, 1))
    hist = histTL + histTR + histBL + histBR
    labels = label_fn(mask)

    if len(np.unique(labels)) < 100:
        if hist[0] > hist[1]:
            return img
        else:
            return 255 - img
    return img


class DataGen(keras.utils.Sequence):
    """
     A data generator that yields batches of images and labels.

     The generator reads the images and labels from the file paths and performs optional filtering, data augmentation,
     and denoising using an autoencoder or PCA. If the 'imagenet' weights are used, the images are also converted to
     RGB.

     Parameters
     ----------
     batch_size : int
         The number of images in each batch.
     img_size : tuple
         The size to which the images should be resized.
     input_paths : list
         A list of file paths to the input images.
     target : list
         A list of labels corresponding to the input images.
     weights : str, optional
         The weights to use for the images. If 'imagenet', the images are converted to RGB.
     filtering : bool, optional
        Whether to apply median and uniform filtering to the images.
    data_aug : bool, optional
        Whether to apply random rotation and flipping to the images.
    autoencoder : str, optional
        The file path to an autoencoder model to use for denoising the images.
    invert_black_bg : bool, optional
        Whether to invert the images if they have a black background and less than 100 unique labels.
    pca_denoising : bool, optional
        Whether to denoise the images using PCA.
    """
    def __init__(self, batch_size, img_size, input_paths, target, weights=None, filtering=False, data_aug=False,
                 autoencoder=None, invert_black_bg=False, pca_denoising=False):
        self.batch_size = batch_size
        self.img_size = img_size  # (400, 400)
        self.input_img_paths = input_paths
        self.target = target
        self.imagenet = weights == "imagenet"
        self.channels = 3 if self.imagenet else 1
        self.directory = 'dataset/'
        self.filtering = filtering
        self.data_aug = data_aug
        self.autoencoder = autoencoder
        self.invert_black_bg = invert_black_bg
        self.pca_denoising = pca_denoising
        if self.autoencoder is not None:
            self.model = tf.keras.models.load_model(autoencoder)

    def __getitem__(self, index):
        i = index * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target = self.target[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (self.channels,))
        y = batch_target

        for j, path in enumerate(batch_input_img_paths):
            img = cv2.imread(self.directory + path, 0)  # read as grayscale
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)

            if self.invert_black_bg:
                img = invert_image(img)

            if self.filtering:
                img = cv2.medianBlur(img, ksize=5)
                img = ndimage.uniform_filter(img, size=3)

            if self.data_aug:
                angle = random.randint(0, 359)
                img = ndimage.rotate(img, angle=angle, reshape=False)
                if random.randrange(0, 100) > 50:
                    if random.randrange(0, 100) > 50:
                        img = np.fliplr(img)
                    if random.randrange(0, 100) > 50:
                        img = np.flipud(img)

            if self.autoencoder is not None:
                img = np.expand_dims(img, 2)
                img = np.expand_dims(img, 0)
                img = img.astype('float32') / 255
                img = self.model.predict(img)
                img = (img * 255).astype('uint8')
                img = img[0, :, :, 0]

            if self.imagenet:
                img = gray2rgb(img)
                x[j] = img
            else:
                x[j] = np.expand_dims(img, 2)

        if self.pca_denoising:
            batch = x[:, :, :, 0]
            x_for_pca = batch.reshape(len(batch), self.img_size[0] * self.img_size[1])
            n_comp = 12
            pca = PCA(n_components=n_comp)
            pca.fit(x_for_pca)
            x_reconstructed_pca = pca.inverse_transform(pca.transform(x_for_pca))
            x_reconstructed_pca = x_reconstructed_pca.reshape(batch.shape)  # 32 256 256
            for i in range(len(x)):
                if self.imagenet:
                    x[i] = gray2rgb(x_reconstructed_pca[i])
                else:
                    x[i] = np.expand_dims(x_reconstructed_pca[i], 2)

        return x, keras.utils.to_categorical(y, num_classes=3)

    def __len__(self):
        return len(self.target) // self.batch_size


def noise(array, type='gaussian'):
    """
    Add noise to an array.

    Parameters
    ----------
    array : numpy array
        The array to which noise should be added.
    type : str, optional
        The type of noise to add. Supported types are 'gaussian' and 'uniform'.

    Returns
    -------
    noisy_array : numpy array
        The array with added noise.
    """
    if type == 'gaussian':
        np.random.seed(1)
        noise_factor = 0.2
        noisy_array = array + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=array.shape)
    elif type == 'uniform':
        uni_noise = np.zeros(array.shape, dtype=np.float32)
        cv2.randu(uni_noise, 0, 1)
        uni_noise = (uni_noise * 0.4).astype(np.float32)
        noisy_array = array + uni_noise
    return np.clip(noisy_array, 0.0, 1.0)


class DG_autoencoder(keras.utils.Sequence):
    """
    A data generator that yields batches of noisy and clean images for training an autoencoder.

    The generator reads the images from the file paths and adds uniform noise to them.

    Parameters
    ----------
    batch_size : int
        The number of images in each batch.
    img_size : tuple
        The size to which the images should be resized.
    input_paths : list
        A list of file paths to the input images.
    target : list
        A list of labels corresponding to the input images.
    """
    def __init__(self, batch_size, img_size, input_paths, target):
        self.batch_size = batch_size
        self.img_size = img_size  # (400, 400)
        self.input_img_paths = input_paths
        self.target = target
        self.directory = 'dataset/'
        self.channels = 1

    def __getitem__(self, index):
        i = index * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]

        x_noisy = np.zeros((self.batch_size,) + self.img_size + (self.channels,))
        x = np.zeros((self.batch_size,) + self.img_size + (self.channels,))

        for j, path in enumerate(batch_input_img_paths):
            img = cv2.imread(self.directory + path, 0)  # read as grayscale
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
            img = img.astype('float32') / 255
            x_noisy[j] = np.expand_dims(noise(img, 'uniform'), 2)
            x[j] = np.expand_dims(img, 2)

        return x_noisy, x

    def __len__(self):
        return len(self.target) // self.batch_size


def make_list_of_patients():
    """
    Read the labels from the 'labels_train.csv' file and create a data frame of patients with lists of file paths and
    labels.

    The function reads the 'labels_train.csv' file and creates a data frame with columns 'ID', 'paths', and 'label'.
    Each row in the data frame represents a patient and contains the patient's ID, a list of file paths to the
    patient's images, and the patient's label. If two samples have the same ID, the function checks that they have the
    same label.

    Returns
    -------
    patients : pandas DataFrame
        A data frame of patients with lists of file paths and labels.
    """
    data = pd.read_csv('dataset/labels_train.csv', sep=',')

    patients = pd.DataFrame(columns=['ID', 'paths', 'label'])

    for _, row in data.iterrows():
        patient_ID = row['file'].split('_')[0]
        if len(patients.loc[patients['ID'] == patient_ID, 'paths']) == 0:
            new_patient = {'ID': patient_ID, 'paths': row['file'], 'label': row['label']}
            patients = patients.append(new_patient, ignore_index=True)
        else:
            old_label = patients.loc[patients['ID'] == patient_ID, 'label']
            assert old_label.values[0] == row['label']  # make sure two samples with the same ID do not have
            # different diagnosis
            paths = patients.loc[patients['ID'] == patient_ID, 'paths'].to_list()
            paths.append(row['file'])
            updated_patient = {'ID': patient_ID, 'paths': paths, 'label': row['label']}
            patients.drop(patients[patients.ID == patient_ID].index, inplace=True)
            patients = patients.append(updated_patient, ignore_index=True)

    patients = patients.replace('N', NORMAL)
    patients = patients.replace('P', PNEUMONIA)
    patients = patients.replace('T', TUBERCULOSIS)
    return patients


def get_images(indexes, filtered=False, input_channels=1, invert_black_bg=True):
    """
    Get a list of images and labels from a data generator.

    The function creates a data generator using the 'DataGen' class and returns a list of images and labels for the
    given indexes. The data generator uses the test split of the data frame created by the 'make_list_of_patients'
    function.

    Parameters
    ----------
    indexes : list
        A list of indexes for the images to get from the data generator.
    filtered : bool, optional
        Whether to apply median and uniform filtering to the images.
    input_channels : int, optional
        The number of channels in the input images. Supported values are 1 and 3.
    invert_black_bg : bool, optional
        Whether to invert the images if they have a black background and less than 100 unique labels.

    Returns
    -------
    imgs : list
        A list of images.
    lbls : list
        A list of labels corresponding to the images.
    """
    patients = make_list_of_patients()
    patients_train, patients_test = test_split(data=patients)
    X_test, y_test = dataframe2lists(patients_test)

    batch_size = 1
    if input_channels == 1:
        weights = None
    elif input_channels == 3:
        weights = "imagenet"
    else:
        raise ValueError

    dg_val0 = DataGen(batch_size, (256, 256), X_test, y_test, weights=weights, filtering=filtered, invert_black_bg=invert_black_bg)

    imgs = []
    lbls = []
    for index in indexes:
        img, lbl = dg_val0.__getitem__(index)
        imgs.append(img)
        lbls.append(lbl)

    return imgs, lbls


def test_split(data: pd.DataFrame, test_size=0.15, shuffle=True, random_state=8432):
    """
    Split the data into training and test sets.

    The function uses the 'train_test_split' function from scikit-learn to split the data into training and test sets.
    The function also shuffles the data and stratifies it by the labels.

    Parameters
    ----------
    data : pandas DataFrame
        The data to split.
    test_size : float, optional
        The proportion of the data to include in the test set.
    shuffle : bool, optional
        Whether to shuffle the data before splitting it.
    random_state : int, optional
        The seed used by the random number generator.

    Returns
    -------
    patients_train : pandas DataFrame
        The training set.
    patients_test : pandas DataFrame
        The test set.
    """
    patients_train, patients_test = train_test_split(data,
                                                     stratify=data.iloc[:, -1].values,
                                                     test_size=test_size,
                                                     shuffle=shuffle,
                                                     random_state=random_state)
    return patients_train, patients_test


def dataframe2lists(data: pd.DataFrame):
    """
    Convert data from a pandas DataFrame to lists.

    The function converts the 'ID', 'paths', and 'label' columns of the input DataFrame into separate lists.

    Parameters
    ----------
    data : pandas DataFrame
        The DataFrame to convert.

    Returns
    -------
    X : list
        A list of 'paths' values.
    y : list
        A list of 'label' values.
    """
    X = []
    y = []

    for i in range(len(data)):
        if len(data.iloc[i, 1]) > 1 and type(data.iloc[i, 1]) == list:
            for element in data.iloc[i, 1]:
                X.append(element)
                y.append(data.iloc[i, 2])
        else:
            X.append(data.iloc[i, 1])
            y.append(data.iloc[i, 2])

    return X, y


def stratified_cross_validation_splits(data: pd.DataFrame, fold=5, shuffle=True, random_state=8432):
    """
    Return stratified cross-validation folds for the input data.

    Parameters:
    data (pandas.DataFrame): Input data.
    fold (int): Number of folds.
    shuffle (bool): Whether to shuffle the data before dividing into folds.
    random_state (int): Seed for shuffling.

    Returns:
    X_train_folds (list): List of training data split into folds.
    y_train_folds (list): List of training labels split into folds.
    X_test_folds (list): List of test data split into folds.
    y_test_folds (list): List of test labels split into folds.
    """
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    skf = StratifiedKFold(n_splits=fold, shuffle=shuffle, random_state=random_state)

    X_train_folds = []
    y_train_folds = []

    X_test_folds = []
    y_test_folds = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for idx_train in train_index:
            if len(data.iloc[idx_train, 1]) > 1 and type(data.iloc[idx_train, 1]) == list:
                for element in data.iloc[idx_train, 1]:
                    X_train.append(element)
                    y_train.append(data.iloc[idx_train, 2])
            else:
                X_train.append(data.iloc[idx_train, 1])
                y_train.append(data.iloc[idx_train, 2])

        X_train_folds.append(X_train)
        y_train_folds.append(y_train)

        for idx_test in test_index:
            if len(data.iloc[idx_test, 1]) > 1 and type(data.iloc[idx_test, 1]) == list:
                for element in data.iloc[idx_test, 1]:
                    X_test.append(element)
                    y_test.append(data.iloc[idx_test, 2])
            else:
                X_test.append(data.iloc[idx_test, 1])
                y_test.append(data.iloc[idx_test, 2])

        X_test_folds.append(X_test)
        y_test_folds.append(y_test)

    return X_train_folds, y_train_folds, X_test_folds, y_test_folds

