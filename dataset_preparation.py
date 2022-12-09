import shutil
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
from PIL import Image


def dataset_2_np(dataset_dir='dataset'):
    """
        This function takes the directory of the dataset as input and returns the X and Y numpy arrays.
        The X array contains the images and the Y array contains the labels.
        The images are resized to 96x96.
        The labels are one-hot encoded.
    """
    IMG_SIZE = 96

    dataset = ImageDataGenerator()

    dataset = dataset.flow_from_directory(directory=dataset_dir,
                                          target_size=(IMG_SIZE, IMG_SIZE),
                                          color_mode='rgb',
                                          classes=None,  # can be set to labels
                                          class_mode='categorical',
                                          batch_size=1,
                                          seed=1)

    X = []
    Y = []
    for i in range(dataset.__len__()):
        x, y = dataset.__getitem__(i)
        X.append(x)
        Y.append(y)

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    return X, Y


def extract_dataset(zipped_dataset, output_directory):
    """
        Extracts the dataset from the zipped file and moves the folders to the root directory.
        If data_aug is True, the function will augment the dataset to have 500 images per class.
        :param zipped_dataset: The path to the zipped dataset.
        :param output_directory: The path to the output directory.
        :param data_aug: Whether to augment the dataset or not.
        :return: None
    """
    os.makedirs(output_directory, exist_ok=True)
    with zipfile.ZipFile(zipped_dataset, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

    path_to_data = output_directory + 'train'
    for folder in os.listdir(path_to_data):
        shutil.rmtree(os.path.join(output_directory, folder), ignore_errors=True)
        shutil.move(os.path.join(path_to_data, folder), output_directory)
    os.rmdir(path_to_data)


if __name__ == '__main__':
    extract_dataset('train_set.zip', 'dataset/')
