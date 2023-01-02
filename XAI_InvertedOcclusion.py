import math
import os
import pickle

import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import execution_settings
from dataset_utils import get_images
import tensorflow as tf
import numpy as np

execution_settings.set_gpu()

LABELS = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]


def sklearn_predictions(predictor, occluded_imgs):
    ex_img = occluded_imgs[0]
    ex_img = ex_img[:, :, 0]
    orientations = 18
    pixels_per_cell = (16, 16)
    cells_per_block = (2, 2)

    fd = hog(ex_img, orientations=orientations, pixels_per_cell=pixels_per_cell,
             cells_per_block=cells_per_block, visualize=False, channel_axis=None)

    X = np.zeros(shape=(len(occluded_imgs),) + fd.shape)
    for i in range(len(occluded_imgs)):
        img = occluded_imgs[i, :, :, 0]
        fd = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block, visualize=False, channel_axis=None)
        X[i] = fd

    y_pred = predictor.predict_proba(X)

    return y_pred


def get_occluded_probabilities(img, predictor, index, patch_size=16, stride=1, sklearn_model=False):
    """Iteratively set a square patch of an image to zero.

    Parameters:
        img (np.array): The image to occlude.
        predictor (keras.model): Keras model to predict occluded images
        index (int): The label to be analyzed
        patch_size (int, optional): The size of the patch to set to zero. Default is 16.
        stride (int, optional): The stride between patches. Default is 1
        sklearn_model (bool, optional): whether the predictor is a keras model or a sklearn one. Default is keras
    Returns:
        occluded_imgs (list[np.array]): list of occluded images.
    """
    dims = img.shape
    img_height = dims[0]
    img_width = dims[1]

    occluded_imgs = []
    probs = np.zeros(shape=(len(range(0, img_height - patch_size + 1, stride)),
                            len(range(0, img_width - patch_size + 1, stride))))

    line = 0
    for y in range(0, img_height - patch_size + 1, stride):
        for x in range(0, img_width - patch_size + 1, stride):
            occluded_image = np.copy(img)
            # Set all the image except the patch to zero
            tmp = np.zeros(occluded_image.shape)
            tmp[y:y + patch_size, x:x + patch_size, :] = 1
            occluded_image = occluded_image * tmp
            occluded_imgs.append(occluded_image)

        occluded_imgs = np.asarray(occluded_imgs)
        if not sklearn_model:
            predictions = predictor.predict(occluded_imgs)
        else:
            predictions = sklearn_predictions(predictor, occluded_imgs)
        predictions = predictions[:, index]
        probs[line] = predictions
        line += 1
        occluded_imgs = []

    return probs


if __name__ == '__main__':
    image_indexes = [31, 30, 99]  # N, P, T
    model_path = 'explainedModels/fold1-0.9776-1.0000-f_model.h5'
    pickle_model_path = 'explainedModels/svm.pkl'
    patch = 64
    stride = 16
    filtered_input = True
    pickle_model = True
    invert_black_bg = True

    if not pickle_model:
        model = tf.keras.models.load_model(model_path)
        input_channels = model.layers[0].input_shape[0][-1]
    else:
        model = pickle.load(open(pickle_model_path, 'rb'))
        input_channels = 1

    images, labels = get_images(image_indexes, filtered=filtered_input, input_channels=input_channels,
                                invert_black_bg=invert_black_bg)

    fig, axs = plt.subplots(nrows=len(image_indexes), ncols=1, constrained_layout=True)
    title = 'Inverted Occlusion Method'
    if pickle_model:
        title = title + ' - SVM'
    else:
        title = title + ' - EfficientNetB3'
    fig.suptitle(title)
    for ax in axs:
        ax.remove()

    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    row = 0
    for image, label, subfig in zip(images, labels, subfigs):
        if not pickle_model:
            pred = model.predict(image)[0]
        else:
            orientations = 18
            pixels_per_cell = (16, 16)
            cells_per_block = (2, 2)
            fd = hog(image[0, :, :, 0], orientations=orientations, pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, visualize=False, channel_axis=None)
            X = np.zeros(shape=(1,) + fd.shape)
            X[0] = fd
            pred = model.predict_proba(X)[0]

        label = label[0]

        top_label_index = np.argmax(pred)
        top_label_probability = pred[top_label_index]

        title = "True: " + LABELS[np.argmax(label)] + " - Predicted: " + LABELS[np.argmax(pred)]

        patches_probabilities = get_occluded_probabilities(image[0, :, :, :], model, top_label_index,
                                                           patch_size=patch, stride=stride, sklearn_model=pickle_model)
        patch_heatmap = 1 - (top_label_probability - patches_probabilities)

        subfig.suptitle(title)
        axs = subfig.subplots(nrows=1, ncols=3)
        axs[0].imshow(image[0, :, :, 0])
        axs[0].axis('off')

        tmp = axs[1].imshow(patch_heatmap, cmap='Blues', vmin=patch_heatmap.min(), vmax=patch_heatmap.max())
        subfig.colorbar(tmp, ax=axs[1])

        axs[2].imshow(image[0, :, :, 0], cmap='gray')
        axs[2].pcolormesh(cv2.resize(patch_heatmap, image[0, :, :, 0].shape, interpolation=cv2.INTER_CUBIC),
                          cmap='Blues', alpha=0.50)

    plt.show()
