import math
import os

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import execution_settings
from dataset_utils import get_images
import tensorflow as tf
import numpy as np

execution_settings.set_gpu()

LABELS = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]


def get_occluded_probabilities(img, predictor, index, patch_size=16, stride=1):
    """Iteratively set a square patch of an image to zero.

    Parameters:
        img (np.array): The image to occlude.
        predictor (keras.model): Keras model to predict occluded images
        index (int): The label to be analyzed
        patch_size (int, optional): The size of the patch to set to zero. Default is 16.
        stride (int, optional): The stride between patches. Default is 1

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
            # Set a patch of the image to zero
            occluded_image[y:y + patch_size, x:x + patch_size] = 0
            occluded_imgs.append(np.expand_dims(occluded_image, axis=-1))

        occluded_imgs = np.asarray(occluded_imgs)
        predictions = predictor.predict(occluded_imgs)
        predictions = predictions[:, index]
        probs[line] = predictions
        line += 1
        occluded_imgs = []

    return probs


if __name__ == '__main__':
    image_indexes = [31, 39, 99]  # N, P, T
    model_path = 'explainedModels/0.9116-0.9416-f_model.h5'
    patch = 32
    stride = 16

    images, labels = get_images(image_indexes)
    model = tf.keras.models.load_model(model_path)

    fig, axs = plt.subplots(nrows=len(image_indexes), ncols=1, constrained_layout=True)
    fig.suptitle('Occlusion Method')
    for ax in axs:
        ax.remove()

    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    row = 0
    for image, label, subfig in zip(images, labels, subfigs):
        pred = model.predict(image)[0]
        label = label[0]

        top_label_index = np.argmax(pred)
        top_label_probability = pred[top_label_index]

        title = "True: " + LABELS[np.argmax(label)] + " - Predicted: " + LABELS[np.argmax(pred)]

        patches_probabilities = get_occluded_probabilities(image[0, :, :, 0], model, top_label_index, patch_size=patch, stride=stride)
        patch_heatmap = top_label_probability - patches_probabilities

        subfig.suptitle(title)
        axs = subfig.subplots(nrows=1, ncols=2)
        axs[0].imshow(image[0, :, :, 0])
        axs[0].axis('off')

        tmp = axs[1].imshow(patch_heatmap, cmap='RdBu', vmin=-patch_heatmap.max(), vmax=patch_heatmap.max())
        subfig.colorbar(tmp, ax=axs[1])

    plt.show()
