# @title Implementation
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import execution_settings

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.python.keras.optimizer_v1 import Adam

from dataset_utils import *

execution_settings.set_gpu()

LABELS = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]


def display_gradcam(img, heatmap, emphasize=False, thresh=None):
    def sigmoid(x, a, b, c):
        return c / (1 + np.exp(-a * (x - b)))

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]

    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)

    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))

    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Display Grad CAM
    return superimposed_img


def GradCam(model, img_array, label, layer_name, eps=1e-8):
    '''
    Creates a grad-cam heatmap given a model and a layer name contained with that model


    Args:
      model: tf model
      img_array: (img_width x img_width) numpy array
      layer_name: str


    Returns
      uint8 numpy array with shape (img_height, img_width)

    '''

    gradModel = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).input,
                 model.output])

    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(img_array, tf.float32)  # we use the preprocessed image
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, np.argmax(label)]

    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)

    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    # heatmap = (heatmap * 255).astype("uint8")
    # return the resulting heatmap to the calling function
    return heatmap, predictions


if __name__ == '__main__':
    image_indexes = [31, 39, 99]  # N, P, T
    model_path = 'explainedModels/0.9722-0.9999-f_model.h5'
    filtered_input = True  # If DataGenFiltered is used during train set this to true
    model = tf.keras.models.load_model(model_path)
    input_channels = model.layers[0].input_shape[0][-1]

    model.compile(optimizer='Adam',
                  loss=keras.losses.categorical_crossentropy,
                  metrics='accuracy')

    images, labels = get_images(image_indexes, filtered=filtered_input, input_channels=input_channels)

    fig, axs = plt.subplots(nrows=len(image_indexes), ncols=1, constrained_layout=True)
    fig.suptitle('GradCam Explainer')
    for ax in axs:
        ax.remove()

    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    for image, label, subfig in zip(images, labels, subfigs):
        grad_cam, predictions = GradCam(model, np.expand_dims(image[0, :, :, :], axis=0), label,
                                        'global_average_pooling2d')

        result = display_gradcam(image[0, :, :, :], grad_cam)

        label = label[0]
        pred = model.predict(image)

        title = "True: " + LABELS[np.argmax(label)] + " - Predicted: " + LABELS[np.argmax(pred)]
        subfig.suptitle(title)
        axs = subfig.subplots(nrows=1, ncols=2)
        axs[0].imshow(image[0, :, :, :]/255)
        axs[0].axis('off')
        axs[1].imshow(result)
        axs[1].axis('off')

    plt.show()
