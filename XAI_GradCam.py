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
    plt.imshow(superimposed_img)
    plt.show()


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
    model = tf.keras.models.load_model('explainedModels/0.9116-0.9416-f_model.h5')
    model.compile(optimizer='Adam',
                  loss=keras.losses.categorical_crossentropy,
                  metrics='accuracy')

    print(model.summary())

    patients = make_list_of_patients()
    X_train_folds, y_train_folds, X_test_folds, y_test_folds = stratified_cross_validation_splits(data=patients)
    x_train_fold0 = X_train_folds[0]
    y_train_fold0 = y_train_folds[0]
    x_test_fold0 = X_test_folds[0]
    y_test_fold0 = y_test_folds[0]
    batch_size = 1
    dg_train0 = DataGen(batch_size, (256, 256), x_train_fold0, y_train_fold0)
    dg_val0 = DataGen(batch_size, (256, 256), x_test_fold0, y_test_fold0)
    for i in range(dg_train0.__len__()):
        img, label = dg_train0.__getitem__(i)

        grad_cam, predictions = GradCam(model, np.expand_dims(img[0, :, :, :], axis=0), label,
                                        'global_average_pooling2d')

        display_gradcam(img[0, :, :, :], grad_cam)

    for i in range(dg_val0.__len__()):
        img, label = dg_val0.__getitem__(i)
        label = np.argmax(label)

        grad_cam, predictions = GradCam(model, np.expand_dims(img[0, :, :, :], axis=0), label,
                                        'global_average_pooling2d')

        display_gradcam(img[0, :, :, :], grad_cam)
