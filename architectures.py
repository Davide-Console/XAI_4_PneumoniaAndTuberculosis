import os
import tempfile

import matplotlib.pyplot as plt
import tensorflow as tf
import visualkeras
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAvgPool2D
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model, model_from_json


def attach_final_layers(model, classes):
    """
    This function takes in a model and the number of classes and returns the model with the final layers attached.
    The final layers are a Global Average Pooling layer, a Dropout layer and a Dense layer with the number of classes
    as the number of neurons. Parameters: model (keras.Model): The model to which the final layers are to be
    attached. classes (int): The number of classes in the dataset. Returns: keras.Model: The model with the final
    layers attached.
    """
    model = GlobalAvgPool2D()(model)
    model = Dropout(rate=0.4)(model)

    activation = 'softmax' if classes > 2 else 'sigmoid'
    classes = 1 if classes == 2 else classes

    output_layer = Dense(classes, activation=activation,
                         bias_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001),
                         activity_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))(model)

    return output_layer


def add_regularization(model, l1, l2):
    """
        This function adds regularization to a model.
        Parameters
        ----------
        model : keras.models.Model
            The model to add regularization to.
        l1 : float
            The l1 regularization coefficient.
        l2 : float
            The l2 regularization coefficient.
        Returns
        -------
        keras.models.Model
            The model with regularization.
    """
    regularizer = regularizers.l1_l2(l1=l1, l2=l2)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def get_EfficientNetB3(weights=None, input_shape=(256, 256, 1), classes=8, regularize=True, l1=0.00001, l2=0.00001):
    """
    This function returns a compiled EfficientNetB3 model.
    Parameters
    ----------
    weights : str
        The path to the weights file to be loaded or "imagenet" to load pre-trained network.
        If None, the model will be initialized with random weights.
    input_shape : tuple
        The shape of the input layer.
    classes : int
        The number of classes in the output layer.
    regularize : bool
        Whether to add regularization to the model.
    l1 : float
        The L1 regularization coefficient.
    l2 : float
        The L2 regularization coefficient.
    Returns
    -------
    model : keras.Model
        A compiled EfficientNetB3 model.
    """
    if weights == 'imagenet':
        input_shape = (256, 256, 3)
    model = EfficientNetB3(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=classes)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


if __name__ == '__main__':
    net = get_EfficientNetB3(classes=3, weights="imagenet")

    print(net.summary())

    plt.imshow(visualkeras.layered_view(net, legend=True, scale_xy=15))
    plt.show()
