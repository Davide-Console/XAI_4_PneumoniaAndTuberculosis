import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from skimage.feature import hog
import execution_settings
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
import tensorflow as tf
from skimage.segmentation import mark_boundaries
from dataset_utils import make_list_of_patients, stratified_cross_validation_splits, DataGen, get_images

execution_settings.set_gpu()

LABELS = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]


def predict4lime(img2):
    # print(img2.shape)
    return model.predict(img2[:, :, :, 0] * 255)  # the explain_instance function calls skimage.color.rgb2gray. So I
    # need to take only yhe first channel to make the prediction


def predict4limeRGB(img2):
    # print(img2.shape)
    return model.predict(img2[:, :, :, :] * 255)  # the explain_instance function calls skimage.color.rgb2gray. So I
    # need to take only yhe first channel to make the prediction


def predict4limeSVM(img2):
    ex_img = img2[0, :, :, 0]
    orientations = 18
    pixels_per_cell = (16, 16)
    cells_per_block = (2, 2)

    fd = hog(ex_img, orientations=orientations, pixels_per_cell=pixels_per_cell,
             cells_per_block=cells_per_block, visualize=False, channel_axis=None)

    X = np.zeros(shape=(len(img2),) + fd.shape)
    for i in range(len(img2)):
        img = img2[i, :, :, 0] * 255
        fd = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block, visualize=False, channel_axis=None)
        X[i] = fd

    y_pred = model.predict_proba(X)

    return y_pred


def generate_prediction_sample(lime_exp, exp_class, weight=0.0, show_positive_only=True, hide_background=True):
    """
    Method to display and highlight super-pixels used by the black-box model to make predictions
    """
    img, mask = lime_exp.get_image_and_mask(exp_class,
                                            positive_only=show_positive_only,
                                            num_features=6,
                                            hide_rest=hide_background,
                                            min_weight=weight
                                            )
    return mark_boundaries(img, mask, color=(1, 0, 0))


def explanation_heatmap(lime_exp, exp_class):
    """
    Using heat-map to highlight the importance of each super-pixel for the model prediction
    """
    dict_heatmap = dict(lime_exp.local_exp[exp_class])
    return np.vectorize(dict_heatmap.get)(lime_exp.segments)


if __name__ == '__main__':
    image_indexes = [31, 39, 99]  # N, P, T
    model_path = 'explainedModels/0.9776-1.0000-f_model.h5'
    pickle_model_path = 'explainedModels/svm.pkl'
    filtered_input = True  # If DataGenFiltered is used during train set this to true
    pickle_model = True

    pred2explain = 0  # index of the label to be analyzed. 0 means the label with higher probability
    min_importance = 0.25  # minimum POSITIVE/NEGATIVE importance, in percentage, of superpixels to be shown

    assert 3 > pred2explain > -1
    assert 1 > min_importance > 0

    if not pickle_model:
        model = tf.keras.models.load_model(model_path)
        input_channels = model.layers[0].input_shape[0][-1]
    else:
        model = pickle.load(open(pickle_model_path, 'rb'))
        input_channels = 1
        filtered_input = False

    images, labels = get_images(image_indexes, filtered=filtered_input, input_channels=input_channels)

    fig, axs = plt.subplots(nrows=len(image_indexes), ncols=1, constrained_layout=True)
    fig.suptitle('LIME Explainer')
    for ax in axs:
        ax.remove()

    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    row = 0

    explainer = lime_image.LimeImageExplainer()
    for image, label, subfig in zip(images, labels, subfigs):
        if input_channels == 1:
            if not pickle_model:
                exp = explainer.explain_instance(image[0, :, :, 0] / 255, predict4lime, top_labels=3, hide_color=0,
                                                 num_samples=1000, random_seed=333)
            else:
                exp = explainer.explain_instance(image[0, :, :, 0] / 255, predict4limeSVM, top_labels=3, hide_color=0,
                                                 num_samples=1000, random_seed=333)
        else:
            exp = explainer.explain_instance(image[0, :, :, :] / 255, predict4limeRGB, top_labels=3, hide_color=0,
                                             num_samples=1000, random_seed=333)

        label = label[0]
        if not pickle_model:
            pred = model.predict(image)
        else:
            orientations = 18
            pixels_per_cell = (16, 16)
            cells_per_block = (2, 2)
            fd = hog(image[0, :, :, 0], orientations=orientations, pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, visualize=False, channel_axis=None)
            X = np.zeros(shape=(1,) + fd.shape)
            X[0] = fd
            pred = model.predict_proba(X)

        print("True class:", end=' ')
        print(label)
        print("Pred probabilities:", end=' ')
        pred = [round(i, 3) for i in pred[0]]
        print(pred)
        print("Explainer top labels:", end=' ')
        print(exp.top_labels)

        title = "True: " + LABELS[np.argmax(label)] + " - Predicted: " + LABELS[np.argmax(pred)]
        subfig.suptitle(title)
        axs = subfig.subplots(nrows=1, ncols=3)
        axs[0].imshow(exp.segments)
        axs[0].axis('off')

        heatmap = explanation_heatmap(exp, exp.top_labels[pred2explain])
        tmp = axs[1].imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
        subfig.colorbar(tmp, ax=axs[1])

        min_weight = min_importance * np.max(heatmap)

        masked_image = generate_prediction_sample(exp, exp.top_labels[pred2explain], show_positive_only=False,
                                                  hide_background=False, weight=min_weight)
        axs[2].imshow(masked_image)
        axs[2].axis('off')
        row += 1

    plt.show()
