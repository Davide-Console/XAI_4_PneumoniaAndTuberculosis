import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import execution_settings

import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
import tensorflow as tf
from skimage.segmentation import mark_boundaries
from dataset_utils import make_list_of_patients, cross_validation_splits, DataGen

execution_settings.set_gpu()


def predict4lime(img2):
    # print(img2.shape)
    return model.predict(img2[:, :, :, 0] * 255)  # the explain_instance function calls skimage.color.rgb2gray. So I
    # need to take only yhe first channel to make the prediction


def generate_prediction_sample(lime_exp, exp_class, weight=0.0, show_positive=True, hide_background=True):
    """
    Method to display and highlight super-pixels used by the black-box model to make predictions
    """
    img, mask = lime_exp.get_image_and_mask(exp_class,
                                            positive_only=show_positive,
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


def get_images(indexes):
    patients = make_list_of_patients()
    X_train_folds, y_train_folds, X_test_folds, y_test_folds = cross_validation_splits(data=patients)

    x_test_fold0 = X_test_folds[0]
    y_test_fold0 = y_test_folds[0]

    batch_size = 1
    dg_val0 = DataGen(batch_size, (256, 256), x_test_fold0, y_test_fold0)

    imgs = []
    lbls = []
    for index in indexes:
        img, lbl = dg_val0.__getitem__(index)
        imgs.append(img)
        lbls.append(lbl)

    return imgs, lbls


if __name__ == '__main__':
    image_indexes = [0, 31, 99]
    model_path = 'explainedModels/0.9116-0.9416-f_model.h5'

    pred2explain = 0  # index of the label to be analyzed. 0 means the label with higher probability
    min_importance = 0.5  # minimum POSITIVE importance, in percentage, of superpixels to be shown

    assert 3 > pred2explain > -1
    assert 1 > min_importance > 0

    images, labels = get_images(image_indexes)
    model = tf.keras.models.load_model(model_path)

    explainer = lime_image.LimeImageExplainer()
    for image, label in zip(images, labels):
        exp = explainer.explain_instance(image[0, :, :, 0] / 255, predict4lime, top_labels=3, hide_color=0,
                                         num_samples=1000)

        label = label[0]
        pred = model.predict(image)

        print("True class:", end=' ')
        print(label)
        print("Pred probabilities:", end=' ')
        print(pred[0])
        print("Explainer top labels:", end=' ')
        print(exp.top_labels)

        plt.imshow(exp.segments)
        plt.axis('off')
        plt.show()

        heatmap = explanation_heatmap(exp, exp.top_labels[pred2explain])
        plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
        plt.colorbar()
        plt.show()

        min_weight = min_importance * np.max(heatmap)

        masked_image = generate_prediction_sample(exp, exp.top_labels[pred2explain], show_positive=True, hide_background=False)
        plt.imshow(masked_image)
        plt.axis('off')
        plt.show()
