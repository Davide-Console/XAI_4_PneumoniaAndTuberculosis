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

model = tf.keras.models.load_model('float_model/0.6069-0.6039-f_model.h5')

patients = make_list_of_patients()
X_train_folds, y_train_folds, X_test_folds, y_test_folds = cross_validation_splits(data=patients)

x_train_fold0 = X_train_folds[0]
y_train_fold0 = y_train_folds[0]

x_test_fold0 = X_test_folds[0]
y_test_fold0 = y_test_folds[0]

batch_size = 1
dg_train0 = DataGen(batch_size, (256, 256), x_train_fold0, y_train_fold0)
dg_val0 = DataGen(batch_size, (256, 256), x_test_fold0, y_test_fold0)

img, label = dg_val0.__getitem__(0)
label = label[0]
pred = model.predict(img)

explainer = lime_image.LimeImageExplainer()


def predict4lime(img2):
    # print(img2.shape)
    return model.predict(img2[:, :, :, 0] * 255)  # the explain_instance function calls skimage.color.rgb2gray. So I
    # need to take only yhe first channel to make the prediction


exp = explainer.explain_instance(img[0, :, :, 0] / 255, predict4lime, top_labels=3, hide_color=0, num_samples=1000)
print("True class:", end=' ')
print(label)
print("Pred probabilities:", end=' ')
print(pred[0])
print("Explainer top labels:", end=' ')
print(exp.top_labels)

plt.imshow(exp.segments)
plt.axis('off')
plt.show()


def generate_prediction_sample(exp, exp_class, weight=0.0, show_positive=True, hide_background=True):
    """
    Method to display and highlight super-pixels used by the black-box model to make predictions
    """
    image, mask = exp.get_image_and_mask(exp_class,
                                         positive_only=show_positive,
                                         num_features=6,
                                         hide_rest=hide_background,
                                         min_weight=weight
                                         )
    plt.imshow(mark_boundaries(image, mask, color=(1, 0, 0)))
    plt.axis('off')
    plt.show()


def explanation_heatmap(exp, exp_class):
    """
    Using heat-map to highlight the importance of each super-pixel for the model prediction
    """
    dict_heatmap = dict(exp.local_exp[exp_class])
    heatmap = np.vectorize(dict_heatmap.get)(exp.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.show()


pred2explain = 0  # index of the label to be analyzed. 0 means the label with higher probability

explanation_heatmap(exp, exp.top_labels[pred2explain])
generate_prediction_sample(exp, exp.top_labels[pred2explain], show_positive=True, hide_background=False)
