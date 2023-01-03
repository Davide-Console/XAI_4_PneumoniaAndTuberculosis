import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import classification_report
from dataset_utils import *
from tqdm import tqdm
import execution_settings

execution_settings.set_gpu()


def evaluate(model, test_datagen, data_aug=False):
    """
    Evaluate the performance of a model on a test dataset.

    Parameters:
    model (keras.Model): The model to be evaluated.
    test_datagen (keras.preprocessing.image.ImageDataGenerator): The test dataset.
    data_aug (bool, optional): Whether to use data augmentation. Defaults to False.

    Returns:
    None: The function prints the classification report of the model.
    """
    Y = []
    for j in range(test_datagen.__len__()):
        x, y = test_datagen.__getitem__(j)
        Y.append(y)

    y_true = np.concatenate(Y)

    y_pred = model.predict(test_datagen)
    if data_aug:
        for i in tqdm(range(17)):
            y_pred = y_pred + model.predict(test_datagen)

    print(classification_report(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), digits=4,
                                output_dict=False))


if __name__ == '__main__':
    # Evaluate the model specified in the path.
    # Please, set the preprocessing according to the ones used during training.
    model_path = 'explainedModels/fold4-0.9714-1.0000-f_model.h5'
    data_aug = False
    filtering = True
    invert_black_bg = True

    weights = "imagenet"
    model = tf.keras.models.load_model(model_path)

    patients = make_list_of_patients()
    patients_train, patients_test = test_split(data=patients)
    X_test, y_test = dataframe2lists(patients_test)

    dg_test = DataGen(32, (256, 256), X_test, y_test, weights=weights, filtering=filtering,
                      data_aug=data_aug, invert_black_bg=invert_black_bg)

    evaluate(model, dg_test, data_aug=data_aug)
