import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from dataset_utils import *

from architectures import *
import execution_settings

execution_settings.set_gpu()


def evaluate(model, test_datagen):
    Y = []
    for j in range(test_datagen.__len__()):
        x, y = test_datagen.__getitem__(j)
        Y.append(y)

    y_true = np.concatenate(Y)
    y_pred = model.predict(test_datagen)

    print(classification_report(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), digits=4,
                                output_dict=False))


if __name__ == '__main__':
    model_path = 'explainedModels/0.9710-0.9991-f_model.h5'
    model = tf.keras.models.load_model(model_path)

    patients = make_list_of_patients()
    patients_train, patients_test = test_split(data=patients)
    X_test, y_test = dataframe2lists(patients_test)

    dg_test = DataGen(32, (256, 256), X_test, y_test)

    evaluate(model, dg_test)
