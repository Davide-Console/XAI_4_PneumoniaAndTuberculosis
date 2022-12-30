import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from dataset_utils import *
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
import execution_settings

def get_callbacks():
    tboard = 'tb_logs'
    os.makedirs(tboard, exist_ok=True)
    tb_call = TensorBoard(log_dir=tboard)

    chkpt_dir = 'AE_model'
    os.makedirs(chkpt_dir, exist_ok=True)
    chkpt_call = ModelCheckpoint(
        filepath=os.path.join(chkpt_dir, '{loss:.4f}f_model.h5'),
        monitor='loss',
        verbose=1,
        save_best_only=True)

    logdir = 'train_log_AE.csv'
    csv_logger = CSVLogger(logdir, append=True, separator=';')

    return [tb_call, chkpt_call, csv_logger]

def get_autoencoder():
    # The encoding process
    input_img = Input(shape=(256, 256, 1))

    ############
    # Encoding #
    ############

    # Conv1 #
    x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Conv2 #
    x = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Conv3 #
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Conv4 #
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Note:
    # padding is a hyperparameter for either 'valid' or 'same'.
    # "valid" means "no padding".
    # "same" results in padding the input such that the output has the same length as the original input.

    ############
    # Decoding #
    ############

    # DeConv1
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)

    # DeConv1
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # DeConv2
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Deconv3
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    # Declare the model
    autoencoder = Model(input_img, decoded)


    autoencoder.compile(optimizer='adadelta', loss='mse')

    return autoencoder


if __name__ == '__main__':
    seed = 1
    classes = 3
    batch_size = 32

    patients = make_list_of_patients()

    patients_train, patients_test = test_split(data=patients)
    X_train_folds, y_train_folds, X_val_folds, y_val_folds = stratified_cross_validation_splits(data=patients_train)
    X_test, y_test = dataframe2lists(patients_test)

    x_train_fold0 = X_train_folds[0]
    y_train_fold0 = y_train_folds[0]

    x_val_fold0 = X_val_folds[0]
    y_val_fold0 = y_val_folds[0]

    dg_train0 = DG_autoencoder(batch_size, (256, 256), x_train_fold0, y_train_fold0)
    dg_val0 = DG_autoencoder(batch_size, (256, 256), x_val_fold0, y_val_fold0)

    model = get_autoencoder()
    print(model.summary())
    callbacks = get_callbacks()
    #Train the model
    model.fit(dg_train0,
             epochs=1,
             batch_size=batch_size,
             shuffle=True,
             validation_data=dg_val0,
             callbacks=callbacks,
             verbose=1
             )

    n_img = 3
    fig, axs = plt.subplots(nrows=n_img, ncols=3, constrained_layout=True)
    fig.suptitle('< withoutNoise Synthetic withNoise >')

    for i in range(n_img):
        x, y = dg_train0.__getitem__(i)
        y_pred = model.predict(x)
        axs[i, 0].imshow(y[0, :, :, 0], cmap='bone')
        axs[i, 1].imshow(y_pred[0, :, :, 0], cmap='bone')
        axs[i, 2].imshow(x[0, :, :, 0], cmap='bone')
    plt.show()

