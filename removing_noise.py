from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from dataset_utils import *
import matplotlib.pyplot as plt


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
    x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Conv 3 #
    x = Conv2D(filters=8, kernel_size=(5, 5), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Note:
    # padding is a hyperparameter for either 'valid' or 'same'.
    # "valid" means "no padding".
    # "same" results in padding the input such that the output has the same length as the original input.

    ############
    # Decoding #
    ############

    # DeConv1
    x = Conv2D(8, (5, 5), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)

    # DeConv2
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Deconv3
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

    # Declare the model
    autoencoder = Model(input_img, decoded)


    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')

    return autoencoder


if __name__ == '__main__':
    seed = 1
    classes = 3
    batch_size = 16

    patients = make_list_of_patients()

    patients_train, patients_test = test_split(data=patients)
    X_train_folds, y_train_folds, X_val_folds, y_val_folds = stratified_cross_validation_splits(data=patients_train)

    x_train_fold0 = X_train_folds[0]
    y_train_fold0 = y_train_folds[0]

    x_test_fold0 = X_val_folds[0]
    y_test_fold0 = y_val_folds[0]

    dg_train0 = DataGen(batch_size, (256, 256), x_train_fold0, y_train_fold0)
    dg_val0 = DataGen(batch_size, (256, 256), x_test_fold0, y_test_fold0)

    model = get_autoencoder()
    # image_indexes = [0, 31, 99]

    x_train = []
    y_train = []
    for index in range(dg_train0.__len__()):
        img, lbl = dg_train0.__getitem__(index)
        x_train.append(img)
        y_train.append(lbl)

    x_train =np.concatenate(x_train)

    x_val = []
    y_val = []
    for index in range(dg_val0.__len__()):
        img, lbl = dg_val0.__getitem__(index)
        x_val.append(img)
        y_val.append(lbl)

    x_val =np.concatenate(x_val)
    # Train the model
    # model.fit(x_train, x_train,
    #          epochs=100,
    #          batch_size=batch_size,
    #          shuffle=True,
    #          validation_data=(x_val, x_val),
    #          verbose=1
    #          )
    # decoded_imgs = model.predict(x_val)

    # n = 10

    #plt.figure(figsize=(20, 4))
    # for i in range(n):
        # display original
      #  ax = plt.subplot(2, n, i + 1)
      #  plt.imshow(x_val[i].reshape(28, 28))
      #  plt.gray()
      #  ax.get_xaxis().set_visible(False)
      #  ax.get_yaxis().set_visible(False)

        # display reconstruction
       # ax = plt.subplot(2, n, i + 1 + n)
        #plt.imshow(decoded_imgs[i].reshape(28, 28))
        #plt.gray()
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
    #plt.show()

    noise_factor = 0.4
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_val_noisy = x_val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_val.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_val_noisy = np.clip(x_val_noisy, 0., 1.)

    model.fit(x_train_noisy, x_train,
              epochs=10,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_val_noisy, x_val),
              verbose=1
              )

    decoded_imgs_noisy = model.predict(x_val)

    n = 10

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_val_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
