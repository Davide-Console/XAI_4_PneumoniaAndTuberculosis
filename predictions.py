import os

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dataset_utils import *
import zipfile
import numpy as np
import execution_settings

execution_settings.set_gpu()


def save_to_dataframe(files, labels):
    data = {'file': files, 'label': labels}
    df = pd.DataFrame(data)
    return df


def prepare_test_set(test_set_dir='test_set', zipped_dataset='test_set.zip'):
    if not os.path.isdir(test_set_dir):
        os.makedirs(test_set_dir, exist_ok=True)
        with zipfile.ZipFile(zipped_dataset, 'r') as zip_ref:
            zip_ref.extractall(test_set_dir)


if __name__ == '__main__':
    test_set_directory = 'test_set'
    prepare_test_set()
    model_path = 'explainedModels/fold4-0.9714-1.0000-f_model.h5'

    img_size = (256, 256)
    model = tf.keras.models.load_model(model_path)

    input_img_paths = sorted(
        [
            os.path.join(test_set_directory, fname)
            for fname in os.listdir(test_set_directory)
        ]
    )

    predictions = []

    for i in tqdm(range(len(input_img_paths))):
        path = input_img_paths[i]
        image = cv2.imread(path, 0)  # read as grayscale
        image = cv2.resize(image, img_size, interpolation=cv2.INTER_CUBIC)
        image = invert_image(image)
        image = cv2.medianBlur(image, ksize=5)
        image = ndimage.uniform_filter(image, size=3)
        image = gray2rgb(image)
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        predictions.append(np.argmax(prediction))

    stripped_filenames = [f.split('/')[-1] for f in input_img_paths]
    file_name = 'labels_test.csv'
    df = save_to_dataframe(stripped_filenames, predictions)
    df = df.replace(NORMAL, 'N')
    df = df.replace(PNEUMONIA, 'P')
    df = df.replace(TUBERCULOSIS, 'T')
    df.to_csv(file_name, index=False)
