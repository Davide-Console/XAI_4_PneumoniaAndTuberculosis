import tensorflow as tf
import tensorflow.keras as keras
import execution_settings
import cv2
execution_settings.set_gpu()

from dataset_utils import *
from XAI_GradCam import *
from XAI_Lime import *
from XAI_Occlusion import *

def predict4lime(img2):
    # print(img2.shape)
    return model.predict(img2[:, :, :, 0] * 255)  # the explain_instance function calls skimage.color.rgb2gray. So I
    # need to take only yhe first channel to make the prediction

if __name__ == '__main__':
    patch = 32
    stride = 16
    pred2explain = 0
    min_importance = 0.25
    model_path='explainedModels/0.9116-0.9416-f_model.h5'
    image_indexes = [31, 39, 99]

    images, labels = get_images(image_indexes)

    model = tf.keras.models.load_model(model_path)

    print(model.summary())

    fig, axs = plt.subplots(nrows=len(image_indexes), ncols=1, constrained_layout=True)
    fig.suptitle('Explainers')
    for ax in axs:
        ax.remove()
    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]
    row = 0

    explainer = lime_image.LimeImageExplainer()
    for image, label, subfig in zip(images, labels, subfigs):

        # LIME
        exp = explainer.explain_instance(image[0, :, :, 0] / 255, predict4lime, top_labels=3, hide_color=0,
                                         num_samples=1000, random_seed=333)
        label = label[0]
        pred = model.predict(image)
        print("True class:", end=' ')
        print(label)
        print("Pred probabilities:", end=' ')
        pred = [round(i, 3) for i in pred[0]]
        print(pred)
        print("Explainer top labels:", end=' ')
        print(exp.top_labels)
        title = "True: " + LABELS[np.argmax(label)] + " - Predicted: " + LABELS[np.argmax(pred)]
        subfig.suptitle(title)
        axs = subfig.subplots(nrows=1, ncols=5)
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

        # GRADCAM
        label = np.argmax(label)
        heatmap, predictions = GradCam(model, np.expand_dims(image[0, :, :, :], axis=0), label, 'global_average_pooling2d')
        img=image[0, :, :, :]
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * 0.4 + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        axs[3].imshow(superimposed_img)
        axs[3].axis('off')


        # OCCLUSION
        top_label_index = np.argmax(pred)
        top_label_probability = pred[top_label_index]
        patches_probabilities = get_occluded_probabilities(image[0, :, :, 0], model, top_label_index, patch_size=patch, stride=stride)
        patch_heatmap = top_label_probability - patches_probabilities
        heatmap_occlusion = cv2.resize(patch_heatmap, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        axs[4].imshow(heatmap_occlusion, cmap='RdBu', vmin=-patch_heatmap.max(), vmax=patch_heatmap.max())
        axs[4].imshow(image[0, :, :, 0], cmap='bone', alpha=0.5)
        axs[4].axis('off')


        row += 1

    plt.show()