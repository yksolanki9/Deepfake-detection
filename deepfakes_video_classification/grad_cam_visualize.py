import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.optimizers import Nadam
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras_efficientnets import EfficientNetB5, EfficientNetB0
from vis.utils import utils
from vis.visualization import visualize_cam
import numpy as np
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import cv2

test_data = pd.read_csv("test_vids_label.csv")

videos = test_data["vids_list"]
true_labels = test_data["label"]
classlabel = true_labels


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings

# Create face detector
mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device="cuda:0"
)


def plot_map(grads, img, subtitle=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[1].imshow(img)
    i = axes[1].imshow(grads, cmap="jet", alpha=0.3)
    axes[1].axis("off")
    fig.colorbar(i)
    # plt.suptitle("Pr(class={}) = {:5.2f}".format(
    #                   classlabel[class_idx],
    #                   y_pred[0,class_idx]))
    plt.savefig(subtitle)


def cnn_model(model_name, img_size):
    """
    Model definition using Xception net architecture
    """
    input_size = (img_size, img_size, 3)
    if model_name == "xception":
        baseModel = Xception(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "iv3":
        baseModel = InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "irv2":
        baseModel = InceptionResNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "resnet":
        baseModel = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "nasnet":
        baseModel = NASNetLarge(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "ef0":
        baseModel = EfficientNetB0(
            input_size,
            weights="imagenet",
            include_top=False
        )
    elif model_name == "ef5":
        baseModel = EfficientNetB5(
            input_size,
            weights="imagenet",
            include_top=False
        )

    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
    # headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
    #     headModel
    # )
    # headModel = Dropout(0.5)(headModel)
    headModel = Dropout(0.5)(headModel)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=baseModel.input, outputs=predictions)

    for layer in baseModel.layers:
        layer.trainable = True

    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model


def main():
    model = xception_model()

    model.load_weights("trained_wts/xception_50_I.hdf5")

    print("Weights loaded...")

    # Utility to search for layer index by name. Alternatively we can
    # specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, "dense_4")
    # Swap softmax with linear
    model.layers[layer_idx].activation = keras.activations.linear
    model = utils.apply_modifications(model)

    penultimate_layer_idx = utils.find_layer_idx(model, "block14_sepconv2_act")

    # block14_sepconv2_act
    # dense_4

    counter = 0
    for i in videos[:4]:
        cap = cv2.VideoCapture(i)
        batches = []
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            try:
                face = face.permute(1, 2, 0).int().numpy()
                batches.append(face)
            except AttributeError:
                print("Image Skipping")
            if counter == 4:
                break
            counter += 1
        batches = np.asarray(batches).astype("float32")
        batches /= 255
        print(batches.shape)

        predictions = model.predict(batches)
        pred_mean = np.mean(predictions, axis=0)
        y_pred = pred_mean.argmax(0)

        imgs = batches[0]
        print(imgs.shape)
        seed_input = imgs
        class_idx = y_pred
        grad_top1 = visualize_cam(
            model,
            layer_idx,
            class_idx,
            seed_input,
            penultimate_layer_idx=penultimate_layer_idx,  # None,
            backprop_modifier=None,
            grad_modifier=None,
        )

        plot_map(
            grad_top1,
            img=seed_input,
            subtitle="Class Activation maps" + str(counter)
        )
        print("Figure saved..")
        cap.release()

        counter += 1


if __name__ == '__main__':
    main()
