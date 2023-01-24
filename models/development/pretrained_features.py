from typing import Tuple, List
import tensorflow as tf
from tqdm import tqdm

import numpy as np
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as inception_preprocess,
)
from utils import preprocess_image, prepare_target

MODELS = (Xception, ResNet50, InceptionV3)
PREPROCESS_FUNCTIONS = (xcep_preprocess, resnet_preprocess, inception_preprocess)


def main(
    folder_path: str,
    model_out_path: str,
    batch_size: int = 10,
    epochs: int = 10,
):
    target = prepare_target.run(folder_path=folder_path)
    features = prepare_features(folder_path=folder_path)

    model = custom_model(input_shape=features.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(x=features, y=target, batch_size=batch_size, epochs=epochs)
    model.save(model_out_path)
    return model


def predict(folder_path: str, model_out_path: str) -> List[str]:
    """

    :param folder_path: folder to the test set
    :param model_out_path: path to the pretrained model
    :return:
    """
    model = tf.keras.models.load_model(model_out_path)

    input_images = prepare_features(folder_path=folder_path)

    predicted = model.predict(input_images)
    predicted = [(i == i.max()).astype(int) for i in predicted]
    class_names = [prepare_target.target_to_class_name(p) for p in predicted]
    return class_names


def prepare_features(
    folder_path: str,
):
    """
    Run the pretrained models on the image dataset to get the set of features extracted from each
    of the models.
    Model uses these features concatenated to one vector as the input.
    Model consists of the dropout and FC layer.

    :param folder_path:
    :return:
    """
    features = []

    for model, preprocess_func in tqdm(
        zip(MODELS, PREPROCESS_FUNCTIONS), desc=f"Running the model...", total=3
    ):
        inputs = preprocess_image.run(
            folder_path=folder_path, preprocess_func=preprocess_func
        )
        models_features = model(
            include_top=False, weights="imagenet", pooling="avg"
        ).predict(inputs)
        features.append(models_features)
    features = np.concatenate(features, axis=-1)

    return features


def custom_model(
    input_shape: Tuple[int, int, int],
):
    input_img = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dropout(0.7)(input_img)
    x = tf.keras.layers.Dense(units=120, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_img, outputs=x)
    return model
