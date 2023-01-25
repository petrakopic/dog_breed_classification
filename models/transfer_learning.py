from typing import Tuple
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
from utils import preprocess_image

MODELS = (Xception, ResNet50, InceptionV3)
PREPROCESS_FUNCTIONS = (xcep_preprocess, resnet_preprocess, inception_preprocess)


def prepare_features(
    folder_path: str,
):
    """
    Run the three pretrained models on the image dataset to get the high level features.
    Remove the last layers from the model and concatenates the output to one tensor.
    Use the tensor as the input to the trainable custom model.
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
    """
    Model consisting of the dropout layer and the one fully connected layer with 120
    output classes. The activation function is softmax.
    :param input_shape:
    :return:
    """

    input_img = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dropout(0.4)(input_img)
    x = tf.keras.layers.Dense(units=120, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_img, outputs=x)
    return model
