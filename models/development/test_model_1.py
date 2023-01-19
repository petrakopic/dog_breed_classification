from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from models.data_augmentation import data_augmenter


def combine_models(
    input_shape: Tuple[int, int, int],
    num_classes: int = 120,
    training: bool = True,
):
    """
    Combine the results of the pretrained resnet50 model.
    (https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)
    and the custom model. The last layers of both networks are global average pooling
    followed by softmax with the number of classes given by num_classes.
    :return:
    """

    input_img = tf.keras.Input(shape=input_shape)
    input_img = data_augmenter()(input_img)

    base_model = model_from_resnet(input_shape=input_shape, num_classes=num_classes)
    output1 = base_model(input_img)

    custom_model = conv_skip_model_block(
        input_shape=input_shape, num_classes=num_classes, training=training
    )
    output2 = custom_model(input_img)

    # output = np.concatenate(output1, output2)

    output = tf.keras.layers.Add()([output1, output2])
    model = tf.keras.Model(input_img, output)
    return model


def conv_skip_model_block(
    input_shape: Tuple[int, int, int],
    num_classes: int = 120,
    training: bool = True,
):
    """
    Model consists of three blocks Conv->BatchNorm->Relu and one skip connection.
    Number of trainable parameters is:
    The output shape of is (7,7,120)
    :param input_shape:
    :param num_classes: Number of classes in the target variable
    :param training: True is the model is in the training phase, otherwise False.
    :return:
    """

    input_img = tf.keras.Input(shape=input_shape)
    input_shortcut = input_img

    x = _conv_block(
        input_img,
        filters=3,
        kernel_size=1,
        strides=(2, 2),
        padding="valid",
        training=training,
    )

    x = _conv_block(
        x, filters=10, kernel_size=1, strides=(2, 2), padding="valid", training=training
    )

    x = _conv_block(
        x, filters=20, kernel_size=4, strides=(4, 4), padding="valid", training=training
    )

    x = tf.keras.layers.Add()([x, input_shortcut])

    # Decrease the dimension before flattening:
    x = tf.keras.layers.AveragePooling2D(
        pool_size=4, strides=2, padding="valid", data_format="channels_last"
    )(x)

    # Global average pooling flattens the input to the num_channels (use instead of FCL to avoid
    # over fitting)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_img, outputs=x)
    return model


def _conv_block(
    x_0: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: Tuple[int, int],
    padding: str,
    training: bool,
):
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=tf.keras.initializers.random_uniform,
    )(x_0)
    x = tf.keras.layers.BatchNormalization(axis=3)(x, training=training)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def model_from_resnet(input_shape: Optional[Tuple] = None, num_classes: int = 120):
    """
    load pretrained ResNet50 model. The model is trained to classify different objects.
    :param num_classes: Number of classes in the target variable
    :param input_shape:
    :return:
    """
    input_img = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=500,
    )

    base_model.trainable = False
    x = base_model(input_img)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_img, outputs=output)
    return model
