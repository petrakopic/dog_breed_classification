from typing import Tuple, Optional
import tensorflow as tf
from models.development.data_augmentation import data_augmenter
from utils import train_dataset
from utils.config import ModelConfig
from utils.preprocess_image import IMAGE_SIZE


def train(
    config_file: str = "model_config.yaml",
):
    """
    Main function for training the model.
    :param config_file: path to the config file which contains the model parameters
    :return:
    """
    conf = ModelConfig(config_file)

    dataset = train_dataset.prepare(
        folder_path=conf.get_value("data_train_folder"),
        batch_size=conf.get_value("batch_size"),
        drop_remainder=True,
        preprocess_func=lambda x: x,
    )

    model = combine_models(input_shape=IMAGE_SIZE, num_classes=120, training=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=conf.get_value("learning_rate")
        ),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(dataset, epochs=conf.get_value("epochs"))
    model.save(conf.get_value("model_out_path"))
    return model


def combine_models(
    input_shape: Tuple[int, int, int],
    num_classes: int = 120,
    training: bool = True,
):
    """
    Combine the results of the pretrained resnet50 model
    (https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50),
    xception model (https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception)
    and the custom model.
    :return:
    """

    input_img = tf.keras.Input(shape=input_shape)
    input_img = data_augmenter()(input_img)

    resnet_model = model_from_resnet(input_shape=input_shape)
    output_resnet = resnet_model(input_img)

    xception_model = model_from_xception(input_shape=input_shape)
    output_xception = xception_model(input_img)

    custom_model = conv_skip_model_block(input_shape=input_shape, training=training)
    output_custom = custom_model(input_img)
    output_combined = tf.keras.layers.concatenate(
        [output_resnet, output_xception, output_custom], axis=-1
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(output_combined)

    if training:
        x = tf.keras.layers.Dropout(0.4)(x)

    output = tf.keras.layers.Dense(units=num_classes, activation="relu")(x)
    model = tf.keras.Model(input_img, output)

    return model


def conv_skip_model_block(
    input_shape: Tuple[int, int, int],
    training: bool = True,
):
    """
    Model consists of three blocks Conv->BatchNorm->Relu
    Number of trainable parameters is: 3338
    The output shape of is (7,7,20)
    :param input_shape:
    :param num_classes: Number of classes in the target variable
    :param training: True is the model is in the training phase, otherwise False.
    :return:
    """

    input_img = tf.keras.Input(shape=input_shape)

    x = _conv_block(
        input_img,
        filters=3,
        kernel_size=1,
        strides=(2, 2),
        padding="valid",
        training=training,
    )

    x = _conv_block(
        x, filters=10, kernel_size=1, strides=(4, 4), padding="valid", training=training
    )

    x = _conv_block(
        x, filters=20, kernel_size=4, strides=(4, 4), padding="valid", training=training
    )

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


def model_from_resnet(input_shape: Optional[Tuple] = None):
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
        classes=1000,
    )

    base_model.trainable = False
    x = base_model(input_img)

    model = tf.keras.Model(inputs=input_img, outputs=x)
    return model


def model_from_xception(input_shape: Optional[Tuple] = None):
    """
    load pretrained ResNet50 model. The model is trained to classify different objects.
    :param num_classes: Number of classes in the target variable
    :param input_shape:
    :return:
    """
    input_img = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.xception.Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=500,
    )

    base_model.trainable = False
    x = base_model(input_img)
    model = tf.keras.Model(inputs=input_img, outputs=x)
    return model
