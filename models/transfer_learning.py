from typing import Tuple, Optional
import tensorflow as tf
from models.data_augmentation import data_augmenter
from utils.preprocess_image import get_input_dataset
from utils.prepare_target import get_output_dataset


def train(
    folder_path: str,
    base_learning_rate: float = 0.001,
    model_out_path: str = "data/models/my_model_weights.h5",
):

    inputs = get_input_dataset(folder_path=folder_path)
    outputs = get_output_dataset(folder_path=folder_path)
    dataset = tf.data.Dataset.zip((inputs, outputs))

    input_shape = inputs.element_spec.shape[1:]
    model = build_model(input_shape=input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(dataset, batch_size=100, epochs=10)
    model.save(model_out_path)
    return model


def build_model(input_shape: Tuple[int, int, int]):

    input_img = tf.keras.Input(shape=input_shape)
    input_img = data_augmenter()(input_img)

    base_model = model_from_resnet(input_shape=input_shape)
    # This can be adjusted to freeze only some layers:
    base_model.trainable = False

    x = base_model(input_img)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    f = tf.keras.layers.Flatten()(x)
    output1 = tf.keras.layers.Dense(units=120, activation="softmax")(f)

    custom_model = conv_skip_model_block(input_shape=input_shape)
    x2 = custom_model(input_img)
    f2 = tf.keras.layers.Flatten()(x2)
    output2 = tf.keras.layers.Dense(units=120, activation="softmax")(f2)

    output = tf.keras.layers.Add()([output1, output2])

    model = tf.keras.Model(input_img, output)
    return model


def conv_skip_model_block(
    input_shape: Tuple[int, int, int],
    training=True,
    initializer=tf.keras.initializers.random_uniform,
):
    """
    Add one layer of convolution followed by batch normalization and relu activation and
    connect to the input image with skip connection.
    :param input_shape:
    :param training:
    :param initializer:
    :return:
    """
    input_img = tf.keras.Input(shape=input_shape)
    input_shortcut = input_img
    x = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=1,
        strides=(1, 1),
        padding="valid",
        kernel_initializer=initializer(seed=0),
    )(input_img)
    x = tf.keras.layers.BatchNormalization(axis=3)(x, training=training)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Add()([x, input_shortcut])
    x = tf.keras.layers.Activation("relu")(x)

    model = tf.keras.Model(inputs=input_img, outputs=x)
    return model


def model_from_resnet(input_shape: Optional[Tuple] = None, classes: int = 1000):
    """
    load pretrained ResNet50 model. The model is trained to classify different objects.
    :param classes:
    :param input_shape:
    :return:
    """

    return tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=classes,
    )
