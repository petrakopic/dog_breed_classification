from typing import Callable
import tensorflow as tf
from utils import preprocess_image, prepare_target


def prepare(
    folder_path: str, batch_size: int, drop_remainder: bool, preprocess_func: Callable
) -> tf.data.Dataset:
    """
    Function that prepares the dataset for the model.fit function
    :param folder_path:
    :param batch_size:
    :param drop_remainder:
    :param preprocess_func:

    :return:
    """
    inputs = preprocess_image.run(
        folder_path=folder_path, preprocess_func=preprocess_func
    )
    outputs = prepare_target.run(folder_path=folder_path)

    inputs = tf.data.Dataset.from_tensor_slices(inputs)
    outputs = tf.data.Dataset.from_tensor_slices(outputs)

    return tf.data.Dataset.zip((inputs, outputs)).batch(
        batch_size=batch_size, drop_remainder=drop_remainder
    )
