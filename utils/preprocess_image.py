from typing import Tuple
import os
import tensorflow as tf


def get_input_dataset(folder_path: str):
    """
    Loads the images to dataset object
    :param folder_path:
    :return:
    """

    image_names = os.listdir(folder_path)
    dataset = tf.data.Dataset.from_tensors(
        list(
            map(
                lambda img_name: _prepare_for_resnet(img_name, folder_path), image_names
            )
        )
    )
    return dataset


def _prepare_for_resnet(file_name: str, folder_path: str) -> tf.data.Dataset:
    """
    Prepare the input data for the ResNet50 network architecture.
    First normalize and resize the image and then pass it to the resnet50 preprocess.
    :param file_name: name of the image file
    :param folder_path: path to the folder with images.
    :return:
    """
    images = _preprocess_img(filename=file_name, folder_path=folder_path, output_size=(224, 224))
    return tf.keras.applications.resnet50.preprocess_input(images, data_format=None)


def _preprocess_img(filename: str, folder_path: str, output_size: Tuple[int, int]) -> tf.Tensor:
    image_string = tf.io.read_file(f"{folder_path}/{filename}")
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image = _resize(image=image, output_size=output_size)
    return image


def _resize(image: tf.Tensor, output_size: Tuple[int, int]) -> tf.Tensor:
    """
    Resizing the image to 224x224 dimension
    :param image:
    :param output_size:
    :return:
    """
    return tf.image.resize(image, output_size)
