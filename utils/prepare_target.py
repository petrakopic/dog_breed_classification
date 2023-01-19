from functools import lru_cache
import tensorflow as tf
import os
import re
from typing import Dict, Optional, List, Tuple
import numpy as np
import csv

IMAGE_FORMAT = ".jpg"
LABELS_PATH = "data/labels.csv"


def get_output_dataset(folder_path: str) -> tf.data.Dataset:

    image_names = os.listdir(folder_path)
    labels = [fetch_label(re.sub(IMAGE_FORMAT, "", img)) for img in image_names]
    return tf.data.Dataset.from_tensors(labels)


def fetch_label(img_id: str) -> Optional[np.array]:
    """
    Returns a 120-dimensional vector with 1 on the index which corresponds to the class of the image
    img_id.
    :param img_id:
    :return:
    """
    labels_mapping = _get_labels_mapping()

    class_name = labels_mapping.get(img_id)
    if not class_name:
        raise ValueError(f"Class unknown for the img_id {img_id}")

    return class_name_to_target(class_name)


def target_to_class_name(target_vector: np.array) -> str:
    """
    Given the target vector (120-dimensional vector) returns the class name
    :param target_vector:
    :return:
    """
    class_number = _numpy_to_int(target_vector=target_vector)
    classes = _get_all_classes_list(LABELS_PATH)
    class_name = _map_number_to_class_name(classes).get(class_number)
    return class_name


def class_name_to_target(class_name) -> np.array:
    """
    Given the class name returns the 120-dimensional vector used as a target variable in training the model.
    :param class_name:
    :return:
    """
    classes = _get_all_classes_list(LABELS_PATH)
    class_number = _map_class_name_to_number(classes=classes).get(class_name)
    target_variable = _int_to_numpy(index=class_number, classes_num=len(classes))
    return target_variable


@lru_cache(1)
def _get_all_classes_list(labels_path):
    breeds_set = set()
    with open(labels_path) as file:
        reader = csv.DictReader(file)
        for line in reader:
            breeds_set.add(line["breed"])
    return sorted(list(breeds_set))


@lru_cache(1)
def _get_labels_mapping() -> Dict[str, np.array]:
    """
    Returns the mapping from the image id to the class name. Only used for train and test.
    :return: Mapping image_id: class name.
            Example:
                'ffa6a8d29ce57eb760d0f182abada4bf': 'english_foxhound'
    """
    with open(LABELS_PATH) as file:
        reader = csv.DictReader(file)
        return {line["id"]: line["breed"] for line in reader}


def _map_class_name_to_number(classes) -> Dict[str, int]:
    """
    Given the list of classes returns the mapping from class name to the integer number
    :param classes:
    :return:
    """
    return {class_name: number for class_name, number in _fetch_pairs(classes=classes)}


def _map_number_to_class_name(classes) -> Dict[int, str]:
    """
    Given the list of classes returns the mapping from the integer number to the class name.
    :param classes:
    :return:
    """
    return {number: class_name for class_name, number in _fetch_pairs(classes=classes)}


def _fetch_pairs(classes: List[str]) -> List[Tuple[str, int]]:
    """
    Returns the list of tuples (class_name, class_number)
    :param classes:
    :return:
    """
    return [(class_name, i) for i, class_name in enumerate(classes)]


def _int_to_numpy(index: int, classes_num: int) -> np.array:
    """
    One-hot encoded representation for the given index and number of classes 
    :param index:
    :param classes_num:
    :return:
    """
    target_vector = np.zeros(classes_num)
    target_vector[index] = 1
    return target_vector


def _numpy_to_int(target_vector: np.array) -> int:
    return target_vector.tolist().index(1)
