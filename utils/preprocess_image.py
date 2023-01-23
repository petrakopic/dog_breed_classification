from typing import Callable
import os
from keras_preprocessing.image import load_img
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = (224,224,3)


def run(folder_path:str, preprocess_func: Callable)->np.array:
    image_paths = [f"{folder_path}/{filename}" for filename in os.listdir(folder_path)]
    imgs = [load_img(img_path, target_size=IMAGE_SIZE) for img_path in image_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_func(img_array)
