import tensorflow as tf


def data_augmenter() -> tf.keras.Sequential:

    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(tf.keras.layers.RandomFlip("horizontal"))
    data_augmentation.add(tf.keras.layers.RandomRotation(0.2))
    return data_augmentation
