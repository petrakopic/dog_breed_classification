"""
Main function for running the training and the validation of the model. All the function can also
be run from the command line using the commands from the cli/ folder.
"""
import tensorflow as tf
from models.transfer_learning import combine_models
from utils import train_dataset
from utils.preprocess_image import IMAGE_SIZE


def train(
    folder_path: str,
    base_learning_rate,
    model_out_path: str,
    batch_size: int = 10,
    epochs: int = 10,
):
    """
    Main function for training the model.
    :param folder_path:
    :param base_learning_rate:
    :param model_out_path:
    :param batch_size:
    :param epochs:

    :return:
    """

    dataset = train_dataset.prepare(
        folder_path=folder_path,
        batch_size=batch_size,
        drop_remainder=True,
        preprocess_func=lambda x: x,
    )

    model = combine_models(input_shape=IMAGE_SIZE, num_classes=120, training=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(dataset, epochs=epochs)
    model.save(model_out_path)
    return model


# ToDo: add validation. Add config file for loading model paramters.
