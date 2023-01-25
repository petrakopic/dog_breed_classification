"""
Main function for running the training and the validation of the model. All the function can also
be run from the command line using the commands from the cli/ folder.
"""
import tensorflow as tf
from models.transfer_learning import prepare_features, custom_model
from utils import prepare_target
from utils.config import ModelConfig


def train(config_file: str = "model_config.yaml"):
    conf = ModelConfig(config_file)

    target = prepare_target.run(folder_path=conf.get_value("data_train_folder"))
    features = prepare_features(folder_path=conf.get_value("data_train_folder"))

    model = custom_model(input_shape=features.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        x=features,
        y=target,
        batch_size=conf.get_value("batch_size"),
        epochs=conf.get_value("epochs"),
    )
    model.save(conf.get_value("model_out_path"))
    return model


def predict(config_file: str = "model_config.yaml"):
    """
    Loads the pretrained model from the memory and predicts the results given the images
    in the test folder.
    :param config_file:
    :return: List of class names given the images in the test folder.
    """
    conf = ModelConfig(config_file)
    model = tf.keras.models.load_model(conf.get_value("model_out_path"))
    input_images = prepare_features(folder_path=conf.get_value("data_test_folder"))

    predicted = model.predict(input_images)

    class_names = [prepare_target.probs_to_class_name(p) for p in predicted]
    return class_names
