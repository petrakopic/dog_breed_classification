"""
Usages:
python -m cli.commands show_multiple --image_path "data/test/" --num_img 7
"""

import click
from cli import options
from main import train
from utils import visualize


@click.group(invoke_without_command=True)
def cli():
    """
    Image plotting
    """


@cli.command("train_model")
@options.image_path
@click.option("--learning_rate", default=0.001)
@click.option("--model_out_path", default="data/")
@click.option("--batch_size", default=100)
@click.option("--epochs", default=10)
def train_model(
    image_path: str,
    learning_rate: str,
    model_out_path: str,
    batch_size: int = 10,
    epochs: int = 10,
):
    train(
        folder_path=image_path,
        base_learning_rate=learning_rate,
        model_out_path=model_out_path,
        batch_size=batch_size,
        epochs=epochs,
    )


@cli.command("show_one")
@options.image_path
def show_one(image_path: str):
    visualize.show_image(image_path=image_path)


@cli.command("show_multiple")
@options.image_path
@options.num_images
def show_multiple(image_path: str, num_img: int):
    visualize.show_multiple(folder_path=image_path, num_img=num_img)


@cli.command("show_many")
@options.image_path
@options.num_images
def show_many(image_path: str, num_img: int):
    visualize.show_many(folder_path=image_path, num_img=num_img)


if __name__ == "__main__":
    cli(prog_name="cli.commands")
