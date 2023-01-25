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
@click.option("--config_file", default="model_config.yaml")
def train_model(
    config_file: str,
):
    train(
        config_file=config_file
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
