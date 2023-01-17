"""
Usages:
python -m cli.commands show_multiple --image_path "data/test/" --num_img 7
"""

import click
from cli import options
from utils.show_image import plot_multiple, plot_one, plot_many


@click.group(invoke_without_command=True)
def cli():
    """
    Image plotting
    """


@cli.command("show_one")
@options.image_path
def show_one(image_path: str):
    plot_one(image_path=image_path)


@cli.command("show_multiple")
@options.image_path
@options.num_images
def show_multiple(image_path: str, num_img: int):
    plot_multiple(folder_path=image_path, num_img=num_img)


@cli.command("show_many")
@options.image_path
@options.num_images
def show_many(image_path: str, num_img: int):
    plot_many(folder_path=image_path, num_img=num_img)


if __name__ == "__main__":
    cli(prog_name="cli.commands")
