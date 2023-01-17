import click


image_path = click.option(
    "--image_path",
    "--path",
    default="",
    help="Path to the image folder or image.",
)

num_images = click.option(
    "--num_img",
    "--n",
    default=6,
    help="Number of images to show",
)
