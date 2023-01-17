import math
from logging import getLogger
from typing import List
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt


def show_image(image_path: str):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()


def show_multiple(folder_path: str, num_img: int = 6):
    """
    Input parameter is the path to the folder with pictures.
    Function shows num_pics from the folder in one plot with two columns.
    If the argument num_pics is not given it shows the first 6 photos.
    :param folder_path: path to the folder with images
    :param num_img: number of pictures to show
    :return:
    """
    log = getLogger("plot_img")
    img_paths = _fetch_first_n_paths(folder_path=folder_path, n=num_img)

    nrows = math.floor(len(img_paths) / 2)
    fig, ax = plt.subplots(nrows=math.ceil(len(img_paths) / 2), ncols=2)

    try:
        if num_img == 2:
            images = [Image.open(img) for img in img_paths[2 * i: 2 * i + 2]]
            ax[0].imshow(images[0])
            ax[1].imshow(images[1])
        else:
            for i in range(nrows):
                images = [Image.open(img) for img in img_paths[2 * i: 2 * i + 2]]
                ax[i][0].imshow(images[0])
                ax[i][1].imshow(images[1])

    except UnidentifiedImageError as err:
        log.error(f"Please make sure that folder path contains only images. {str(err)}")
        return

    if nrows < len(img_paths) / 2:
        last = Image.open(img_paths[-1])
        ax[nrows][0].imshow(last)

    plt.show()


def show_many(folder_path: str, num_img: int = 6):

    """
    Input parameter is the path to the folder with pictures.
    Function shows num_pics from the folder, each picture in its own plot.
    :param folder_path:
    :param num_img: number of pictures to show

    :return:
    """

    img_paths = _fetch_first_n_paths(folder_path=folder_path, n=num_img)

    for i in range(num_img):
        img = Image.open(img_paths[i])
        plt.imshow(img)
        plt.show()


def _fetch_first_n_paths(folder_path: str, n: int) -> List[str]:
    full_paths = [
        str(Path(folder_path).joinpath(file_name))
        for file_name in os.listdir(folder_path)
    ]
    return full_paths[: min(n, len(full_paths))]
