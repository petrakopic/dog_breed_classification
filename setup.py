from setuptools import find_packages, setup
from pathlib import Path


def _open_readme(file_path: str) -> str:
    with open(Path(".", file_path), encoding="utf-8") as f:
        return f.read()


setup(
    name="dog_breed_identification",
    description="Module for training and validating ML models on the dataset from the kaggle competition",
    long_description=_open_readme("README.md"),
    install_requires=["tensorflow", "numpy", "click", "matplotlib"],
    python_requires=">=3.7",
)
