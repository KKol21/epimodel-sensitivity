from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="emsa",
    version="1.0.2",
    author="Kolos Kovács",
    author_email="kovkol21@gmail.com",
    description="Efficient sensitivity analysis and evaluation of epidemiological models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KKol21/epimodel_sensitivity",
    packages=find_packages(exclude=("tests", "tests.*", "emsa_examples", "emsa_examples.*")),
    install_requires=[
        "torch~=2.3.0",
        "torchode~=0.2.0",
        "numpy~=1.26.4",
        "matplotlib~=3.7.5",
        "tqdm~=4.66.5",
        "smt~=2.6.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
