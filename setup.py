import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

requirements = [
    "numpy",
    "pandas",
    "plotly",
    "scipy",
    "scikit-learn",
    "torch",
]

# This call to setup() does all the work
setup(
    name="svGPFA",
    version="1.0.0",
    description="Python implementation of the svGPFA algorithm (Duncker and Sahani, 2018)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/joacorapela/svGPFA",
    author="Joaquin Rapela",
    author_email="j.rapela@ucl.ac.uk",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("ci",)),
    include_package_data=True,
    install_requires=requirements,
)
