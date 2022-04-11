#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="mcfa",
    version="0.1",
    description="Mixtures of Common Factor Analyzers",
    packages=find_packages(),
    py_modules=["mcfa"],
    install_requires=[
        "tensorflow",
        "tensorflow-probability",
        "matplotlib",
        "numpy",
        "tqdm",
        "pandas",
        "pyppca",
        "scipy",
        "scikit-learn",
    ],
)
