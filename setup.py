#!/usr/bin/env python3
"""
Mostly taken from https://github.com/rochacbruno/python-project-template/blob/main/setup.py
"""
import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]


setup(
    name="lost_in_the_middle",
    version=read("src", "lost_in_the_middle", "VERSION"),
    description="Development repository for analyzing how LLMs use retrieved context.",
    url="https://github.com/nelson-liu/lost-in-the-middle",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=read_requirements("requirements.txt"),
)
