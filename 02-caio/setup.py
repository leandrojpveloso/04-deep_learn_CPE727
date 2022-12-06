import pathlib

import pkg_resources
import setuptools


from setuptools import setup, find_packages
import pip

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]
    
setup(
    name="sonarcode",
    packages=find_packages(),
    version="0.1.8",
    description="A short description of the project.",
    install_requires=install_requires,
    author='jovenil',
)
