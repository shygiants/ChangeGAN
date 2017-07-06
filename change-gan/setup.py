"""Setup script for change-gan."""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='change-gan',
    version='0.1',
    include_package_data=True,
    author='Sanghoon Yoon',
    author_email='shygiants@gmail.com',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    description='ChangeGAN',
)
