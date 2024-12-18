import os
from setuptools import setup, find_packages
from setuptools import Extension
from distutils.command.build import build as build_orig

from setuptools import dist

__version__ = "0.1.0b11"

setup(name='vlkit',
    version=__version__,
    description='vision and learning kit',
    url='https://github.com/vlkit/vlkit',
    author_email='kz@kaizhao.net',
    license='MIT',
    packages=find_packages(),
    install_requires=["numpy", 'scikit-image'],
    zip_safe=False,
)
