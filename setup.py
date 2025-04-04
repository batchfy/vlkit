from setuptools import setup, find_packages

__version__ = "0.1.0b11"

setup(
    name='vlkit',
    version=__version__,
    description='Vision and Learning Toolkit',
    url='https://github.com/batchfy/vlkit',
    author_email='batchfy@gmail.com',
    author='https://batchfy.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "numpy>=2.2.4",
        "pillow>=11.1.0",
        "scikit-image>=0.25.2",
    ],
)
