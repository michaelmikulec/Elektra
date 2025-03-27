from setuptools import setup, find_packages

setup(
  name="Elektra",
  version="0.1.0",
  description="Software application to classify harmful brain activities in EEG signals using deep learning and machine learning",
  author="Michael Mikulec, Kevin Tran, Intisarul Huda, Manuel Jimenez, Estuardo Melendez",
  author_email="michaelmikulec1@gmail.com",
  url="https://github.com/michaelmikulec/Elektra",
  packages=find_packages(),
  install_requires=[
    "pandas",
    "pyarrow",
    "fastparquet",
    "torch",
    "torchvision",
    "torchaudio",

  ],
  classifiers=[
    "Programming Language :: Python :: 3.13",
    "Operating System :: OS Independent"
  ],
  python_requires=">=3.13.2"
)
