# Keras Heatmaps for 3D Convolutions

![supported versions](https://img.shields.io/badge/python-2.7%2C%203.5-green.svg)
[![Bugzilla bug status](https://img.shields.io/github/issues/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions.svg)](https://github.com/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions/issues)
[![license](https://img.shields.io/github/license/alexandrosstergiou/Keras-3DCNN-Heatmap.svg)](https://github.com/alexandrosstergiou/Keras-3DCNN-Heatmap/blob/master/LICENSE)
[![GitHub language count](https://img.shields.io/badge/library-pytorch%2Ckeras-blue.svg)](https://keras.io/)

## About
The purpose of this repository is to provide a multi-dimensional implementation of the heatmap visualisation for Deep Learning models in video data. The output of the script is a folder containing consecutive frames. The base for the script has been Gabriel de Marmiesse's easy-to-follow [repo](https://github.com/gabrieldemarmiesse/heatmaps) of his heatmap implementation.

For videos, these frames can be turned to video/GIFs with tools such as [`ImageMagic`](https://github.com/ImageMagick/ImageMagick) or [`imageio`](http://imageio.github.io/).





## Installation
Please make sure, Git is installed in your machine:
```sh
$ sudo apt-get update
$ sudo apt-get install git
$ git clone https://github.com/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions.git
```

## Dependencies
Currently the repository supports either Keras or Pytorch models.  `OpenCV` was used for processes in the frame level. For resizing the  to the original video dimensions we used `scipy.ndimage`.
```sh
$ pip install opencv-python
$ pip install scipy
```

## License
MIT


## Contact
Alexandros Stergiou

a.g.stergiou at uu.nl

Any queries or suggestions are much appreciated!
