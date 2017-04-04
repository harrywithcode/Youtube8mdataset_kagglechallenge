import os
import sys
import cv2
import Image
import fnmatch
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
%matplotlib inline

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
