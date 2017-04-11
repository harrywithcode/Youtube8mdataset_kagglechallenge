import os
import csv
import sys
import cv2
import random
import base64
import fnmatch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.utils import np_utils
from globals import RGB_FEAT_SIZE, AUDIO_FEAT_SIZE, MAX_FRAMES, NUM_CLASSES, \
    FRM_LVL_FEAT_NAMES, VID_LVL_FEAT_NAMES, GLOBAL_FEAT_NAMES, \
    VIDEO_TRAIN_DIR, VIDEO_VAL_DIR, VIDEO_TEST_DIR, \
    FRAME_TRAIN_DIR, FRAME_VAL_DIR, FRAME_TEST_DIR, \
    EX_DATA_DIR

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias

def to_multi_categorical(labels, num_classes):
    result = np.array([np.sum(np_utils.to_categorical(label, num_classes), axis=0) for label in labels])
    return result

def get_framediff(frame_features):
    frmdiff = -np.diff(frame_features, axis=0)
    return  list(frmdiff)
