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

def get_data(data_path,
             data_lvl,
             feature_type="rgb",
             batch=32,
             preprocess=None,
             shuffle=True,
             num_epochs=1):
    files_pattern = "train*.tfrecord"
    data_files = gfile.Glob(data_path + files_pattern)
    filename_queue = tf.train.string_input_producer(data_files, num_epochs=num_epochs, shuffle=shuffle)
    tfrecord_list = tfrecord_reader(filename_queue, data_lvl)
    vid_train = np.array([tfrecord_list[i][GLOBAL_FEAT_NAMES[0]] for i, _ in enumerate(tfrecord_list)])
    labels_train = np.array([tfrecord_list[i][GLOBAL_FEAT_NAMES[1]] for i, _ in enumerate(tfrecord_list)])

    if data_lvl == "video":
        mean_rgb_train = np.array([tfrecord_list[i][VID_LVL_FEAT_NAMES[0]] for i, _ in enumerate(tfrecord_list)])
        mean_audio_train = np.array([tfrecord_list[i][VID_LVL_FEAT_NAMES[1]] for i, _ in enumerate(tfrecord_list)])
        if feature_type == "rgb":
            X_train = mean_rgb_train
        elif feature_type == "audio":
            X_train = mean_audio_train
    elif data_lvl == "frame":
        rgb_train = np.array([tfrecord_list[i][FRM_LVL_FEAT_NAMES[0]] for i, _ in enumerate(tfrecord_list)])
        audio_train = np.array([tfrecord_list[i][FRM_LVL_FEAT_NAMES[1]] for i, _ in enumerate(tfrecord_list)])
        if feature_type == "rgb":
            X_train = rgb_train
        elif feature_type == "audio":
            X_train = audio_train

    Y_train = to_multi_categorical(labels_train, NUM_CLASSES)
    print "get_data done."
    return X_train, Y_train

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """
        Dequantize the feature from the byte format to the float format.
        Args:
        feat_vector: the input 1-d vector.
        max_quantized_value: the maximum of the quantized value.
        min_quantized_value: the minimum of the quantized value.
        Returns:
        A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias

def to_multi_categorical(labels, num_classes):
    for i, label in enumerate(labels):
        if i == 0:
            result = np.array([np.sum(np_utils.to_categorical(label, num_classes), axis=0)])
        else:
            result = np.concatenate((result, np.array([np.sum(np_utils.to_categorical(label, num_classes), axis=0)])),
                                    axis=0)
    return result