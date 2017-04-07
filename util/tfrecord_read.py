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
from util.utils import Dequantize, to_multi_categorical
from globals import RGB_FEAT_SIZE, AUDIO_FEAT_SIZE, MAX_FRAMES, NUM_CLASSES, \
    FRM_LVL_FEAT_NAMES, VID_LVL_FEAT_NAMES, GLOBAL_FEAT_NAMES, \
    VIDEO_TRAIN_DIR, VIDEO_VAL_DIR, VIDEO_TEST_DIR, \
    FRAME_TRAIN_DIR, FRAME_VAL_DIR, FRAME_TEST_DIR, \
    EX_DATA_DIR

class yt8mFeatureReader:
    def __init__(self,
                 num_classes=NUM_CLASSES,
                 feature_size=[RGB_FEAT_SIZE, AUDIO_FEAT_SIZE],
                 feature_name=FRM_LVL_FEAT_NAMES,
                 max_frames=MAX_FRAMES,
                 sequence_data=True):
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.feature_name = feature_name
        self.max_frames = max_frames
        self.sequence_data = sequence_data

    def prepare_reader(self,
                       filename_queue,
                       max_quantized_value=2,
                       min_quantized_value=-2):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        context_features, sequence_features = {"video_id": tf.FixedLenFeature([], tf.string),
                                               "labels": tf.VarLenFeature(tf.int64)}, None
        if self.sequence_data:
            sequence_features = {self.feature_name[0]: tf.FixedLenSequenceFeature([], dtype=tf.string),
                                 self.feature_name[1]: tf.FixedLenSequenceFeature([], dtype=tf.string), }
        else:
            context_features[self.feature_name[0]] = tf.FixedLenFeature(self.feature_size[0], tf.float32)
            context_features[self.feature_name[1]] = tf.FixedLenFeature(self.feature_size[1], tf.float32)

        contexts, features = tf.parse_single_sequence_example(serialized_example,
                                                              context_features=context_features,
                                                              sequence_features=sequence_features)
        labels = (tf.cast(contexts["labels"].values, tf.int64))

        if self.sequence_data:
            decoded_features = tf.reshape(tf.cast(tf.decode_raw(features[self.feature_name[0]], tf.uint8), tf.float32),
                                          [-1, self.feature_size[0]])
            video_matrix = Dequantize(decoded_features, max_quantized_value, min_quantized_value)

            decoded_features = tf.reshape(tf.cast(tf.decode_raw(features[self.feature_name[1]], tf.uint8), tf.float32),
                                          [-1, self.feature_size[1]])
            audio_matrix = Dequantize(decoded_features, max_quantized_value, min_quantized_value)

            num_frames = tf.minimum(tf.shape(decoded_features)[0], self.max_frames)
        else:
            video_matrix = contexts[self.feature_name[0]]
            audio_matrix = contexts[self.feature_name[1]]
            num_frames = tf.constant(-1)

        # Pad or truncate to 'max_frames' frames.
        # video_matrix = resize_axis(video_matrix, 0, self.max_frames)
        return contexts["video_id"], video_matrix, audio_matrix, labels, num_frames

def tfrecord_reader(filename_queue, data_lvl):  # , outpath):
    labelcsv = pd.read_csv(EX_DATA_DIR + "label_names.csv")
    alllabels = labelcsv["label_name"].values

    if data_lvl == 'frame':
        print "%s level features: %s" & (data_lvl, str(FRM_LVL_FEAT_NAMES))
        reader = yt8mFeatureReader()
    elif data_lvl == 'video':
        print "%s level features: %s" % (data_lvl, str(VID_LVL_FEAT_NAMES))
        reader = yt8mFeatureReader(feature_name=VID_LVL_FEAT_NAMES, sequence_data=False)

    vals = reader.prepare_reader(filename_queue)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # f = codecs.open(outpath, "w", encoding='utf-8')
        try:
            counter = 0
            recordlist = []

            while not coord.should_stop():
                counter += 1
                vid, vfeatures, afeatures, labels, numframes = sess.run(vals)
                vfeatureslist = np.round(vfeatures, decimals=5).tolist()
                afeatureslist = np.round(afeatures, decimals=5).tolist()
                labelName = alllabels[labels]
                if data_lvl == 'frame':
                    record = {GLOBAL_FEAT_NAMES[0]: vid,
                              FRM_LVL_FEAT_NAMES[0]: vfeatureslist,
                              FRM_LVL_FEAT_NAMES[1]: afeatureslist,
                              FRM_LVL_FEAT_NAMES[2]: numframes,
                              GLOBAL_FEAT_NAMES[1]: labels,
                              GLOBAL_FEAT_NAMES[2]: labelName,
                              }
                elif data_lvl == 'video':
                    record = {GLOBAL_FEAT_NAMES[0]: vid,
                              VID_LVL_FEAT_NAMES[0]: vfeatureslist,
                              VID_LVL_FEAT_NAMES[1]: afeatureslist,
                              GLOBAL_FEAT_NAMES[1]: labels,
                              GLOBAL_FEAT_NAMES[2]: labelName,
                              }
                recordlist.append(record)

        except tf.errors.OutOfRangeError:
            print('Finished extracting from tfrecord data.')
        finally:
            coord.request_stop()
            coord.join(threads)
        return recordlist


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


if __name__ == '__main__':
    # data level: "frame" and "video"
    # feature type: "rgb" and "audio"
    data_lvl = "frame"
    feature_type = "rgb"
    X_train, Y_train = get_data(EX_DATA_DIR, data_lvl, feature_type)
