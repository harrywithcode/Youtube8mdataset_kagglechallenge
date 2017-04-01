import os
import csv
import sys
import cv2
import random
import base64
import fnmatch
import numpy as np
import codecs, json
import pandas as pd
from PIL import Image
import skimage.io as io
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.preprocessing.image import ImageDataGenerator
%matplotlib inline

DATA_DIR = "/home/jasonlee/Documents/dmproject_kaggle/data/"
NUM_CLASSES=4716
VID_LVL_FEAT_NAMES=["mean_rgb", "mean_audio"]
FRM_LVL_FEAT_NAMES=["rgb", "audio"]
MAX_FRAMES=300
RGB_FEAT_SIZE = 1024
AUDIO_FEAT_SIZE = 128

class YouTube8MtfrecordReader:
    
    def __init__(self,
                 num_classes=NUM_CLASSES, feature_size=[RGB_FEAT_SIZE, AUDIO_FEAT_SIZE], 
                 feature_name=FRM_LVL_FEAT_NAMES, max_frames=MAX_FRAMES, sequence_data=True):
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
                                 self.feature_name[1]: tf.FixedLenSequenceFeature([], dtype=tf.string),}
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

    
def tfrecord_reader(filename_queue, data_lvl):#, outpath):
    labelcsv=pd.read_csv(DATA_DIR+"label_names.csv")
    alllabels = labelcsv["label_name"].values
    
    if data_lvl == 'frame':
        print "Features: "+str(FRM_LVL_FEAT_NAMES)
        reader = YouTube8MFrameFeatureReader()
    elif data_lvl == 'video':
        print "Features: "+str(VID_LVL_FEAT_NAMES)
        reader = YouTube8MFrameFeatureReader(feature_name=VID_LVL_FEAT_NAMES, sequence_data=False)
    
    vals = reader.prepare_reader(filename_queue)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #f = codecs.open(outpath, "w", encoding='utf-8')
        try:
            counter=0
            recordlist=[]
            
            while not coord.should_stop():
                counter+=1
                vid, vfeatures, afeatures, labels, numframes = sess.run(vals)
                vfeatureslist=np.round(vfeatures,decimals=5).tolist()
                afeatureslist=np.round(afeatures,decimals=5).tolist()
                labelName=alllabels[labels]
                if data_lvl == 'frame':
                    record={"vid": vid,
                            FRM_LVL_FEAT_NAMES[0]: vfeatureslist,
                            FRM_LVL_FEAT_NAMES[1]: afeatureslist,
                            "numframes": numframes,
                            "labels": labels,
                            "labelName": labelName,
                           }
                elif data_lvl == 'video':
                    record={"vid": vid,
                            VID_LVL_FEAT_NAMES[0]: vfeatureslist,
                            VID_LVL_FEAT_NAMES[1]: afeatureslist,
                            "labels": labels,
                            "labelName": labelName,
                           }
                '''
                if counter==1:
                    f.write("{}\n".format(json.dumps(record)))
                else:
                    fa.write("{}\n".format(json.dumps(record)))
                '''
                recordlist.append(record)
                
        except tf.errors.OutOfRangeError:
            print('\nFinished extracting.')
        finally:
            #json.dump(recordlist, f, separators=(',', ':'))
            coord.request_stop()
            coord.join(threads)
            #f.close()
            print "Loaded."
            
        return recordlist

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
    assert max_quantized_value >  min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias

def data_generator(featurename, features, labels, batch_size):
    print featurename
    if featurename == 'rgb':
        batch_features = np.zeros((batch_size, 1, RGB_FEAT_SIZE))
        print batch_features.shape
    elif featurename == 'audio':
        batch_features = np.zeros((batch_size, 1, AUDIO_FEAT_SIZE))
    batch_labels = np.zeros((batch_size,1))
    
    indexlist = list(range(len(features)))
    for i in range(batch_size):
        index= random.choice(indexlist)
        print index
        batch_features[i,:,:] = np.asarray(features[index])
        batch_labels[i] = labels[index]
        print batch_labels[i]
        indexlist.remove(index)


def get_data(data_path, batch=64, preprocess=None, shuffle=True):
    data_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    return data_datagen.flow_from_directory(data_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=batch,
            shuffle=shuffle)
            
   
