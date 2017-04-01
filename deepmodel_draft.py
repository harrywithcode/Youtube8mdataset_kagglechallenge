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

DATA_DIR = "/home/jasonlee/Documents/dmproject_kaggle/data/"
NUM_CLASSES=4716
VID_LVL_FEAT_NAMES=["mean_rgb", "mean_audio"]
FRM_LVL_FEAT_NAMES=["rgb", "audio"]
MAX_FRAMES=300
FEAT_SIZE = [1024, 128]
SAVE_WEIGHTS_FILE = "/home/jasonlee/Documents/dmproject_kaggle/data/"

class yt8mNet_video:
    
    def __init__(self):
        self.data_path = DATA_DIR
        self.featureName = VID_LVL_FEAT_NAMES
        self.featureSize = FEAT_SIZE
        self.numclasses = NUM_CLASSES

    def load_model(self):
        imgfeat_input = Input(tensor=Input(shape=(FEAT_SIZE[0],)))
        
        x = imgfeat_input
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dense(numclasses, activation='softmax', name='predictions')(x)
        model = Model(input=imgfeat_input, output=x)
        model.load_weights(SAVE_WEIGHTS_FILE, by_name=True)
        print('Video level model loaded with weights from %s.' % SAVE_WEIGHTS_FILE)

        return model


    def train(self, model, savepath=''):

        # Center and normalize each sample.
        normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

        # Get streaming data.
        train_generator = get_data(TRAIN_DIR, preprocess=normalize)
        valid_generator = get_data(VALID_DIR, preprocess=normalize)

        print('%d training samples.' % train_generator.n)
        print('%d validation samples.' % valid_generator.n)

        sgd = SGD(lr=0.01,
                  decay=1e-6,
                  momentum=0.9,
                  nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        callbacks = list()

        callbacks.append(CSVLogger(LOG_FILE))
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001))

        if save_to:
            callbacks.append(ModelCheckpoint(filepath=MODEL_WEIGHTS, verbose=1))

        model.fit_generator(generator=train_generator,
                            samples_per_epoch=train_generator.n,
                            nb_epoch=5,
                            validation_data=valid_generator,
                            nb_val_samples=1000,a
                            callbacks=callbacks,
                            verbose=VERBOSITY)

        # Save the weights on completion.
        if save_to:
            model.save_weights(savepath)
