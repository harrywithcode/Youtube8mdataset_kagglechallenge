import os
import sys
import cv2
import fnmatch
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

TRAIN_DIR = "/your/path/to/train/data/"
VALID_DIR = "/your/path/to/valid/data/"
SAVE_WEIGHTS_FILE="/your/path/to/save/weights/file/"
NUM_CLASSES=4716
GLOBAL_FEAT_NAMES=["vid", "labels", "labelName"]
VID_LVL_FEAT_NAMES=["mean_rgb", "mean_audio"]
FRM_LVL_FEAT_NAMES=["rgb", "audio", "numframes"]
MAX_FRAMES=300
RGB_FEAT_SIZE = 1024
AUDIO_FEAT_SIZE = 128

class yt8mNet_video:
    def __init__(self,
                 train_data_path,
                 valid_data_path,
                 feature_type,
                 feature_size,
                 numclasses):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.feature_type = feature_type
        self.feature_size = feature_size
        self.numclasses = numclasses

    def load_model(self):
        model = Sequential()
        model.add(Dense(2048, input_dim=self.feature_size, kernel_initializer='normal', activation='relu', name='fc1'))
        model.add(Dropout(0.2))
        model.add(Dense(2048, kernel_initializer='normal', activation='relu', name='fc2'))
        model.add(Dropout(0.2))
        model.add(Dense(self.numclasses, activation='softmax', name='predictions'))
        model.load_weights(SAVE_WEIGHTS_FILE, by_name=True)
        print('Video level model loaded with weights from %s.' % SAVE_WEIGHTS_FILE)
        return model

    def train(self, model, saveto_path=''):
        x_train, y_train = get_data(self.train_data_path, "video", self.feature_type)
        x_valid, y_valid = get_data(self.valid_data_path, "video", self.feature_type)
        print('%d training samples.' % len(x_train))
        print('%d validation samples.' % len(x_valid))

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

        if saveto_path:
            callbacks.append(ModelCheckpoint(filepath=MODEL_WEIGHTS, verbose=1))
        model.fit(x_train,
                  y_train,
                  epochs=5,
                  callbacks=callbacks,
                  validation_data=(x_valid, yvalid))

        # Save the weights on completion.
        if saveto_path:
            model.save_weights(saveto_path)

if __name__ == '__main__':
    yt8m_vid = yt8mNet_video(TRAIN_DIR, 
                             VALID_DIR, 
                             FRM_LVL_FEAT_NAMES[0],
                             RGB_FEAT_SIZE,
                             NUM_CLASSES)
    yt8m_vid_model = yt8m_vid.load_model()
    yt8m_vid.train(yt8m_vid_model, SAVE_WEIGHTS_FILE)
  
