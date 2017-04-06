import os
import sys
import cv2
import fnmatch
import numpy as np
from matplotlib import pyplot as plt

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tfrecord_read import get_data
from globals import RGB_FEAT_SIZE, AUDIO_FEAT_SIZE, MAX_FRAMES, NUM_CLASSES, \
    FRM_LVL_FEAT_NAMES, VID_LVL_FEAT_NAMES, GLOBAL_FEAT_NAMES, MODEL_WEIGHTS, \
    VIDEO_TRAIN_DIR, VIDEO_VAL_DIR, VIDEO_TEST_DIR, \
    FRAME_TRAIN_DIR, FRAME_VAL_DIR, FRAME_TEST_DIR, \
    EX_DATA_DIR, LOG_FILE

class yt8mNet_video:
    def __init__(self,
                 feature_type,
                 feature_size,
                 train_data_path=VIDEO_TRAIN_DIR,
                 valid_data_path=VIDEO_VAL_DIR,
                 numclasses=NUM_CLASSES,
                 modelweights=''):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.feature_type = feature_type
        self.feature_size = feature_size
        self.numclasses = numclasses
        self.modelweights = modelweights

    def load_model(self):
        model = Sequential()
        model.add(Dense(2048, input_dim=self.feature_size, kernel_initializer='normal', activation='relu', name='fc1'))
        model.add(Dropout(0.2))
        model.add(Dense(2048, kernel_initializer='normal', activation='relu', name='fc2'))
        model.add(Dropout(0.2))
        model.add(Dense(self.numclasses, activation='softmax', name='predictions'))
        model.load_weights(MODEL_WEIGHTS, by_name=True)

        # rgbfeat_input = Input(tensor=Input(shape=(FEAT_SIZE[0],)))
        # x = rgbfeat_input
        # x = Dense(1024, activation='relu', name='fc1')(x)
        # x = Dense(512, activation='relu', name='fc2')(x)
        # x = Dense(numclasses, activation='softmax', name='predictions')(x)
        # model = Model(input=rgbfeat_input, output=x)
        # model.load_weights(SAVE_WEIGHTS_FILE, by_name=True)

        if self.modelweights:
            model.load_weights(self.modelweights, by_name=True)
            print('Video level model loaded with weights from %s.' % MODEL_WEIGHTS)
        else:
            print "Empty video level model loaded."
        return model

    def train(self, model, saveto_path=''):
        VERBOSITY = 2
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
                  validation_data=(x_valid, y_valid))

        # Save the weights on completion.
        if saveto_path:
            model.save_weights(saveto_path)

class yt8mNet_frame:
    def __init__(self,
                 feature_type,
                 feature_size,
                 train_data_path=FRAME_TRAIN_DIR,
                 valid_data_path=FRAME_VAL_DIR,
                 numclasses=NUM_CLASSES,
                 modelweights=''):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.feature_type = feature_type
        self.feature_size = feature_size
        self.numclasses = numclasses
        self.modelweights = modelweights


if __name__ == '__main__':
    yt8m_vid = yt8mNet_video(VID_LVL_FEAT_NAMES[0], RGB_FEAT_SIZE)
    yt8m_vid_model = yt8m_vid.load_model()
    yt8m_vid.train(yt8m_vid_model, MODEL_WEIGHTS)