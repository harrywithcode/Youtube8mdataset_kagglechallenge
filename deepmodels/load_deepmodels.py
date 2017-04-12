from keras.layers import Input, Dense, Dropout, Merge
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import sys
sys.path.append('/path/to/utils')
from tfrecord_read import get_data
from globals import RGB_FEAT_SIZE, AUDIO_FEAT_SIZE, MAX_FRAMES, NUM_CLASSES, \
    FRM_LVL_FEAT_NAMES, VID_LVL_FEAT_NAMES, GLOBAL_FEAT_NAMES, MODEL_WEIGHTS, MAX_FRAMES, \
    VIDEO_TRAIN_DIR, VIDEO_VAL_DIR, VIDEO_TEST_DIR, \
    FRAME_TRAIN_DIR, FRAME_VAL_DIR, FRAME_TEST_DIR, \
    EX_DATA_DIR, LOG_FILE

class yt8mNet_video:
    def __init__(self,
                 feature_type,
                 feature_size,
                 train_data_path=VIDEO_TRAIN_DIR,
                 valid_data_path=VIDEO_VAL_DIR,
                 numclasses=NUM_CLASSES):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.feature_type = feature_type
        self.feature_size = feature_size
        self.numclasses = numclasses

    def load_model(self, frm_modelweights='', frmdiff_modelweights=''):
        frm_model = Sequential()
        frm_model.add(GRU(4096,
                          return_sequences=True,
                          input_dim=self.feature_size,
                          input_length=MAX_FRAMES,
                          activation='relu',
                          name='fc1'))
        frm_model.add(Dropout(0.3))
        frm_model.add(GRU(4096,
                          return_sequences=False,
                          activation='relu',
                          name='fc2'))
        frm_model.add(Dropout(0.3))
        frm_model.add(Dense(self.numclasses, activation='softmax', name='frm_prediction'))
        if frm_modelweights:
            frm_model.load_weights(frm_modelweights, by_name=True)
            print("Frame model loaded with weights from %s." % frm_modelweights)
        else:
            print "Empty frame model loaded."

        '''
        frmdiff_model = Sequential()
        frmdiff_model.add(GRU(4096, input_dim=self.feature_size, activation='relu', name='fc1'))
        frmdiff_model.add(Dropout(0.3))
        frmdiff_model.add(GRU(4096, activation='relu', name='fc2'))
        frmdiff_model.add(Dropout(0.3))
        frmdiff_model.add(Dense(self.numclasses, activation='softmax', name='frmdiff_feature'))
        
        if frmdiff_modelweights:
            frmdiff_model.load_weights(frmdiff_modelweights, by_name=True)
            print('Frame model loaded with weights from %s.' % frmdiff_modelweights)
        else:
            print "Empty frame model loaded."

        model = Sequential()
        model.add(Merge([frm_model, frmdiff_model], mode='concat'))
        model.add(Dense(self.numclasses, activation='softmax', name='predictions'))
        '''

        return frm_model

    def train(self, model, saveto_path=''):
        x_train, y_train = get_data(self.train_data_path, "train", "video", self.feature_type)
        print('%d training video level samples.' % len(x_train))
        x_valid, y_valid = get_data(self.valid_data_path, "valid", "video", self.feature_type)
        print('%d validation video level samples.' % len(x_valid))

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

    def load_model(self):
        model = Sequential()
        model.add(LSTM(2048,
                       input_shape=(None, self.feature_size),
                       dropout=0.2,
                       recurrent_dropout=0.2,
                       return_sequences=True,
                       name='fc1'))
        model.add(LSTM(4096,
                       dropout=0.2,
                       recurrent_dropout=0.2,
                       return_sequences=True,
                       name='fc2'))
        model.add(Dense(self.numclasses, activation='softmax', name='predictions'))
        model.load_weights(MODEL_WEIGHTS, by_name=True)

        if self.modelweights:
            model.load_weights(self.modelweights, by_name=True)
            print('Frame level model loaded with weights from %s.' % MODEL_WEIGHTS)
        else:
            print "Empty video level model loaded."
        return model

    def train(self, model, saveto_path=''):
        x_train, y_train = get_data(self.train_data_path, "train", "frame", self.feature_type)
        print('%d training frame level samples.' % len(x_train))
        x_valid, y_valid = get_data(self.valid_data_path, "valid", "frame", self.feature_type)
        print('%d validation frame level samples.' % len(x_valid))

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

if __name__ == '__main__':
    yt8m_vid = yt8mNet_video(VID_LVL_FEAT_NAMES[0], RGB_FEAT_SIZE)
    yt8m_vid_model = yt8m_vid.load_model()
    yt8m_vid.train(yt8m_vid_model, MODEL_WEIGHTS)
