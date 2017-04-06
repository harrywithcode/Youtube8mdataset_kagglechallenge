import os
import time
import numpy as np

VIDEO_TRAIN_DIR = "/home/jasonlee/Documents/dmproject_kaggle/data/yt8m_video_level/train/"
VIDEO_VAL_DIR = "/home/jasonlee/Documents/dmproject_kaggle/data/yt8m_video_level/valid/"
VIDEO_TEST_DIR = "/home/jasonlee/Documents/dmproject_kaggle/data/yt8m_video_level/test/"

FRAME_TRAIN_DIR = "/home/jasonlee/Documents/dmproject_kaggle/data/yt8m_frame_level/train/"
FRAME_VAL_DIR = "/home/jasonlee/Documents/dmproject_kaggle/data/yt8m_frame_level/valid/"
FRAME_TEST_DIR = "/home/jasonlee/Documents/dmproject_kaggle/data/yt8m_frame_level/test/"

GLOBAL_FEAT_NAMES=["vid", "labels", "labelName"]
VID_LVL_FEAT_NAMES=["mean_rgb", "mean_audio"]
FRM_LVL_FEAT_NAMES=["rgb", "audio", "numframes"]

NUM_CLASSES=4716
BATCH_SIZE=64
MAX_FRAMES=300
RGB_FEAT_SIZE = 1024
AUDIO_FEAT_SIZE = 128

EX_DATA_DIR = "/home/jasonlee/Documents/dmproject_kaggle/data/yt8m_video_level/"

PACKAGE_PATH = os.path.dirname(__file__)
MODEL_WEIGHTS = os.path.join(PACKAGE_PATH, "packages/model_weights.h5")
LOG_FILE = os.path.join(PACKAGE_PATH, "logs", "training_%s.log" % time.strftime("%m_%d_%H_%M"))