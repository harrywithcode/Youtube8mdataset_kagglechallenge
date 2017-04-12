import numpy as np
from keras.utils import np_utils


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
