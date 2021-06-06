import tensorflow as tf
import os, sys

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense


def get_lstm(units, num_classes, dropout=0.5):
    values = Input(shape=(200, 107), name='input')
    mask   = Input(shape=(200, 1), dtype=tf.bool, name='mask')
    inputs = {'values': values, 'mask': mask}
    x = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_0')(inputs['values'])
    x = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_1')(x)
    x = Dense(num_classes, name='FCN')(x)
    return Model(inputs=inputs, outputs=x, name="RNNCLF")
