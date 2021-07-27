import tensorflow as tf
import os, sys

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, LayerNormalization, RNN
from core.phased import PhasedLSTM

def get_lstm(units, num_classes, max_obs=200, dropout=0.5):
    values = Input(shape=(max_obs, 3), name='input')
    maskbol   = Input(shape=(max_obs), dtype=tf.float32, name='mask')

    inputs = {'values': values, 'mask': maskbol}

    x = LSTM(units,
             return_sequences=True,
             dropout=dropout,
             name='RNN_0')(inputs['values'], mask=tf.cast(inputs['mask'], dtype=tf.bool))
    x = LayerNormalization(axis=1)(x)
    x = LSTM(units,
             return_sequences=True,
             dropout=dropout,
             name='RNN_1')(x, mask=tf.cast(inputs['mask'], dtype=tf.bool))
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN')(x)
    return Model(inputs=inputs, outputs=x, name="LSTMCLF")


def get_phased(units, num_classes, max_obs=200, dropout=0.5):
    values = Input(shape=(max_obs, 3), name='input')
    maskbol   = Input(shape=(max_obs), dtype=tf.float32, name='mask')

    inputs = {'values': values, 'mask': maskbol}

    x = RNN(PhasedLSTM(units, dropout=dropout),
             name='RNN_0',
             return_sequences=True)(inputs['values'], mask=inputs['mask'])
    x = LayerNormalization(axis=1)(x)
    x = RNN(PhasedLSTM(units, dropout=dropout),
             name='RNN_1',
             return_sequences=True)(x, mask=inputs['mask'])
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN')(x)
    return Model(inputs=inputs, outputs=x, name="PhasedCLF")
