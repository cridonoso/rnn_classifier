import tensorflow as tf
import os, sys

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, LayerNormalization, RNN
from core.phased import PhasedLSTM

def get_lstm(units, num_classes, max_obs=200, dropout=0.5):
    values  = Input(shape=(max_obs, 4), name='input')
    mask  = Input(shape=(max_obs, ), name='mask')

    inputs = {'input': values, 'mask': mask}

    x = LSTM(units,
             return_sequences=True,
             dropout=dropout,
             name='RNN_0')(inputs['input'], mask=tf.cast(inputs['mask'], dtype=tf.bool))
    x = LayerNormalization()(x)
    x = LSTM(units,
             return_sequences=True,
             dropout=dropout,
             name='RNN_1')(x, mask=tf.cast(inputs['mask'], dtype=tf.bool))
    x = LayerNormalization()(x)
    x = Dense(num_classes, name='FCN')(x)
    x = tf.concat([x, tf.expand_dims(inputs['mask'], 2)], 2)
    return Model(inputs=inputs, outputs=x, name="LSTMCLF")


def get_phased(units, num_classes, max_obs=200, dropout=0.5):
    values  = Input(shape=(max_obs, 4), name='input')
    mask  = Input(shape=(max_obs, ), name='mask')

    inputs = {'input': values, 'mask': mask}

    x = RNN(PhasedLSTM(units, dropout=dropout),
             name='RNN_0',
             return_sequences=True)(inputs['input'], mask=tf.cast(inputs['mask'], dtype=tf.bool))
    x = LayerNormalization()(x)
    x = RNN(PhasedLSTM(units, dropout=dropout),
             name='RNN_1',
             return_sequences=True)(x, mask=tf.cast(inputs['mask'], dtype=tf.bool))
    x = LayerNormalization()(x)
    x = Dense(num_classes, name='FCN')(x)
    x = tf.concat([x, tf.expand_dims(inputs['mask'], 2)], 2)
    return Model(inputs=inputs, outputs=x, name="PhasedCLF")
