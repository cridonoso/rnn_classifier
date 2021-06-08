import tensorflow as tf
import os, sys

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from core.mask import create_mask


def get_lstm_attention(units, num_classes, max_obs=200, inp_dim=108, dropout=0.5):
    values = Input(shape=(max_obs, inp_dim), name='input')
    lengths   = Input(shape=(), dtype=tf.int32, name='mask')
    inputs = {'values': values, 'length': lengths}
    mask = create_mask(inputs['values'], inputs['length'])
    x = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_0')(inputs['values'], mask=mask)
    x = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_1')(x, mask=mask)
    x = Dense(num_classes, name='FCN')(x)
    return Model(inputs=inputs, outputs=x, name="RNNCLF")

def get_lstm_no_attention(units, num_classes, dropout=0.5, max_obs=200):
    values = Input(shape=(max_obs, 3), name='input')
    lengths   = Input(shape=(), dtype=tf.int32, name='mask')
    inputs = {'values': values, 'length': lengths}
    mask = create_mask(inputs['values'], inputs['length'])
    x = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_0')(inputs['values'], mask=mask)
    x = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_1')(x, mask=mask)
    x = Dense(num_classes, name='FCN')(x)
    return Model(inputs=inputs, outputs=x, name="RNNCLF")
