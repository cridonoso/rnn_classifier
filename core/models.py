import tensorflow as tf
import os, sys

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, LayerNormalization
from core.mask import create_mask

from astropackage.embeddings import ASTROMER_EMBEDDING

def get_lstm_attention(units, num_classes, max_obs=200, inp_dim=108, dropout=0.5):
    values = Input(shape=(max_obs, inp_dim), name='input')
    lengths   = Input(shape=(), dtype=tf.int32, name='mask')
    inputs = {'values': values, 'length': lengths}
    mask = create_mask(inputs['values'], inputs['length'])

    times = tf.slice(values, [0,0,0],[-1,-1,1])
    magns = tf.slice(values, [0,0,1],[-1,-1,1])
    x_cls, x = ASTROMER_EMBEDDING()(magns, times, inputs['length'])

    rnn_0 = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_0')
    rnn_1 = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_1')

    state = Dense(units, name='IniState', activation='tanh')(x_cls)
    state = tf.reshape(state, [-1, units])

    x = rnn_0(x, mask=mask, initial_state=[state, state])
    x = LayerNormalization(axis=1)(x)
    x = rnn_1(x, mask=mask)
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN')(x)
    return Model(inputs=inputs, outputs=x, name="RNNCLF")

def get_lstm_no_attention(units, num_classes, max_obs=200, inp_dim=108, dropout=0.5):
    values = Input(shape=(max_obs, 3), name='input')
    lengths   = Input(shape=(), dtype=tf.int32, name='mask')
    inputs = {'values': values, 'length': lengths}
    mask = create_mask(inputs['values'], inputs['length'])
    x = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_0')(inputs['values'], mask=mask)
    x = LayerNormalization(axis=1)(x)
    x = LSTM(units, return_sequences=True, dropout=dropout, name='RNN_1')(x, mask=mask)
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN')(x)
    return Model(inputs=inputs, outputs=x, name="RNNCLF")
