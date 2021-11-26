import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import categorical_crossentropy as cc_obj

@tf.function
def custom_bce(y_true, y_pred, sample_weight=None):
    true = tf.expand_dims(y_true, 1)
    true = tf.tile(true, [1, 200, 1])   
    
    pred = y_pred[...,:-1]
    mask = y_pred[...,-1]
    
    cce = cc_obj(true, pred, from_logits=True)

    cce = cce*mask
    cce = tf.reduce_sum(cce, 1)
    
    cce = tf.math.divide_no_nan(cce, tf.reduce_sum(mask, 1))

    cce = tf.reduce_mean(cce)
    return cce
