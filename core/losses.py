import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


@tf.function
def custom_bce(y_true, y_pred, sample_weight=None):
    num_steps = tf.shape(y_pred)[1]
    y_true = tf.expand_dims(y_true, 1)
    y_true = tf.tile(y_true, [1, num_steps, 1])

    losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

    return tf.reduce_mean(losses)
