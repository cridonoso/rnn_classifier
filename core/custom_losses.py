import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


@tf.function
def custom_bce(y_true, y_pred, sample_weight=None):
    num_classes = tf.shape(y_pred)[-1]
    num_steps = tf.shape(y_pred)[1]
    y_one = tf.one_hot(y_true, num_classes)
    y_one = tf.expand_dims(y_one, 1)
    y_one =tf.tile(y_one, [1,num_steps,1])
    losses = bce = tf.nn.softmax_cross_entropy_with_logits(y_one, y_pred)
    losses = tf.transpose(losses)
    losses = tf.reduce_sum(losses, 1)
    return tf.reduce_mean(losses)
