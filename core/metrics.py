import tensorflow as tf

from tensorflow.keras.metrics import categorical_accuracy

@tf.function
def custom_acc(y_true, y_pred):
    y_pred  = tf.nn.softmax(y_pred)[:,-1,:]


    tf.print(y_pred)
    tf.print(y_true)
    acc = categorical_accuracy(y_true, y_pred)

    tf.print(acc)

    return tf.reduce_mean(acc)
