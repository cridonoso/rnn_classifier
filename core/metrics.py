import tensorflow as tf

from tensorflow.keras.metrics import categorical_accuracy

@tf.function
def custom_acc(y_true, y_pred):    
    y_pred  = tf.nn.softmax(y_pred)[:,-1,:-1]
    acc = categorical_accuracy(y_true, y_pred)
    return tf.reduce_mean(acc)
