import tensorflow as tf

@tf.function
def get_padding_mask(steps, lengths):
    ''' Create mask given a tensor and true length '''
    with tf.name_scope("get_padding_mask") as scope:
        lengths_transposed = tf.expand_dims(lengths, 1, name='Lengths')
        range_row = tf.expand_dims(tf.range(0, steps, 1), 0, name='Indices')
        # Use the logical operations to create a mask
        mask = tf.greater(range_row, lengths_transposed)
        return tf.cast(mask, tf.float32, name='LengthMask')

def create_mask(tensor, lengths):
    ''' Create mask given a tensor and true length '''
    lengths_transposed = tf.expand_dims(lengths, 1)
    rangex = tf.range(0, tf.shape(tensor)[1], 1)
    range_row = tf.expand_dims(rangex, 0)
    # Use the logical operations to create a mask
    mask = tf.less(range_row, lengths_transposed)
    return mask
