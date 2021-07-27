import tensorflow as tf
import unittest

from tensorflow.keras.layers import LayerNormalization
from core.models import get_lstm, get_phased
from core.data import load_records

class TestStringMethods(unittest.TestCase):

    def test_model_output(self):

        x = tf.random.normal([1, 10, 3])
        m = tf.convert_to_tensor([[0.,0.,1.,1.,0.,0.,0.,0.,0.,0.]])
        m = tf.expand_dims(m, 2)

        input_dict = {'values':x, 'mask':m}
        model = get_lstm(16, 4, max_obs=10, dropout=0.5)

        y = model(input_dict)

        self.assertEqual(y.shape[-1], 4, 'number of classes does not match')


    def test_model_output(self):

        x = tf.random.normal([1, 10, 3])
        m = tf.convert_to_tensor([[0.,0.,1.,1.,0.,0.,0.,0.,0.,0.]])
        m = tf.expand_dims(m, 2)

        input_dict = {'values':x, 'mask':m}
        model = get_lstm(16, 4, max_obs=10, dropout=0.5)

        first_layer = model.get_layer('RNN_0')

        output = first_layer(x, mask=tf.cast(m, tf.bool))
        output = LayerNormalization(axis=1)(output)
        print(output)
        # self.assertEqual(y.shape[-1], 4, 'number of classes does not match')

if __name__ == '__main__':
    unittest.main()
