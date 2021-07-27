import tensorflow as tf
import unittest

from core.data import load_records

class TestStringMethods(unittest.TestCase):

    def test_input_standard(self):

        dataset = load_records('./data/records/ztf/test', batch_size=20, max_obs=200, repeat=1)
        for batch in dataset:
            lc = tf.boolean_mask(batch['values'][0], batch['mask'][0][...,0])
            mean = tf.reduce_mean(lc, 0)
            break

        self.assertTrue(tf.reduce_mean(mean) < 0.1, 'Input vector was not standardized')

    def test_nan_values(self):

        dataset = load_records('./data/records/ztf/test', batch_size=256, max_obs=200, repeat=1)
        there_is_nan = False
        for batch in dataset:
            response = tf.math.is_nan(batch['values'])
            self.assertFalse(tf.math.reduce_any(response), 'NaN detected on the input')


if __name__ == '__main__':
    unittest.main()
