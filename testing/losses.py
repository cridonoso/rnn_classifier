import tensorflow as tf
import unittest

from core.data import load_records
from core.losses import custom_bce
from core.models import get_lstm
from core.metrics import custom_acc

class TestStringMethods(unittest.TestCase):

    def test_xentropy(self):

        dataset, num_classes = load_records('./data/records/ztf/test',
                                            batch_size=128,
                                            max_obs=200,
                                            take=1,
                                            return_num_classes=True)

        model = get_lstm(16, num_classes, max_obs=200, dropout=0.5)

        for batch in dataset:
            y_pred = model(batch)

            bce = custom_bce(batch['label'], y_pred)
            acc = custom_acc(batch['label'], y_pred)
            print(acc)
        pass

if __name__ == '__main__':
    unittest.main()
