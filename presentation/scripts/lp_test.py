import tensorflow as tf
import datetime
import argparse
import warnings
import os, sys
import h5py
import json

from core.metrics import custom_acc
from core.losses import custom_bce
from core.tboard  import save_scalar, draw_graph
from core.models import get_lstm, get_phased
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from core.data import load_records
from time import gmtime, strftime
from tqdm import tqdm


@tf.function
def valid_step(model, batch, return_pred=False):
    with tf.GradientTape() as tape:
        y_pred = model(batch, training=False)
        ce = custom_bce(y_true=batch['label'],
                         y_pred=y_pred)
        acc = custom_acc(batch['label'], y_pred)
    if return_pred:
        return acc, ce, y_pred, batch['label']
    return acc, ce

def run(opt):

    # Load Hyperparameters for LSTM
    conf_file = os.path.join(opt.lstm, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf_lstm = json.load(handle)

    # Load Hyperparameters for PHASED LSTM
    conf_file = os.path.join(opt.phased, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf_phased = json.load(handle)


    # Read Data
    test_batches = load_records(os.path.join(opt.data, 'test'),
                                batch_size=opt.batch_size,
                                take=opt.take,
                                max_obs=conf_lstm['max_obs'])


    # Instance the model
    lstm_model = get_lstm(units=conf_lstm['units'],
                          num_classes=conf_lstm['num_classes'],
                          max_obs=conf_lstm['max_obs'],
                          dropout=conf_lstm['dropout'])


    phased_model = get_phased(units=conf_phased['units'],
                              num_classes=conf_phased['num_classes'],
                              max_obs=conf_phased['max_obs'],
                              dropout=conf_phased['dropout'])


    lstm_model.load_weights('{}/weights'.format(opt.lstm)).expect_partial()
    phased_model.load_weights('{}/weights'.format(opt.phased)).expect_partial()


    predictions = []
    true_labels = []
    for batch in tqdm(test_batches, desc='test'):
        _, _, y_pred_lstm, y_true = valid_step(lstm_model, batch, return_pred=True)
        _, _, y_pred_phased, _ = valid_step(phased_model, batch, return_pred=True)
        y_pred = (y_pred_lstm + y_pred_phased)/2
        predictions.append(y_pred[:, -1, :])
        true_labels.append(y_true)

    y_pred = tf.concat(predictions, 0)
    y_true = tf.concat(true_labels, 0)

    pred_labels = tf.argmax(y_pred, 1)
    y_true = tf.argmax(y_true, 1)

    precision, \
    recall, \
    f1, _ = precision_recall_fscore_support(y_true,
                                            pred_labels,
                                            average='macro')
    acc = accuracy_score(y_true, pred_labels)
    results = {'f1': f1,
               'recall': recall,
               'precision': precision,
               'accuracy':acc}

    os.makedirs(os.path.join(opt.p, 'test'), exist_ok=True)
    results_file = os.path.join(opt.p, 'test', 'test_results.json')

    with open(results_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    h5f = h5py.File(os.path.join(opt.p, 'test', 'predictions.h5'), 'w')
    h5f.create_dataset('y_pred', data=y_pred.numpy())
    h5f.create_dataset('y_true', data=y_true.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='./astromer/data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path for L+P predictions.')
    parser.add_argument('--lstm', default="./runs/debug", type=str,
                        help='Proyect path for LSTM.')
    parser.add_argument('--phased', default="./runs/debug", type=str,
                        help='Proyect path for PLSTM.')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--take', default=1, type=int,
                        help='batches for testing')

    opt = parser.parse_args()
    run(opt)
