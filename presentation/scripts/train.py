import tensorflow as tf
import datetime
import argparse
import os, sys
import json

from core.custom_metrics import custom_acc
from core.custom_losses import custom_bce
from core.tboard  import save_scalar, draw_graph
from core.models import get_lstm_attention, get_lstm_no_attention, get_fc_attention
from core.data import load_records
from time import gmtime, strftime
from tqdm import tqdm

@tf.function
def train_step(model, batch, opt):
    with tf.GradientTape() as tape:
        y_pred = model(batch)
        ce = custom_bce(y_true=batch['label'], y_pred=y_pred)
        acc = custom_acc(batch['label'], y_pred)
    grads = tape.gradient(ce, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return acc, ce

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
    # Read Data
    num_classes = tf.reduce_sum([1 for _ in os.listdir(
                                 os.path.join(opt.data, 'train'))])
    train_batches = load_records(os.path.join(opt.data, 'train'),
                                 batch_size=opt.batch_size,
                                 max_obs=opt.max_obs,
                                 repeat=opt.repeat)
    valid_batches = load_records(os.path.join(opt.data, 'val'),
                                 batch_size=opt.batch_size,
                                 max_obs=opt.max_obs,
                                 repeat=opt.repeat)

    inp_dim = [t['values'].shape[-1] for t in train_batches][0]
    # Instance the model
    if opt.mode == 0:
        model = get_lstm_attention(units=opt.units,
                                   num_classes=num_classes,
                                   max_obs=opt.max_obs,
                                   inp_dim=inp_dim,
                                   dropout=opt.dropout)
    if opt.mode == 1:
        model = get_lstm_no_attention(units=opt.units,
                                      num_classes=num_classes,
                                      max_obs=opt.max_obs,
                                      inp_dim=inp_dim,
                                      dropout=opt.dropout)

    if opt.mode == 2:
        model = get_fc_attention(num_classes=num_classes,
                                   max_obs=opt.max_obs,
                                   inp_dim=inp_dim,
                                   dropout=opt.dropout)

    # Tensorboard
    train_writter = tf.summary.create_file_writer(
                                    os.path.join(opt.p, 'logs', 'train'))
    valid_writter = tf.summary.create_file_writer(
                                    os.path.join(opt.p, 'logs', 'valid'))

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(opt.lr)
    # To save metrics
    train_bce  = tf.keras.metrics.Mean(name='train_bce')
    valid_bce  = tf.keras.metrics.Mean(name='valid_bce')
    train_acc  = tf.keras.metrics.Mean(name='train_acc')
    valid_acc  = tf.keras.metrics.Mean(name='valid_acc')

    # Save Hyperparameters
    conf_file = os.path.join(opt.p, 'conf.json')
    varsdic = vars(opt)
    varsdic['num_classes'] = int(num_classes.numpy())
    varsdic['exp_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    with open(conf_file, 'w') as json_file:
        json.dump(varsdic, json_file, indent=4)

    # ==============================
    # ======= Training Loop ========
    # ==============================
    best_loss = 999999.
    es_count = 0
    for epoch in range(opt.epochs):
        for train_batch in tqdm(train_batches, desc='train'):
            acc, bce = train_step(model, train_batch, optimizer)
            train_acc.update_state(acc)
            train_bce.update_state(bce)

        for valid_batch in tqdm(valid_batches, desc='validation'):
            acc, bce = valid_step(model, valid_batch)
            valid_acc.update_state(acc)
            valid_bce.update_state(bce)

        save_scalar(train_writter, train_acc, epoch, name='accuracy')
        save_scalar(valid_writter, valid_acc, epoch, name='accuracy')
        save_scalar(train_writter, train_bce, epoch, name='xentropy')
        save_scalar(valid_writter, valid_bce, epoch, name='xentropy')

        if opt.verbose == 0:
            print('EPOCH {} - ES COUNT: {}'.format(epoch, es_count))
            print('train acc: {:.2f} - train ce: {:.2f}'.format(train_acc.result(),
                                                                train_bce.result()))
            print('val acc: {:.2f} - val ce: {:.2f}'.format(valid_acc.result(),
                                                            valid_bce.result(),
                                                            ))
        if valid_bce.result() < best_loss:
            best_loss = valid_bce.result()
            es_count = 0.
            model.save_weights(os.path.join(opt.p, 'weights'))
        else:
            es_count+=1.
        if es_count == opt.patience:
            print('[INFO] Early Stopping Triggered')
            break

        valid_bce.reset_states()
        train_bce.reset_states()
        train_acc.reset_states()
        valid_acc.reset_states()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='./astromer/data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')

    parser.add_argument('--max-obs', default=51, type=int,
                    help='Max number of observations')
    parser.add_argument('--dropout', default=0.5 , type=float,
                        help='dropout proba.')
    parser.add_argument('--units', default=128, type=int,
                        help='Recurrent unit size')
    parser.add_argument('--patience', default=200, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=2000, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--repeat', default=1, type=int,
                        help='number of times to repeat the training and validation dataset')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')

    parser.add_argument('--mode', default=0, type=int,
                        help='0: RNN + Attention - 1: RNN Only')
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbose to show scores during training')

    opt = parser.parse_args()
    run(opt)
