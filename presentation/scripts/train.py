import tensorflow as tf
import pandas as pd
import datetime
import argparse
import os, sys
import json

from core.metrics import custom_acc
from core.losses import custom_bce
from core.models import get_lstm, get_phased
from core.data import load_records

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop

import os


def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
    # ===============================
    # ======== Load Records =========
    # ===============================
    metadata = pd.read_csv(os.path.join(opt.data, 'test_samples.csv'))
    num_classes = len(metadata['alerceclass'].unique())

    train_batches = load_records(os.path.join(opt.data, 'train'),
                                 opt.batch_size,
                                 max_obs=opt.max_obs,
                                 num_classes=num_classes,
                                 sampling=False,
                                 shuffle=True)

    val_batches   = load_records(os.path.join(opt.data, 'val'),
                                 opt.batch_size,
                                 max_obs=opt.max_obs,
                                 num_classes=num_classes,
                                 sampling=False,
                                 shuffle=False)

    if opt.rnn_type == 'lstm':
        print('[INFO] Using LSTM unit')
        model = get_lstm(opt.units,
                         num_classes,
                         max_obs=opt.max_obs,
                         dropout=opt.dropout)
    if opt.rnn_type == 'phased':
        print('[INFO] Using PHASED LSTM unit')
        model = get_phased(opt.units,
                         num_classes,
                         max_obs=opt.max_obs,
                         dropout=opt.dropout)


    # Compile
    model.compile(optimizer=Adam(opt.lr),
                  loss=custom_bce,
                  metrics=custom_acc)

    estop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=opt.patience,
                          verbose=0,
                          mode='auto',
                          baseline=None,
                          restore_best_weights=True)

    tb = TensorBoard(log_dir=os.path.join(opt.p, 'logs'),
                     write_graph=False,
                     write_images=False,
                     write_steps_per_second=False,
                     update_freq='epoch',
                     profile_batch=0,
                     embeddings_freq=0,
                     embeddings_metadata=None)

    hist = model.fit(train_batches,
                     epochs=opt.epochs,
                     batch_size=opt.batch_size,
                     callbacks=[estop, tb],
                     validation_data=val_batches)

    model.save(os.path.join(opt.p, 'model.h5'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='./astromer/data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--rnn-type', default="lstm", type=str,
                        help='lstm or phased')
    
    parser.add_argument('--gpu', default="0", type=str,
                        help='GPU device number')

    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')
    parser.add_argument('--dropout', default=0.2 , type=float,
                        help='dropout proba.')
    parser.add_argument('--units', default=256, type=int,
                        help='Recurrent unit size')
    parser.add_argument('--patience', default=200, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=2000, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch-size', default=512, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')



    opt = parser.parse_args()
    run(opt)
