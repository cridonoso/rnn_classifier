import tensorflow as tf
import os, sys
sys.path.append(os.environ['ROOTDIR'])

from core.tboard  import save_scalar, draw_graph
from core.custom_metrics import custom_acc
from core.custom_losses import custom_bce
from data.record_utils import load_records
from core.models import get_lstm
from tqdm import tqdm

exp_path = './runs/model_1'
datadir  = './astromer/data/records/macho'
epochs   = 3000
verbose  = 0
patience = 1000
units    = 128
dropout  = 0.5
learning_rate = 1e-3


# Read Data
num_classes = tf.reduce_sum([1 for _ in os.listdir(
                             os.path.join(datadir, 'train'))])
train_batches = load_records(os.path.join(datadir, 'train'),
                             batch_size=16)
valid_batches = load_records(os.path.join(datadir, 'val'),
                             batch_size=16)

# Instance the model
model = get_lstm(units=units, num_classes=num_classes, dropout=dropout)

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

optimizer = tf.optimizers.Adam()
# Tensorboard
train_writter = tf.summary.create_file_writer(
                                os.path.join(exp_path, 'logs', 'train'))
valid_writter = tf.summary.create_file_writer(
                                os.path.join(exp_path, 'logs', 'valid'))

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
# To save metrics
train_bce  = tf.keras.metrics.Mean(name='train_bce')
valid_bce  = tf.keras.metrics.Mean(name='valid_bce')
train_acc  = tf.keras.metrics.Mean(name='train_acc')
valid_acc  = tf.keras.metrics.Mean(name='valid_acc')

# Training Loop
best_loss = 999999.
es_count = 0
for epoch in range(epochs):
    for step, train_batch in tqdm(enumerate(train_batches), desc='train'):
        print(train_batch['values'].shape)
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

    if verbose == 0:
        print('EPOCH {} - ES COUNT: {}'.format(epoch, es_count))
        print('train acc: {:.2f} - train ce: {:.2f}'.format(train_acc.result(),
                                                            train_bce.result()))
        print('val acc: {:.2f} - val ce: {:.2f}'.format(valid_acc.result(),
                                                        valid_bce.result(),
                                                        ))
    if valid_bce.result() < best_loss:
        best_loss = valid_bce.result()
        es_count = 0.
        model.save_weights(exp_path+'/weights')
    else:
        es_count+=1.
    if es_count == patience:
        print('[INFO] Early Stopping Triggered')
        break

    valid_bce.reset_states()
    train_bce.reset_states()
    train_acc.reset_states()
    valid_acc.reset_states()
