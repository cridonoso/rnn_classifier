import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

from joblib import Parallel, delayed
from tqdm import tqdm
from time import time

from joblib import wrap_non_picklable_objects

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def normalice(tensor, axis=0):
    min_value = tf.reduce_min(tensor, axis, name='min_value')
    max_value = tf.reduce_max(tensor, axis, name='max_value')
    if axis == 1:
        min_value = tf.expand_dims(min_value, axis)
        max_value = tf.expand_dims(max_value, axis)
    normed = (tensor - min_value)/(max_value-min_value)
    return normed

def get_example(lcid, label, lightcurve):
    """
    Create a record example from numpy values.
    Args:
        lcid (string): object id
        label (int): class code
        lightcurve (numpy array): time, magnitudes and observational error
    Returns:
        tensorflow record
    """

    f = dict()

    dict_features={
    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(lcid).encode()])),
    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[lightcurve.shape[0]])),
    }
    element_context = tf.train.Features(feature = dict_features)

    dict_sequence = {}
    for col in range(lightcurve.shape[1]):
        seqfeat = _float_feature(lightcurve[:, col])
        seqfeat = tf.train.FeatureList(feature = [seqfeat])
        dict_sequence['dim_{}'.format(col)] = seqfeat

    element_lists = tf.train.FeatureLists(feature_list=dict_sequence)
    ex = tf.train.SequenceExample(context = element_context,
                                  feature_lists= element_lists)
    return ex

@wrap_non_picklable_objects
def process_lc2(lc_obs, oid, unique_labels):
    label = unique_labels.index(lc_obs['alerceclass'].iloc[0])
    lc_obs = lc_obs[['mjd', 'forcediffimflux', 'forcediffimfluxunc', 'forcediffimsnr']]
    lc_obs.columns = ['mjd', 'mag', 'errmag', 'snr']
    lc_obs = lc_obs.dropna()
    lc_obs = lc_obs.sort_values('mjd')
    lc_obs = lc_obs.drop_duplicates(keep='last')

    numpy_lc = lc_obs.values

    return oid, label, numpy_lc

def process_lc3(lc_index, label, numpy_lc, writer):
    try:
        ex = get_example(lc_index, label, numpy_lc)
        writer.write(ex.SerializeToString())
    except:
        print('[INFO] {} could not be processed'.format(lc_index))

def create_dataset(meta_df,
                   observations,
                   target='data/records/ztf/',
                   record_name='chunk',
                   n_jobs=None,
                   unique_labels = [],
                   band=1):
    os.makedirs(target, exist_ok=True)

    # Separate by class
    observations = observations[observations['oid'].isin(meta_df['oid'])]
    observations = pd.merge(observations, meta_df[['oid', 'alerceclass']], on='oid')
    lightcurves  = observations.groupby('oid')

    print('[INFO] Preprocessing lighcurves...')
    var = Parallel(n_jobs=n_jobs)(delayed(process_lc2)(lc, oid, unique_labels) \
                                for oid, lc in lightcurves)

    with tf.io.TFRecordWriter('{}/{}.record'.format(target, record_name)) as writer:
        for data_lc in tqdm(var):
            process_lc3(*data_lc, writer)

# ==============================
# ====== LOADING FUNCTIONS =====
# ==============================
def adjust_fn(func, *arguments):
    def wrap(*args, **kwargs):
        result = func(*args, *arguments)
        return result
    return wrap

def deserialize(sample, inp_dim=4):
    """
    Read a serialized sample and convert it to tensor
    Context and sequence features should match with the name used when writing.
    Args:
        sample (binary): serialized sample
    Returns:
        type: decoded sample
    """
    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'length': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = dict()
    for i in range(inp_dim):
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float32)

    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )

    input_dict = dict()
    input_dict['lcid']   = tf.cast(context['id'], tf.string)
    input_dict['length'] = tf.cast(context['length'], tf.int32)
    input_dict['label']  = tf.cast(context['label'], tf.int32)

    casted_inp_parameters = []
    for i in range(inp_dim):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)

    sequence = tf.stack(casted_inp_parameters, axis=2)[0]
    input_dict['input'] = sequence
    return input_dict

def get_window(sequence, length, pivot, max_obs):
    pivot = tf.minimum(length-max_obs, pivot)
    pivot = tf.maximum(0, pivot)
    end = tf.minimum(length, max_obs)

    sliced = tf.slice(sequence, [pivot, 0], [end, -1])
    return sliced

def get_windows(sample, max_obs):
    input_dict = deserialize(sample)

    sequence = input_dict['input']
    rest = input_dict['length']%max_obs

    pivots = tf.tile([max_obs], [tf.cast(input_dict['length']/max_obs, tf.int32)])
    pivots = tf.concat([[0], pivots], 0)
    pivots = tf.math.cumsum(pivots)

    splits = tf.map_fn(lambda x: get_window(sequence,
                                            input_dict['length'],
                                            x,
                                            max_obs),  pivots,
                       infer_shape=False,
                       fn_output_signature=(tf.float32))

    # aqui falta retornar labels y oids
    y = tf.tile([input_dict['label']], [len(splits)])
    ids = tf.tile([input_dict['lcid']], [len(splits)])

    return splits, y, ids

def format_lc(x, y, i, num_classes, max_obs):
    x = normalice(x)
    time_steps = tf.shape(x)[0]

    mask = tf.ones([time_steps])
    if time_steps < max_obs:
        mask_fill = tf.zeros([max_obs - time_steps], dtype=tf.float32)
        mask  = tf.concat([mask, mask_fill], 0)

    input_dict = dict()
    input_dict['input'] = x
    input_dict['mask']  = mask
    input_dict['id']    = i

    return input_dict, tf.one_hot(y, num_classes)


def load_records(source, batch_size, max_obs=100, num_classes=2, sampling=False, shuffle=False):
    """
    Pretraining data loader.
    This method build the ASTROMER input format.
    ASTROMER format is based on the BERT masking strategy.
    Args:
        source (string): Record folder
        batch_size (int): Batch size
        no_shuffle (bool): Do not shuffle training and validation dataset
        max_obs (int): Max. number of observation per serie
        msk_frac (float): fraction of values to be predicted ([MASK])
        rnd_frac (float): fraction of [MASKED] values to replace with random values
        same_frac (float): fraction of [MASKED] values to replace with true values
    Returns:
        Tensorflow Dataset: Iterator withg preprocessed batches
    """
    rec_paths = [os.path.join(source, x) for x in os.listdir(source)]

    if sampling:
        fn_0 = adjust_fn(sample_lc, max_obs)
    else:
        fn_0 = adjust_fn(get_windows, max_obs)

    fn_1 = adjust_fn(format_lc, num_classes, max_obs)

    dataset = tf.data.TFRecordDataset(rec_paths)
    if shuffle:
        dataset = dataset.shuffle(10000)

    dataset = dataset.map(fn_0)
    if not sampling:
        dataset = dataset.flat_map(lambda x,y,i: tf.data.Dataset.from_tensor_slices((x,y,i)))

    dataset = dataset.map(fn_1)
    padded_shapes = ({'input': (None, 4), 'mask': (None, ), 'id': ()},(num_classes))
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes).cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
