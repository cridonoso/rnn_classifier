import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from core.masking import get_padding_mask
from joblib import Parallel, delayed
from tqdm import tqdm

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def standardize(tensor, axis=0):
    mean_value = tf.reduce_mean(tensor, axis, name='mean_value')
    if axis == 1:
        mean_value = tf.expand_dims(mean_value, axis)
    normed = tensor - mean_value
    return normed

def divide_training_subset(frame, train, val):
    frame = frame.sample(frac=1)
    n_samples = frame.shape[0]
    n_train = int(n_samples*train)
    n_val = int(n_samples*val//2)

    sub_train = frame.iloc[:n_train]
    sub_val   = frame.iloc[n_train:n_train+n_val]
    sub_test  = frame.iloc[n_train+n_val:]

    return ('train', sub_train), ('val', sub_val), ('test', sub_test)

def get_example(lcid, label, lightcurve):
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

def process_lc(observations, oid, label, band, writer):
    observations = observations[observations['fid'] == band]
    observations = observations[['mjd', 'magpsf_corr', 'sigmapsf_corr_ext']]
    observations = observations.dropna()
    observations = observations[observations['magpsf_corr']<25]
    observations = observations.sort_values('mjd')
    observations = observations.drop_duplicates(keep='last')
    if observations.shape[0] > 5:
        numpy_lc = observations.values
        ex = get_example(oid, label, numpy_lc)
        writer.write(ex.SerializeToString())

def write_records(frame, dest, max_lcs_per_record, detections, ylabel, band=1, n_jobs=None):
    n_jobs = mp.cpu_count() if n_jobs is not None else n_jobs
    # Get frames with fixed number of lightcurves
    collection = [frame.iloc[i:i+max_lcs_per_record] \
                  for i in range(0, frame.shape[0], max_lcs_per_record)]
    # Iterate over subset
    for counter, subframe in enumerate(collection):
        partial_det = detections[detections['oid'].isin(subframe['oid'])]
        lightcurves = partial_det.groupby('oid')
        with tf.io.TFRecordWriter(dest+'/chunk_{}.record'.format(counter)) as writer:
            Parallel(n_jobs=n_jobs)(delayed(process_lc)(obs, oid, ylabel, band, writer) \
                                    for oid, obs in lightcurves if obs.shape[0]>=5)


def create_dataset(meta_df,
                   source='data/raw_data/detections.csv',
                   target='data/records/macho/',
                   n_jobs=None,
                   subsets_frac=(0.5, 0.25),
                   max_lcs_per_record=100,
                   band=1,
                   debug=False):
    os.makedirs(target, exist_ok=True)

    if debug:
        detections = pd.read_csv(source, chunksize=1000)
        for det in detections:
            detections = det
            break
    else:
        detections = pd.read_csv(source)

    dist_labels = meta_df['classALeRCE'].value_counts().reset_index()
    unique = list(dist_labels['index'].unique())
    dist_labels.to_csv(os.path.join(target, 'objects.csv'), index=False)

    # Separate by class
    cls_groups = meta_df.groupby('classALeRCE')

    for cls_name, cls_meta in tqdm(cls_groups, total=len(cls_groups)):
        subsets = divide_training_subset(cls_meta,
                                         train=subsets_frac[0],
                                         val=subsets_frac[0])

        ylabel = unique.index(cls_name)

        for subset_name, frame in subsets:
            dest = os.path.join(target, subset_name, cls_name)
            os.makedirs(dest, exist_ok=True)
            write_records(frame, dest, max_lcs_per_record, detections,
                          ylabel, band, n_jobs)



def _decode(sample, max_obs=200):
    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'length': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = dict()
    for i in range(3):
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float32)

    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )

    input_dict = dict()
    input_dict['lcid']   = tf.cast(context['id'], tf.string)
    input_dict['label']  = tf.cast(context['label'], tf.int32)

    casted_inp_parameters = []
    for i in range(3):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)

    input_serie = tf.stack(casted_inp_parameters, axis=2)[0]

    # Sampling "max_obs" observations
    serie_len = tf.shape(input_serie)[0]
    curr_max_obs = tf.minimum(serie_len, max_obs)
    pivot = 0
    if tf.greater(serie_len, max_obs):
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-curr_max_obs,
                                  dtype=tf.int32)

        input_serie = tf.slice(input_serie, [pivot,0], [curr_max_obs, -1])
    else:
        input_serie = tf.slice(input_serie, [0,0], [curr_max_obs, -1])


    time_steps = tf.shape(input_serie)[0]
    mask = get_padding_mask(max_obs, tf.expand_dims(time_steps-1, 0))

    input_serie = standardize(input_serie, 0)
    if curr_max_obs < max_obs:
        filler    = tf.zeros([max_obs-time_steps, 3])
        input_serie  = tf.concat([input_serie, filler], 0)

    input_dict['values'] = input_serie
    input_dict['mask'] = 1. - tf.transpose(mask)

    return input_dict

def adjust_fn(func, max_obs):
    def wrap(*args, **kwargs):
        result = func(*args, max_obs)
        return result
    return wrap

def load_records(source, batch_size, max_obs=200, repeat=1):

    fn = adjust_fn(_decode, max_obs)

    objdf = pd.read_csv('/'.join(source.split('/')[:-1])+'/objects.csv')
    objdf['w']  = 1 - (objdf['classALeRCE'] - objdf['classALeRCE'].min())/(objdf['classALeRCE'].max()-objdf['classALeRCE'].min())
    repeats = [int(x*repeat) if x != 0 else 1 for x in objdf['w']]
    
    records_files = []
    for folder in os.listdir(source):
        for x in os.listdir(os.path.join(source, folder)):
            path = os.path.join(source, folder, x)
            records_files.append(path)

    datasets = [tf.data.TFRecordDataset(x) for x in records_files]

    datasets = [dataset.repeat(r) for dataset, r in zip(datasets, repeats)]
    datasets = [dataset.map(fn) for dataset in datasets]
    datasets = [dataset.shuffle(batch_size, reshuffle_each_iteration=True) for dataset in datasets]
    dataset = tf.data.experimental.sample_from_datasets(datasets)
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    return dataset
