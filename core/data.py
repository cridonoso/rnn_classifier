import tensorflow as tf
import os

from astromer.core.embedding import BASE_ASTROMER

astromer_emb = BASE_ASTROMER()

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
    std_value = tf.math.reduce_std(tensor, axis, name='std_value')
    if tf.rank(tensor) != tf.rank(mean_value):
        mean_value = tf.expand_dims(mean_value, axis)
        std_value = tf.expand_dims(std_value, axis)
    normed = tf.where(std_value == 0.,
                 (tensor - mean_value),
                 (tensor - mean_value)/std_value)
    return normed

def get_delta(tensor):
    tensor = tensor[1:] - tensor[:-1]
    tensor = tf.concat([tf.expand_dims([0.], 1), tensor], 0)
    return tensor

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
    curr_max_obs = tf.minimum(max_obs, serie_len)
    pivot = 0
    if tf.greater(serie_len, max_obs):
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-curr_max_obs,
                                  dtype=tf.int32)

    input_serie = tf.slice(input_serie, [pivot,0], [curr_max_obs, -1])

    input_dict['length'] = curr_max_obs
    input_dict['values'] = input_serie

    return input_dict

def _parse(input_dict):
    times = tf.slice(input_dict['values'], [0,0],[-1, 1])
    dtimes = get_delta(times)
    values = tf.slice(input_dict['values'], [0,1],[-1, 1])
    values = standardize(values)
    out = astromer_emb(times, values)
    out = standardize(out, axis=1)
    inputs = tf.concat([dtimes, values, out], 1)
    input_dict['values'] = inputs
    input_dict['times'] = times
    return input_dict

def _parse_2(input_dict):
    times  = tf.slice(input_dict['values'], [0,0],[-1, 1])
    dtimes = get_delta(times)
    values = tf.slice(input_dict['values'], [0,1],[-1, 1])
    values = standardize(values)
    std = tf.slice(input_dict['values'], [0,2],[-1, 1])
    std = standardize(std)
    inputs = tf.concat([dtimes, values, std], 1)
    input_dict['values'] = inputs
    input_dict['times'] = times
    return input_dict

def load_records(source, batch_size, max_obs=200, repeat=1, mode=0):

    if repeat != -1:
        datasets = [tf.data.TFRecordDataset(os.path.join(source, folder, x)) \
                    for folder in os.listdir(source) if not folder.endswith('.csv')\
                    for x in os.listdir(os.path.join(source, folder))]
        datasets = [dataset.map(lambda x: _decode(x, max_obs)) for dataset in datasets]
        if mode == 0:
            datasets = [dataset.map(_parse) for dataset in datasets]
        if mode == 1:
            datasets = [dataset.map(_parse_2) for dataset in datasets]
        datasets = [dataset.repeat(repeat) for dataset in datasets]
        datasets = [dataset.cache() for dataset in datasets]
        datasets = [dataset.shuffle(1000, reshuffle_each_iteration=True) for dataset in datasets]
        dataset = tf.data.experimental.sample_from_datasets(datasets)

    else: # TESTING LOADER
        datasets = [os.path.join(source, folder, x) \
                    for folder in os.listdir(source) if not folder.endswith('.csv')\
                    for x in os.listdir(os.path.join(source, folder))]
        dataset = tf.data.TFRecordDataset(datasets)
        dataset = dataset.map(lambda x: _decode(x, max_obs))
        if mode == 0:
            dataset = dataset.map(_parse)
        if mode == 1:
            dataset = dataset.map(_parse_2)

    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
