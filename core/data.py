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

def create_mask(tensor, lengths):
    ''' Create mask given a tensor and true length '''
    lengths_transposed = lengths
    rangex = tf.range(0, tf.shape(tensor)[0], 1)
    range_row = tf.expand_dims(rangex, 0)
    # Use the logical operations to create a mask
    mask = tf.less(range_row, lengths_transposed)
    return tf.transpose(mask)


def _parse(sample, max_obs=200):
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

    serie_len = tf.shape(input_serie)[0]
    curr_max_obs = tf.minimum(max_obs, serie_len)

    pivot = 0
    if tf.greater(serie_len, max_obs):
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-curr_max_obs,
                                  dtype=tf.int32)

    input_serie = tf.slice(input_serie, [pivot,0], [curr_max_obs, -1])

    times = tf.slice(input_serie, [0,0],[-1, 1])
    values = tf.slice(input_serie, [0,1],[-1, 1])

    out = astromer_emb(values, times)

    inputs = tf.concat([times, out], 1)
    mask = create_mask(inputs, max_obs)

    input_dict['values'] = inputs
    input_dict['mask'] = mask
    return input_dict

def load_records(source, batch_size, max_obs=200, repeat=1):
    datasets = [tf.data.TFRecordDataset(os.path.join(source, folder, x)) \
                        for folder in os.listdir(source) if not folder.endswith('.csv')\
                        for x in os.listdir(os.path.join(source, folder))]
    datasets = [dataset.map(lambda x: _parse(x, max_obs)) for dataset in datasets]
    datasets = [dataset.repeat(repeat) for dataset in datasets]
    datasets = [dataset.cache() for dataset in datasets]
    datasets = [dataset.shuffle(1000, reshuffle_each_iteration=True) for dataset in datasets]
    dataset = tf.data.experimental.sample_from_datasets(datasets)
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
