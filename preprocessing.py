import os 
import tensorflow as tf


def preprocess_for_tfrecord(tfrecord):
    """ preprocess for reading
    """
    feature_descriptions = {
        "review": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "sentiment": tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    example = tf.io.parse_single_example(tfrecord, feature_descriptions)
    return example["review"], example["sentiment"]

def load_tfrecord(filepaths, n_read_threads=5, shuffle_buffer_size=10000,
                n_parse_threads=5, batch_size=32, cache=True):
    """ first convert the filepath into tfdataset, then shuffle the files
        then reading the data from files
        by using prefetch in -> make faster (prepare the next data even the current data not finishes)"""
    dataset = tf.data.TFRecordDataset(filepaths,
                                      num_parallel_reads=n_read_threads)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess_for_tfrecord, num_parallel_calls=n_parse_threads)
    return dataset.batch(batch_size).prefetch(1)      

def list_files_in_path(pathfile):
    files = []
    for file in os.listdir(pathfile):
        files.append(os.path.join(pathfile,file))
    return files
