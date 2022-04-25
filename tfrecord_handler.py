import tensorflow as tf
import os
def create_example(review,sentiment):
    """Create Example for tfrecord to write"""
    review=review.encode("utf-8")
    return Example(
        features = Features(
            feature={
                "review": Feature(bytes_list=BytesList(value=[review])),
                "sentiment": Feature(int64_list=Int64List(value=[sentiment]))
            }
            ))

            

def write_tfrecords(data,name,n_parts):
    """ write the data in tf record by splitting the data in n_parts"""
    path = os.path.join("datasets\\tfrecords",name)
    os.makedirs(path, exist_ok=True)
    path_format = os.path.join(path, "{}-{:05d}.tfrecord")
    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_file = path_format.format(name,file_idx)
        filepaths.append(part_file)
        with tf.io.TFRecordWriter(part_file) as f:
            for row in row_indices:
                f.write(create_example(data[row][0],data[row][1]).SerializeToString())
    return filepaths




def preprocess_for_tfrecord(tfrecord):
    """ preprocess for reading
    """
    feature_descriptions = {
        "review": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "sentiment": tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    example = tf.io.parse_single_example(tfrecord, feature_descriptions)
    return example["review"], example["sentiment"]

def load_tfrecord(filepaths, n_read_threads=5, shuffle_buffer_size=None,
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


if __name__=="__main__":

    import numpy as np

    import pandas as pd
    from sklearn.model_selection import train_test_split
    Feature = tf.train.Feature
    Features = tf.train.Features
    Example = tf.train.Example
    BytesList = tf.train.BytesList
    Int64List = tf.train.Int64List



    """read the csv and splitting in train, validation and test set"""
    data = pd.read_csv("datasets\IMDB Dataset.csv")
    data["sentiment"][data["sentiment"]=="positive"]=1
    data["sentiment"][data["sentiment"]=="negative"]=0
    data.sentiment = data.sentiment.apply(pd.to_numeric)
    X_train, X_test, y_train, y_test = train_test_split(data["review"],data["sentiment"],test_size=0.1,random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=5000)

    train_data = np.c_[X_train,y_train]
    test_data = np.c_[X_test,y_test]
    valid_data = np.c_[X_valid,y_valid]
    write_tfrecords(train_data,"train",20)
    write_tfrecords(test_data,"test",5)
    write_tfrecords(valid_data,"validation",5)









