## TFRecord

Serialize everything into tfrecord and read into python objects.

This is very helpfull when training with large datasets and data IO beccomes
the bottleneck to performance.

## List of contents

* [create_tfrecord.py](create_tfrecord.py): create tfrecord from image and ndarray data.
* [read_tfrecord.py](read_tfrecord.py): read data from tfrecord.
* [tfrecord_dataset.py](tfrecord_dataset): an example torch dataset that reads data from tfrecords.


## Create tfrecord
basic concepts:
* example: an example is a data sample/instance usually contains an image, some attribute (e.g. semantic mask in semantic segmentation, or label in image classification).
* shards: shards are used to split large-scale data into parts. In our case, a shard corresponds to a tfrecord file.
* index: index stores the `shard id`, `offset` of an example.


1. Run `python create_tfrecord.py` to create example tfrecords. 
2. concatenate indices via `cat tfrecords/*-*.index > tfrecords/all.index`.


## Read tfrecord
Run `python read_tfrecord.py` to read examples from tfrecords.

## tfrecord as PyTorch dataset
`tfrecord_dataset.py` is an examplar dataset uses tfrecord as file backend.