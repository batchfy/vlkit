"""
https://github.com/tensorflow/models/blob/master/research/slim/datasets/build_imagenet_data.py
"""
import tensorflow as tf
from tensorflow.train import Example, Features, Feature, FloatList, BytesList, Int64List
from multiprocessing import Pool
import numpy as np
import argparse, os, sys
from os.path import join
from io import BytesIO
from os.path import dirname, abspath, join
import sys, os, shutil
from vlkit.io import bytes2array, array2bytes


def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--shards', type=int, default=36)
    parser.add_argument('--nproc', type=int, default=6)
    parser.add_argument('--output', type=str, default='tfrecords')
    args = parser.parse_args()
    assert args.shards % args.nproc == 0
    return args


def data2example(filename, label=0, **kwargs):
    image_raw = open(filename, mode='rb').read()
    example = Example(
        features=Features(feature={
            'filename': Feature(bytes_list=BytesList(value=[filename.encode('utf-8'),])),
            'image': Feature(bytes_list=BytesList(value=[image_raw,])),
            'attribute': Feature(bytes_list=BytesList(value=[array2bytes(np.random.random((10, 10))),])),
        })
    )
    return example


def batch_process(proc_index, ranges, filenames, args):
    os.makedirs(args.output, exist_ok=True)
    nproc = len(ranges)
    assert args.shards % nproc == 0, "%d %d" % (args.shards, nproc)
    shards_per_batch = int(args.shards / nproc)

    shard_ranges = np.linspace(ranges[proc_index][0],
                             ranges[proc_index][1],
                             shards_per_batch + 1).astype(int)
    samples_per_proc = ranges[proc_index][1] - ranges[proc_index][0]

    indices = []
    tfrecords = []

    for s in range(shards_per_batch):
        shard = proc_index * shards_per_batch + s

        tfrecord_filename = "{shard:05}-of-{shards:05}.tfrecord".format(shard=shard, shards=args.shards)
        tfrecord_filepath = join(args.output, tfrecord_filename)
        index_filepath = join(args.output, "%.5d-of-%.5d.index"% (shard, args.shards))
        writer = tf.io.TFRecordWriter(tfrecord_filepath)
        index = open(index_filepath, 'w')

        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        shard_offset = 0
        for i in files_in_shard:
            filename = filenames[i]
            example = data2example(filename)
            example_bytes = example.SerializeToString()
            writer.write(example_bytes)
            writer.flush()
            index.write('{filename} {shard:05} {offset}\n'
                .format(filename=tfrecord_filename, shard=shard, offset=shard_offset))
            shard_offset += (len(example_bytes) + 16)
            if i % 100 == 0:
                print("proc[{proc:03}/{nproc:03}]-shard[{shard:05}/{shards:05}]"
                    .format(proc=proc_index, nproc=args.nproc, shard=s, shards=shards_per_batch))
        writer.close()
        index.close()

        indices.append(index_filepath)
        tfrecords.append(tfrecord_filepath)
    return tfrecords, indices


if __name__ == "__main__":
    args = parse_args()
    filenames = ('../../data/images/2018.jpg',) * 10000
    spacing = np.linspace(0, len(filenames), args.nproc + 1).astype(np.int)

    """
    batch process samples with multi-process.
    each process deals with several shards (each shard corresponds to a tfrecord file).
    """
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    argin = []
    for proc_index in range(args.nproc):
        argin.append((proc_index, ranges, filenames, args))

    with Pool() as pool:
        results = pool.starmap(batch_process, argin)

    # merge indices
    indices = []
    for r in results:
        indices.extend(r[1])
    with open(join(args.output, 'all.index'), 'wb') as f:
        [shutil.copyfileobj(open(i, 'rb'), f) for i in indices]