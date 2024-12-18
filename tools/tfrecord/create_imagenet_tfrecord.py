"""
https://github.com/tensorflow/models/blob/master/research/slim/datasets/build_imagenet_data.py
"""
import tensorflow as tf
from tensorflow.train import Example, Features, Feature, BytesList
import numpy as np
import argparse, os, shutil
from multiprocessing import Pool
import os
from os.path import join

from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--root', type=str)
    parser.add_argument('--shards', type=int, default=12)
    parser.add_argument('--nproc', type=int, default=6)
    parser.add_argument('--output', type=str, default='tfrecords')
    parser.add_argument('--silent', action='store_true')
    args = parser.parse_args()
    assert args.shards % args.nproc == 0
    return args


def data2example(filename, label):
    image_raw = open(filename, mode='rb').read()
    filename = join(*filename.split(os.sep)[-3:])
    example = Example(
        features=Features(feature={
            'filename': Feature(bytes_list=BytesList(value=[filename.encode('utf-8'),])),
            'image': Feature(bytes_list=BytesList(value=[image_raw,])),
            'label': Feature(bytes_list=BytesList(value=[int(label).to_bytes(16, 'big')])),
        })
    )
    return example


def batch_process(proc_index, ranges, filenames, labels, args):
    os.makedirs(args.output, exist_ok=True)
    nproc = len(ranges)
    assert not args.shards % nproc, "%d %d" % (args.shards, nproc)
    shards_per_proc = int(args.shards / nproc)

    shard_ranges = np.linspace(ranges[proc_index][0],
                             ranges[proc_index][1],
                             shards_per_proc + 1).astype(int)
    indices = []
    tfrecords = []

    for s in range(shards_per_proc):
        shard = proc_index * shards_per_proc + s

        tfrecord_filename = join(args.output, "%.5d-of-%.5d.tfrecord"% (shard, args.shards))
        index_filename = join(args.output, "%.5d-of-%.5d.index"% (shard, args.shards))
        writer = tf.io.TFRecordWriter(tfrecord_filename)
        index = open(index_filename, 'w')

        shard_counter = 0
        samples_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        shard_offset = 0
        for i in samples_in_shard:
            filename = filenames[i]
            label = labels[i]
            example = data2example(filename, label)
            example_bytes = example.SerializeToString()
            writer.write(example_bytes)
            index.write('{} {} {} {}\n'.format(filename, shard, "%.5d-of-%.5d.tfrecord"% (shard, args.shards), shard_offset))
            writer.flush()
            index.flush()

            shard_offset += (len(example_bytes) + 16)
            shard_counter += 1

            if (i-samples_in_shard[0]) % 1000 == 0:
                print("proc[{proc:03}/{nproc:03}]-shard[{shard:05}/{shards:05}] samples [{i}/{samples}]"
                    .format(proc=proc_index, nproc=args.nproc, shard=s, shards=shards_per_proc,
                    i=i-samples_in_shard[0], samples=samples_in_shard[-1]-samples_in_shard[0]))
        writer.close()
        index.close()

        indices.append(index_filename)
        tfrecords.append(tfrecord_filename)
    return tfrecords, indices


if __name__ == "__main__":
    args = parse_args()
    train_ids = glob(join(args.root, 'train') + '/*/')
    val_ids = [i.replace('/train/', '/val/') for i in train_ids]
    assert len(train_ids) == 1000, str(len(train_ids))
    filenames = []
    labels = []
    for idx, i in enumerate(train_ids):
        imgs = glob(i + '*.*')
        filenames.extend(imgs)
        labels += [idx,] * len(os.listdir(i))
    assert len(filenames) > 0, str(len(filenames))
    print("%d samples in total." % len(filenames))

    spacing = np.linspace(0, len(filenames), args.nproc + 1).astype(np.int)

    """
    batch process samples with multi-thread.
    each thread process several shards (each shard corresponds to a tfrecord file).
    """
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    argin = []
    for proc_index in range(args.nproc):
        argin.append((proc_index, ranges, filenames, labels, args))

    with Pool() as pool:
        results = pool.starmap(batch_process, argin)

    # merge indices
    indices = []
    for r in results:
        indices.extend(r[1])
    with open(join(args.output, 'train.index'), 'wb') as f:
        [shutil.copyfileobj(open(i, 'rb'), f) for i in indices]

    # for val set
    writer = tf.io.TFRecordWriter(join(args.output, 'val.tfrecord'))
    index = open(join(args.output, 'val.index'), 'w')
    shard_offset = 0
    for idx, i in enumerate(val_ids):
        imgs = glob(i + '*.*')
        for im in imgs:
            example = data2example(im, idx)
            example_bytes = example.SerializeToString()
            writer.write(example_bytes)
            index.write('{} {} {} {}\n'.format(im, 0, 'val.tfrecord', shard_offset))
            shard_offset += (len(example_bytes) + 16)
    writer.close()
    index.close()
