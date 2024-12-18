import torch
import numpy as np
import os, sys
from os.path import join, isfile, isdir, isfile
from collections import defaultdict
from torchvision.datasets.folder import pil_loader
import struct, example_pb2, io
from vlkit.io import bytes2array, bytes2image2array
from PIL import Image


def parse_example(f, offset):
    f = open(f, 'rb')
    f.seek(offset)
    byte_len_crc = f.read(12)
    proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
    pb_data = f.read(proto_len)
    example = example_pb2.Example()
    example.ParseFromString(pb_data)
    example = {k: v.bytes_list.value[0] for k, v in example.features.feature.items()}
    return example


class TFRecordDataset(torch.utils.data.Dataset):

    def __init__(self, root, index='all.index', transform=None):
        self.root = root
        index = join(self.root, index)
        assert isfile(index)
        self.indices = [i.strip().split(" ") for i in open(index, 'r')]
        self.transform = transform

    def __getitem__(self, i):
        ind = self.indices[i]
        tfrecord_filename, shard_index, offset = ind
        offset = int(offset)
        items = parse_example(join(self.root, tfrecord_filename), offset)

        data = dict()
        data['filename'] = items['filename'].decode('utf-8')
        data['attribute'] = bytes2array(items['attribute'])
        data['image'] = bytes2image2array(items['image'])

        if self.transform is not None:
            data['image'] = self.transform(data['image'])
        return data

    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':
    dataset = TFRecordDataset('tfrecords')

    for d in dataset:
        print(d)
        sys.exit()
