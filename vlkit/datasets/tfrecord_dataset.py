import torch
import numpy as np
from os.path import join, isfile, isdir, isfile
from collections import defaultdict
from torchvision.datasets.folder import pil_loader
import struct, io
from .example_pb2 import Example
from vlkit.io import bytes2image2array, bytes2image
from PIL import Image


def parse_example(f, offset):
    with open(f, 'rb') as f:
        f.seek(int(offset))
        byte_len_crc = f.read(12)
        proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
        pb_data = f.read(proto_len)
    example = Example()
    example.ParseFromString(pb_data)
    example = {k: v.bytes_list.value[0] for k, v in example.features.feature.items()}
    return example


class TFRecordDataset(torch.utils.data.Dataset):
    """
    TFRecord dataset for classification tasks.
    Refer to <https://github.com/vlkit/vlkit/blob/master/tools/tfrecord/create_imagenet_tfrecord.py>
    for how to create tfrecords for imagenet dataset.
    """

    def __init__(self, root, index, transform=None):
        self.root = root
        index = join(self.root, index)

        if not isfile(index):
            raise FileNotFoundError(index)

        with open(index, 'r') as f:
            self.items = [i.strip().split(" ") for i in f.readlines()]
        self.transform = transform

    def __getitem__(self, i):
        ind = self.items[i]
        imfilename, shard, tfrecord_filename, offset = ind
        items = parse_example(join(self.root, tfrecord_filename), offset)

        data = dict()
        data['filename'] = items['filename'].decode('utf-8')
        data['image'] = bytes2image(items['image']).convert('RGB')
        data['label'] = int.from_bytes(items['label'], byteorder='big')

        if self.transform is not None:
            data['image'] = self.transform(data['image'])
        return data

    def __len__(self):
        return len(self.items)
