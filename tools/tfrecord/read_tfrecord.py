import matplotlib
import torch
import tensorflow as tf
import os, sys, struct
import example_pb2
import io
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
from vlkit.io import bytes2array, array2bytes


f = open('tfrecords/00000-of-00012.tfrecord', 'rb')
# here o is the offset inside the tfrecord.
f.seek(0)
byte_len_crc = f.read(12)
proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
pb_data = f.read(proto_len)

example = example_pb2.Example()
example.ParseFromString(pb_data)


for k, v in example.features.feature.items():
    if k == 'filename':
        filename = v.bytes_list.value[0].decode('utf-8')
        print(filename)
    elif k == 'image':
        image = np.array(Image.open(io.BytesIO(v.bytes_list.value[0])))
        print(image.shape)
    elif k == 'attribute':
        # attribute = np.frombuffer(v.bytes_list.value[0], dtype=np.float32)
        array = bytes2array(v.bytes_list.value[0])
        print(k, array.shape, array.dtype)
    elif k == 'shape':
        print(k, v.bytes_list.value)