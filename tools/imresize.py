import os, sys, argparse
from PIL import Image
import os.path as osp


__exts__ = ['.jpg', '.png']


parser = argparse.ArgumentParser()
parser.add_argument('target', type=str, help='target to be resized')
parser.add_argument('--h', type=float, default=0)
parser.add_argument('--w', type=float, default=0)
parser.add_argument('--scale', type=float, default=1)
args = parser.parse_args()


assert bool(args.h) + bool(args.w) + bool(args.scale) == 1, bool(args.h) + bool(args.w) + bool(args.scale)


if osp.isfile(args.target):
    Image.open(args.target)
elif osp.isdir(args.target):
    imgs = [osp.join(args.target, i) for i in os.listdir(args.target) if osp.splitext(i)[-1].lower() in __exts__]
    for i in imgs:
        im = Image.open(i)
        w, h = im.size
        if args.scale != 0:
            new_h= int(h * args.scale)
            new_w = int(w * args.scale)
            im.resize((new_w, new_h))
            fn, ext = osp.splitext(i)
            im.save(fn + '-resize%dx%d' % (new_w, new_h) + ext)
            



