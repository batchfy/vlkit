import os, sys
from os.path import join, abspath, dirname, isdir, split
root = abspath(join(dirname(__file__), '..'))
if not isdir(join(root, 'data/edges-nms')):
  os.makedirs(join(root, 'data/edges-nms'))
from scipy import io
from PIL import Image
import numpy as np
from vlkit import nms

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm

imgs = [i for i in os.listdir(join(root, 'data/edges/')) if ".png" in i]

fig, axes = plt.subplots(4, 3, figsize=(6, 8))

for ax in axes.flatten():
  ax.set_xticks([])
  ax.set_yticks([])

for idx, i in enumerate(imgs):
    I = np.array(Image.open(join(root, "data/images", i.replace("png", "jpg"))))
    E = np.array(Image.open(join(root, "data/edges", i)))
    E1 = nms.nms(E)
    E1 = Image.fromarray((E1 * 255).astype(np.uint8))
    E1.save(join(root, 'data/edges-nms', i))

    axes[idx,0].imshow(I)
    axes[idx,1].imshow(E, cmap=cm.Greys_r)
    axes[idx,2].imshow(E1, cmap=cm.Greys_r)

axes[0, 0].set_title("Image")
axes[0, 1].set_title("Edge")
axes[0, 2].set_title("Edge NMS")

plt.tight_layout()
plt.savefig("nms-results.png")
