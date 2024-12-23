from PIL import Image
import os, sys
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
root_dir = abspath(join(dirname(__file__), "../"))
sys.path.insert(0, root_dir)
from vlkit.transforms.resize import Resize


im = Image.open(join(root_dir, "data/images/2018.jpg"))
t = Resize(size=500, backend="random", interpolation="random")
print(t)
fig, axes = plt.subplots(3, 5)

for ax in axes.flatten():
    ax.imshow(t(im))
plt.savefig(join(root_dir, "test_resize.jpg"))
