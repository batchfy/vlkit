import sys, os
from os.path import dirname, join, abspath
import cv2
root_dir = abspath(join(dirname(__file__), "../"))
sys.path.insert(0, root_dir)
import vlkit
from vlkit.dense import sobel
import matplotlib.pyplot as plt

im = cv2.imread(join(root_dir, "data/images/2018.jpg"))
grad, grady, gradx = sobel(im)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(grad/grad.max())
axes[1].imshow(grady/grady.max())
axes[2].imshow(gradx/gradx.max())
plt.show()