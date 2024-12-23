import matplotlib
import os, sys
from os.path import dirname, abspath, join
TEST_DIR = dirname(__file__)
sys.path.insert(0, abspath(join(TEST_DIR, "../")))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
from vlkit.geometry import random_perspective_matrix, warp_points_perspective

fig, axes = plt.subplots(1, 2)
im = cv2.imread(join(TEST_DIR, '../data/dog.jpg'))
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


h, w, _ = im.shape
npoints = 50
colors = np.random.uniform(0, 1, size=(npoints, 3))
points_x = np.random.uniform(10, w-10, size=(npoints, 1))
points_y = np.random.uniform(10, h-10, size=(npoints, 1))
points = np.concatenate((points_x, points_y), axis=1)

H, startpoints, endpoints = random_perspective_matrix(h, w, 0.4)

axes[0].imshow(im)
axes[0].scatter(points[:, 0], points[:, 1], color=colors, marker='x')
# axes[0].scatter(startpoints[:, 1], startpoints[:, 0], color='red')

imwarp = cv2.warpPerspective(im, H, (w, h))
axes[1].imshow(imwarp)

points1 = warp_points_perspective(points, transform_matrix=H)
axes[1].scatter(points1[:, 0], points1[:, 1], color=colors, marker='x')
# axes[1].scatter(endpoints[:, 1], endpoints[:, 0], color='red')
plt.tight_layout()

plt.savefig('perspective.jpg')