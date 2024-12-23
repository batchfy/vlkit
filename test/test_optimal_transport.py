import os.path as osp
TEST_DIR = osp.dirname(__file__)
import sys, os
sys.path.insert(0, osp.abspath(osp.join(TEST_DIR, "../")))
from vlkit.optimal_transport import sinkhorn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import numpy as np
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

savedir = osp.join(osp.dirname(__file__), "../data/test")
os.makedirs(savedir, exist_ok=True)

n = 100  # nb bins
# bin positions
x = np.arange(n, dtype=np.float64)
# Gaussian distributions
a = gauss(n, m=np.random.uniform(10, 90), s=np.random.uniform(1, 20))  # m= mean, s= std
b = gauss(n, m=np.random.uniform(10, 90), s=np.random.uniform(1, 20))

d1, d2 = a.shape[0], b.shape[0]

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()

lambd = 1e-3
T_ot = ot.sinkhorn(a, b, M, lambd, verbose=False)

T, _, _ = sinkhorn(
    torch.from_numpy(a).view(1, -1),
    torch.from_numpy(b).view(1, -1),
    torch.from_numpy(M).unsqueeze(dim=0),
    num_iters=1000,
    error_thres=1e-9,
    reg=lambd,
)

plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(3, 6)

ax1 = plt.subplot(gs[0, 1:3])
plt.plot(np.arange(b.size), b, 'r', label='Target distribution')
ax2 = plt.subplot(gs[1:, 0])
plt.plot(b, x, 'b', label='Source distribution')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.subplot(gs[1:3, 1:3], sharex=ax1, sharey=ax2)
plt.imshow(T_ot)
plt.axis('off')

ax1 = plt.subplot(gs[0, 4:])
plt.plot(np.arange(b.size), b, 'r', label='Target distribution')
ax2 = plt.subplot(gs[1:, 3])
plt.plot(b, x, 'b', label='Source distribution')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.subplot(gs[1:3, 4:], sharex=ax1, sharey=ax2)
plt.imshow(T.squeeze(dim=0).numpy())
plt.axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0., hspace=0.2)
plt.savefig(osp.join(savedir, 'pth_sinkhorm.png'))
